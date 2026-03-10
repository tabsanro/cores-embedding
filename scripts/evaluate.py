"""
scripts/evaluate.py — CoRes-Embedding 평가 스크립트

새 평가기를 추가하는 방법
──────────────────────────
1. evaluation/ 에 평가기 클래스를 구현합니다.
   (run_full_evaluation 메서드 필요)
2. EVALUATOR_REGISTRY에 항목을 추가합니다:
       "my_eval": EvaluatorSpec(
           cls=MyEvaluator,
           run_fn=lambda e, trl, tel, cfg: e.run_full_evaluation(tel, cfg),
           help="짧은 설명",
       )
3. --evaluators my_eval 으로 바로 실행할 수 있습니다.

사용 예시
─────────
    # 기본 설정 폴더에서 모든 모델 평가 (perturbation만)
    python scripts/evaluate.py

    # 특정 실험 폴더의 cores 모델만 평가
    python scripts/evaluate.py --exp-dir outputs/backbone_sweep_.../resnet34 --model cores

    # 전체 평가기 사용
    python scripts/evaluate.py --evaluators perturbation fewshot manifold

    # 빠른 확인용 (perturbation + fewshot)
    python scripts/evaluate.py --evaluators perturbation fewshot --model baseline
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Callable

import torch

from _utils import (
    _ROOT,
    dump_json,
    free_memory,
    json_safe,
    load_checkpoint,
    load_config,
    print_param_summary,
    resolve_device,
)

sys.path.insert(0, _ROOT)

from data import get_dataloaders
from models import build_model
from visualization.plots import ResultsPlotter


# ---------------------------------------------------------------------------
# 평가기 레지스트리
# ─────────────────────────────────────────────────────────────────────────────
# 새 평가기를 등록하려면 EvaluatorSpec을 작성해 EVALUATOR_REGISTRY에 추가하세요.
# ---------------------------------------------------------------------------

@dataclass
class EvaluatorSpec:
    """평가기 사양 — 클래스, 실행 함수, 설명을 묶습니다."""
    cls: type                    # 평가기 클래스
    run_fn: Callable             # run_fn(evaluator, train_loader, test_loader, config) → results
    help: str = ""               # --help에 표시될 설명


def _lazy_registry() -> dict[str, EvaluatorSpec]:
    """평가기 레지스트리를 반환합니다 (지연 import로 선택적 의존성 지원)."""
    from evaluation.perturbation import PerturbationEvaluator
    from evaluation.fewshot import FewShotEvaluator
    from evaluation.manifold import ManifoldEvaluator

    return {
        "perturbation": EvaluatorSpec(
            cls=PerturbationEvaluator,
            run_fn=lambda e, trl, tel, cfg: e.run_full_evaluation(tel, cfg),
            help="가우시안·적대적·PGD 노이즈에 대한 섭동 안정성 평가",
        ),
        "fewshot": EvaluatorSpec(
            cls=FewShotEvaluator,
            run_fn=lambda e, trl, tel, cfg: e.run_full_evaluation(trl, tel, cfg),
            help="임베딩 기반 퓨샷 분류 정확도 평가",
        ),
        "manifold": EvaluatorSpec(
            cls=ManifoldEvaluator,
            run_fn=lambda e, trl, tel, cfg: e.run_full_evaluation(tel, cfg),
            help="잠재 공간 다양체 부드러움(보간 선형성) 평가",
        ),
        # 새 평가기 추가 예시:
        # "disentanglement": EvaluatorSpec(
        #     cls=DisentanglementEvaluator,
        #     run_fn=lambda e, trl, tel, cfg: e.run_full_evaluation(tel, cfg),
        #     help="개념-잠재 벡터 분리도(DCI, MIG 등) 평가",
        # ),
    }


# ---------------------------------------------------------------------------
# 핵심 함수 (sweep 스크립트에서 재사용 가능)
# ---------------------------------------------------------------------------

def load_model_for_eval(
    config: dict,
    model_type: str,
    num_concepts: int,
    checkpoint_path: str,
) -> torch.nn.Module:
    """모델을 빌드하고 체크포인트를 불러옵니다."""
    config["_model_type"] = model_type
    for key in ("cores", "vcores"):
        if model_type == key:
            config["model"].setdefault(key, {})["num_concepts"] = num_concepts

    model = build_model(config, num_concepts)
    print_param_summary(model)
    load_checkpoint(model, checkpoint_path)
    return model


def run_evaluation(
    model: torch.nn.Module,
    config: dict,
    evaluator_names: list[str],
    train_loader,
    test_loader,
) -> dict:
    """지정된 평가기를 순서대로 실행하고 결과를 반환합니다.

    Args:
        model:           이미 디바이스에 올라가 있지 않아도 됩니다 (평가기가 처리).
        config:          설정 dict.
        evaluator_names: EVALUATOR_REGISTRY 키 목록.
        train_loader:    학습 데이터 로더 (퓨샷 평가에 필요).
        test_loader:     검증 데이터 로더.

    Returns:
        {evaluator_name: results_dict, ...}
    """
    registry = _lazy_registry()
    device = config["experiment"].get("device", "cpu")
    results: dict = {}

    for name in evaluator_names:
        if name not in registry:
            print(f"  [SKIP] 알 수 없는 평가기: '{name}'. "
                  f"사용 가능: {list(registry.keys())}")
            continue

        spec = registry[name]
        print(f"  → {name} ({spec.cls.__name__})")
        evaluator = None
        try:
            evaluator = spec.cls(model, device=device)
            results[name] = spec.run_fn(evaluator, train_loader, test_loader, config)
        except Exception as exc:
            results[name] = {"error": str(exc)}
            print(f"     [ERROR] {exc}")
        finally:
            if evaluator is not None:
                evaluator.model = None
                del evaluator
            free_memory()

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    all_evaluator_names = list(_lazy_registry().keys())  # 도움말용 사전 로드

    p = argparse.ArgumentParser(
        description="CoRes-Embedding 평가 스크립트",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--config", default="configs/default.yaml",
                   help="기본 설정 파일 (기본값: configs/default.yaml)")
    p.add_argument("--model", nargs="+",
                   default=["baseline", "cores"],
                   metavar="MODEL",
                   help="평가할 모델 타입 (기본값: baseline cores)")
    p.add_argument("--evaluators", nargs="+",
                   default=["perturbation"],
                   choices=all_evaluator_names + ["all"],
                   metavar="EVAL",
                   help=("실행할 평가기 목록. 'all' 지정 시 전체 실행.\n"
                         f"사용 가능: {', '.join(all_evaluator_names)}\n"
                         "(기본값: perturbation)"))
    p.add_argument("--checkpoint", default="best.pt",
                   help="불러올 체크포인트 파일명 (기본값: best.pt)")
    p.add_argument("--exp-dir", default=None,
                   help=("실험 폴더 직접 지정. 예: outputs/backbone_sweep_.../resnet34\n"
                         "지정 시 폴더명을 backbone으로 자동 인식합니다."))
    p.add_argument("--device", default=None,
                   help="디바이스 오버라이드 (cuda / cpu)")
    p.add_argument("--output-dir", default=None,
                   help="결과 저장 폴더 오버라이드")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    config = load_config(args.config)

    # 디바이스
    if args.device:
        config["experiment"]["device"] = args.device
    config["experiment"]["device"] = resolve_device(config["experiment"]["device"])

    # 평가기 목록 결정
    registry = _lazy_registry()
    evaluator_names = (
        list(registry.keys()) if "all" in args.evaluators else args.evaluators
    )

    # 평가 결과 및 그래프 저장 폴더 결정
    if args.exp_dir and os.path.isdir(args.exp_dir):
        exp_dir = os.path.abspath(args.exp_dir)
        backbone = os.path.basename(exp_dir)
        print(f"  backbone 자동 감지: {backbone}")

        if isinstance(config["model"]["backbone"], dict):
            config["model"]["backbone"]["name"] = backbone
        else:
            config["model"]["backbone"] = backbone

        eval_output_dir = args.output_dir or os.path.join(exp_dir, "evaluation")

        def _ckpt_path(model_type: str) -> str:
            return os.path.join(exp_dir, model_type, "checkpoints", args.checkpoint)
    else:
        exp = config["experiment"]
        eval_output_dir = args.output_dir or os.path.join(
            exp["output_dir"], exp["name"], "evaluation"
        )

        def _ckpt_path(model_type: str) -> str:
            return os.path.join(
                exp["output_dir"], exp["name"], model_type,
                "checkpoints", args.checkpoint,
            )

    os.makedirs(eval_output_dir, exist_ok=True)

    print("=" * 60)
    print("CoRes-Embedding 평가")
    print("=" * 60)
    print(f"  모델:      {', '.join(args.model)}")
    print(f"  평가기:    {', '.join(evaluator_names)}")
    print(f"  디바이스:  {config['experiment']['device']}")
    print(f"  결과 폴더: {eval_output_dir}")
    print("=" * 60)

    print("\n데이터셋 로딩 중...")
    train_loader, test_loader, num_concepts = get_dataloaders(config)

    all_results: dict = {}
    for model_type in args.model:
        print(f"\n[{model_type.upper()}]")
        ckpt_path = _ckpt_path(model_type)
        model = load_model_for_eval(config, model_type, num_concepts, ckpt_path)
        all_results[model_type] = run_evaluation(
            model, config, evaluator_names, train_loader, test_loader
        )
        model.cpu()
        del model
        free_memory()

    # 결과 저장
    results_path = os.path.join(eval_output_dir, "results.json")
    dump_json(results_path, all_results)
    print(f"\n결과 저장: {results_path}")

    # 시각화
    vis = config.get("visualization", {})
    plotter = ResultsPlotter(
        output_dir=os.path.join(eval_output_dir, "figures"),
        save_format=vis.get("save_format", "pdf"),
        dpi=vis.get("dpi", 300),
        figsize=tuple(vis.get("figsize", [8, 6])),
    )
    plotter.plot_all_results(all_results)
    print("평가 완료!")
    return all_results


if __name__ == "__main__":
    main()
