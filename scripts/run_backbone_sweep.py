"""
scripts/run_backbone_sweep.py — 백본 × 모델 스윕 실험

train.py와 evaluate.py의 핵심 함수를 재사용하므로
학습·평가 로직이 단일 지점에서 관리됩니다.

스윕 설정 파일(--sweep-config) 예시
──────────────────────────────────────
  # configs/sweep_example.yaml
  backbones: [resnet18, resnet34, vit_b_16]
  models: [baseline, cores]
  epochs: 50
  evaluate: true
  evaluators: [perturbation]

사용 예시
─────────
    # 기본 실행 (resnet18 & resnet50 × baseline & cores)
    python scripts/run_backbone_sweep.py

    # 스윕 설정 파일 사용
    python scripts/run_backbone_sweep.py --sweep-config configs/sweep_example.yaml

    # CLI에서 직접 지정
    python scripts/run_backbone_sweep.py --backbones resnet34 vit_b_16 --models cores --epochs 30

    # 체크포인트가 있으면 학습 건너뛰기 + 평가만
    python scripts/run_backbone_sweep.py --eval-only --evaluators perturbation fewshot

    # Dry-run으로 실행 계획 확인
    python scripts/run_backbone_sweep.py --dry-run
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
import traceback
from datetime import datetime

import matplotlib
import numpy as np
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from _utils import (
    _ROOT,
    dump_json,
    free_memory,
    load_config,
    resolve_device,
    set_seed,
)

sys.path.insert(0, _ROOT)

# train.py / evaluate.py에서 핵심 함수를 직접 import
from train import build_and_train, SUPPORTED_MODELS as _SUPPORTED_MODELS
from evaluate import load_model_for_eval, run_evaluation
from data import get_dataloaders

# ---------------------------------------------------------------------------
# 지원 목록
# ---------------------------------------------------------------------------
SUPPORTED_BACKBONES: list[str] = [
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
    "densenet121", "densenet169", "densenet201", "densenet161",
    "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3",
    "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7",
    "vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32", "vit_h_14",
]
SUPPORTED_MODELS: list[str] = _SUPPORTED_MODELS


# ---------------------------------------------------------------------------
# 설정 조립
# ---------------------------------------------------------------------------

def _make_run_config(
    base_config: dict,
    backbone: str,
    model_type: str,
    sweep_output_dir: str,
    epochs_override: int | None,
) -> dict:
    """베이스 설정을 복사하고 backbone, 모델 타입, 출력 경로를 주입합니다."""
    cfg = copy.deepcopy(base_config)
    cfg["model"]["backbone"] = backbone
    cfg["_model_type"] = model_type
    cfg["experiment"]["output_dir"] = os.path.join(sweep_output_dir, backbone)
    cfg["experiment"]["name"] = ""        # 출력 경로: sweep_dir/backbone/model_type/
    if epochs_override is not None:
        cfg["training"]["epochs"] = epochs_override
    if model_type in ("cores", "vcores"):
        cfg["model"].setdefault(model_type, {})["num_concepts"] = -1
    return cfg


# ---------------------------------------------------------------------------
# 요약 테이블 / 저장 / 그래프
# ---------------------------------------------------------------------------

def _safe_get(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
        if d is None:
            return None
    return d


def build_summary_table(all_results: dict) -> list[dict]:
    """all_results에서 비교 테이블 행을 생성합니다."""
    rows = []
    for backbone, models in all_results.items():
        for model_type, res in models.items():
            if "error" in res:
                rows.append({
                    "backbone": backbone, "model": model_type,
                    "status": "FAILED", "error": res["error"],
                })
                continue
            ev = res.get("evaluation", {})
            rows.append({
                "backbone": backbone,
                "model": model_type,
                "status": "OK",
                "train_time_s": res.get("train_time_s"),
                "best_val_loss": res.get("best_val_loss"),
                "fewshot_acc_mean": _safe_get(ev, "fewshot", "mean_accuracy"),
                "perturbation_robustness": _safe_get(ev, "perturbation", "robustness_score"),
            })
    return rows


def save_summary(sweep_output_dir: str, all_results: dict, table_rows: list[dict]) -> str:
    """요약 JSON과 Markdown을 summary/ 폴더에 저장합니다."""
    summary_dir = os.path.join(sweep_output_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    dump_json(os.path.join(summary_dir, "all_results.json"), all_results)

    def _fmt(v, fmt=".4f"):
        return f"{v:{fmt}}" if isinstance(v, float) else (str(v) if v is not None else "-")

    lines = [
        "# Backbone Sweep 실험 결과\n",
        f"생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "| Backbone | Model | Status | Train Time (s) | Best Val Loss | FewShot Acc | Perturbation Robustness |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ] + [
        f"| {r['backbone']} | {r['model']} | {r['status']}"
        f" | {_fmt(r.get('train_time_s'), '.1f')} | {_fmt(r.get('best_val_loss'))}"
        f" | {_fmt(r.get('fewshot_acc_mean'))} | {_fmt(r.get('perturbation_robustness'))} |"
        for r in table_rows
    ]

    md_path = os.path.join(summary_dir, "summary.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\n[요약] {os.path.join(summary_dir, 'all_results.json')}")
    print(f"[요약] {md_path}")
    return summary_dir


def plot_comparison(sweep_output_dir: str, table_rows: list[dict]) -> None:
    """각 지표에 대한 grouped bar chart를 생성합니다."""
    figures_dir = os.path.join(sweep_output_dir, "summary", "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # 그릴 지표 목록 — 새 지표를 추가하려면 여기에 추가하세요
    metrics = [
        ("best_val_loss",            "Best Val Loss (↓)"),
        ("fewshot_acc_mean",         "Few-shot Acc Mean (↑)"),
        ("perturbation_robustness",  "Perturbation Robustness (↑)"),
        ("train_time_s",             "Train Time (s)"),
    ]

    ok_rows = [r for r in table_rows if r["status"] == "OK"]
    if not ok_rows:
        print("[그래프] 성공한 실험 결과가 없어 그래프를 생성하지 않습니다.")
        return

    backbones = sorted({r["backbone"] for r in ok_rows})
    model_types = sorted({r["model"] for r in ok_rows})
    palette = sns.color_palette("Set2", len(model_types))
    x = np.arange(len(backbones))
    width = 0.8 / max(len(model_types), 1)
    offsets = (
        np.linspace(-(len(model_types) - 1) / 2, (len(model_types) - 1) / 2, len(model_types))
        * width
    )

    for mkey, mlabel in metrics:
        vals = {(r["backbone"], r["model"]): r.get(mkey) for r in ok_rows}
        if all(v is None for v in vals.values()):
            continue

        fig, ax = plt.subplots(figsize=(max(6, len(backbones) * 1.5 + 2), 5))
        for i, (mt, color) in enumerate(zip(model_types, palette)):
            heights = [vals.get((bb, mt)) for bb in backbones]
            bars = ax.bar(
                x + offsets[i], [h or 0 for h in heights],
                width, label=mt, color=color, alpha=0.85, edgecolor="white",
            )
            for bar, h in zip(bars, heights):
                if h is None:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.002, "N/A",
                        ha="center", va="bottom", fontsize=7, color="grey",
                    )

        ax.set(
            xlabel="Backbone", ylabel=mlabel,
            title=f"Backbone Sweep: {mlabel}",
            xticks=x, xticklabels=backbones,
        )
        ax.set_xticklabels(backbones, rotation=30, ha="right")
        ax.legend(title="Model")
        plt.tight_layout()

        fname = mkey.replace("/", "_")
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(figures_dir, f"{fname}.{ext}"), dpi=150)
        plt.close(fig)
        print(f"[그래프] {mkey} → {figures_dir}")


def _print_summary_table(rows: list[dict]) -> None:
    """콘솔에 요약 테이블을 출력합니다."""
    def _f(v, fmt=".4f"):
        return f"{v:{fmt}}" if isinstance(v, (int, float)) else "-"

    header = (
        f"{'Backbone':<22} {'Model':<12} {'Status':<8} "
        f"{'Train(s)':>9} {'ValLoss':>9} {'FewAcc':>8} {'Robust':>8}"
    )
    sep = "─" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for r in rows:
        print(
            f"{r['backbone']:<22} {r['model']:<12} {r['status']:<8} "
            f"{_f(r.get('train_time_s'), '9.1f')} {_f(r.get('best_val_loss'), '9.4f')} "
            f"{_f(r.get('fewshot_acc_mean'), '8.4f')} {_f(r.get('perturbation_robustness'), '8.4f')}"
        )
    print(sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Multi-backbone sweep: baseline vs CoRes 계열 모델 비교",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--config", default="configs/default.yaml",
                   help="기본 설정 파일 (기본값: configs/default.yaml)")
    p.add_argument("--sweep-config", default=None,
                   help="스윕 전용 YAML 파일 (backbones, models, epochs 등 정의)")
    p.add_argument("--backbones", nargs="+", default=None, metavar="ARCH",
                   help=f"사용할 backbone 목록. 미지정 시 스윕 설정 또는 기본값(resnet18, resnet50) 사용.")
    p.add_argument("--models", nargs="+", default=None, choices=SUPPORTED_MODELS,
                   metavar="MODEL",
                   help="사용할 모델 목록. 미지정 시 스윕 설정 또는 기본값(baseline, cores) 사용.")
    p.add_argument("--epochs", type=int, default=None,
                   help="에포크 수 오버라이드")
    p.add_argument("--device", default=None,
                   help="디바이스 오버라이드 (cuda / cpu)")
    p.add_argument("--output-dir", default=None,
                   help="결과 저장 루트 폴더 오버라이드")
    p.add_argument("--evaluate", action="store_true",
                   help="학습 후 평가도 실행")
    p.add_argument("--eval-only", action="store_true",
                   help="학습은 건너뛰고 평가만 실행")
    p.add_argument("--evaluators", nargs="+", default=["perturbation"],
                   metavar="EVAL",
                   help="실행할 평가기 목록 (기본값: perturbation). 'all' 지정 시 전체 실행.")
    p.add_argument("--checkpoint", default="best.pt",
                   help="사용할 체크포인트 파일명 (기본값: best.pt)")
    p.add_argument("--skip-existing", action="store_true",
                   help="체크포인트가 이미 존재하면 학습 건너뜀")
    p.add_argument("--dry-run", action="store_true",
                   help="실제 실행 없이 예정된 실험만 출력")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main(argv=None):
    args = _parse_args(argv)

    base_config = load_config(args.config)
    sweep_cfg: dict = yaml.safe_load(open(args.sweep_config)) if args.sweep_config else {}

    # 우선순위: CLI > sweep_config > 기본값
    backbones: list[str] = args.backbones or sweep_cfg.get("backbones") or ["resnet18", "resnet50"]
    models: list[str]    = args.models    or sweep_cfg.get("models")    or ["baseline", "cores"]
    epochs_override      = args.epochs    or sweep_cfg.get("epochs")
    do_evaluate          = args.evaluate  or sweep_cfg.get("evaluate", False)
    eval_only            = args.eval_only or sweep_cfg.get("eval_only", False)
    evaluator_names: list[str] = args.evaluators or sweep_cfg.get("evaluators", ["perturbation"])

    # 유효성 검사
    invalid_bb = [b for b in backbones if b not in SUPPORTED_BACKBONES]
    invalid_m  = [m for m in models    if m not in SUPPORTED_MODELS]
    if invalid_bb:
        sys.exit(f"[ERROR] 지원하지 않는 backbone: {invalid_bb}\n"
                 f"지원 목록: {SUPPORTED_BACKBONES}")
    if invalid_m:
        sys.exit(f"[ERROR] 지원하지 않는 model: {invalid_m}\n"
                 f"지원 목록: {SUPPORTED_MODELS}")

    # 타임스탬프 기반 출력 폴더
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_output_dir = (
        args.output_dir
        or sweep_cfg.get("output_dir")
        or os.path.join(
            base_config["experiment"]["output_dir"],
            f"backbone_sweep_{timestamp}",
        )
    )
    os.makedirs(sweep_output_dir, exist_ok=True)

    # 디바이스 결정
    device = resolve_device(
        args.device or sweep_cfg.get("device") or base_config["experiment"].get("device", "cuda")
    )
    base_config["experiment"]["device"] = device

    total_runs = len(backbones) * len(models)
    print("=" * 65)
    print("  Backbone Sweep 실험")
    print("=" * 65)
    print(f"  Backbone  ({len(backbones):<2}): {', '.join(backbones)}")
    print(f"  Model     ({len(models):<2}): {', '.join(models)}")
    print(f"  총 실행 수: {total_runs}"
          f"  |  에포크: {epochs_override or base_config['training']['epochs']}")
    print(f"  디바이스: {device}  |  출력: {sweep_output_dir}")
    print(f"  평가 실행: {do_evaluate or eval_only}"
          + (f"  ({', '.join(evaluator_names)})" if do_evaluate or eval_only else ""))
    print("=" * 65)

    # 스윕 메타데이터 저장
    sweep_meta = dict(
        timestamp=timestamp, backbones=backbones, models=models,
        epochs=epochs_override or base_config["training"]["epochs"],
        device=device, output_dir=sweep_output_dir,
        base_config=args.config,
        evaluate=do_evaluate or eval_only,
        evaluators=evaluator_names,
    )
    with open(os.path.join(sweep_output_dir, "sweep_config.yaml"), "w") as f:
        yaml.dump(sweep_meta, f, default_flow_style=False, allow_unicode=True)

    # Dry-run
    if args.dry_run:
        print("\n[Dry-run] 다음 실험이 예정되어 있습니다:")
        for i, (bb, mt) in enumerate(
            [(bb, mt) for bb in backbones for mt in models], 1
        ):
            print(f"  {i:>3}. {bb:<30} {mt}")
        return

    # 데이터 로딩 (한 번만)
    print("\n데이터셋 로딩 중...")
    train_loader, test_loader, num_concepts = get_dataloaders(base_config)
    print(f"  num_concepts={num_concepts}  학습={len(train_loader)} 배치  "
          f"검증={len(test_loader)} 배치")

    all_results: dict = {}
    interim_path = os.path.join(sweep_output_dir, "results_interim.json")
    run_pairs = [(bb, mt) for bb in backbones for mt in models]

    for run_idx, (backbone, model_type) in enumerate(run_pairs, 1):
        print(f"\n{'─'*65}")
        print(f"  [{run_idx}/{total_runs}]  backbone={backbone}  model={model_type}")
        print(f"{'─'*65}")

        cfg = _make_run_config(
            base_config, backbone, model_type, sweep_output_dir, epochs_override
        )
        run_result: dict = {}
        ckpt_file = os.path.join(
            sweep_output_dir, backbone, model_type, "checkpoints", args.checkpoint
        )

        # ── 학습 ──────────────────────────────────────────────────────────
        if eval_only:
            print("  [학습 건너뜀] --eval-only 옵션")
        elif args.skip_existing and os.path.exists(ckpt_file):
            print(f"  [학습 건너뜀] 체크포인트 존재: {ckpt_file}")
        else:
            t0 = time.time()
            try:
                history = build_and_train(
                    cfg, model_type, train_loader, test_loader, num_concepts
                )
                val_losses = [
                    ep.get("loss_total", ep.get("loss", float("inf")))
                    for ep in history.get("val", [])
                ]
                if val_losses:
                    run_result["best_val_loss"] = float(min(val_losses))
                run_result["history"] = history
                del history
                elapsed = time.time() - t0
                run_result["train_time_s"] = elapsed
                print(f"  학습 완료: {elapsed:.1f}s  "
                      f"best_val_loss={run_result.get('best_val_loss', 'N/A')}")
            except Exception as exc:
                run_result["error"] = str(exc)
                print(f"  [ERROR] 학습 실패: {exc}")
                traceback.print_exc()

        # ── 평가 ──────────────────────────────────────────────────────────
        if (do_evaluate or eval_only) and "error" not in run_result:
            print("  평가 시작...")
            try:
                eval_model = load_model_for_eval(cfg, model_type, num_concepts, ckpt_file)
                run_result["evaluation"] = run_evaluation(
                    eval_model, cfg, evaluator_names, train_loader, test_loader
                )
                eval_model.cpu()
                del eval_model
                free_memory()
            except Exception as exc:
                run_result["evaluation_error"] = str(exc)
                print(f"  [ERROR] 평가 실패: {exc}")
                traceback.print_exc()

        all_results.setdefault(backbone, {})[model_type] = run_result
        # 중간 결과 저장
        dump_json(interim_path, all_results)

    # ── 최종 요약 ──────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  Sweep 완료! 결과 요약 생성 중...")
    print(f"{'='*65}")

    table_rows = build_summary_table(all_results)
    save_summary(sweep_output_dir, all_results, table_rows)
    plot_comparison(sweep_output_dir, table_rows)
    _print_summary_table(table_rows)
    print(f"\n전체 결과: {sweep_output_dir}")
    return all_results


if __name__ == "__main__":
    main()
