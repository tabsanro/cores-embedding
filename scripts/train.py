"""
scripts/train.py — CoRes-Embedding 학습 스크립트

새 모델 타입을 추가하는 방법
─────────────────────────────
1. models/ 에 모델 클래스를 구현합니다.
2. models/__init__.py 의 build_model()에 등록합니다.
3. 전용 트레이너가 필요하면 TRAINER_DISPATCH에 빌더 함수를 등록합니다.
   (표준 Trainer로 충분하다면 별도 등록이 불필요합니다.)
4. SUPPORTED_MODELS 목록에 이름을 추가합니다.

사용 예시
─────────
    python scripts/train.py --model baseline
    python scripts/train.py --model cores --backbone vit_b_16 --epochs 50
    python scripts/train.py --model seqcores --config configs/seqcores.yaml
    python scripts/train.py --model vcores --device cpu --resume
    python scripts/train.py --model cores --set training.learning_rate=1e-3
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

from _utils import (
    _ROOT,
    apply_overrides,
    dump_json,
    free_memory,
    load_checkpoint,
    load_config,
    print_param_summary,
    resolve_device,
    set_seed,
)

sys.path.insert(0, _ROOT)

from data import get_dataloaders
from models import build_model
from training.trainer import Trainer

# ---------------------------------------------------------------------------
# 지원 목록 — 새 모델 추가 시 여기에 이름을 넣으세요
# ---------------------------------------------------------------------------
SUPPORTED_MODELS: list[str] = ["baseline", "cores", "vcores", "seqcores"]


# ---------------------------------------------------------------------------
# 트레이너 디스패치 레지스트리
# ─────────────────────────────────────────────────────────────────────────────
# 각 항목: model_type → (trainer_factory(model, config) → trainer_instance)
#
# 표준 Trainer로 처리되는 모델은 등록하지 않아도 됩니다.
# "seqcores"처럼 전용 트레이너가 필요한 모델만 등록합니다.
# ---------------------------------------------------------------------------

def _build_seqcores_trainer(model, config):
    from models.seqcores import SeqCoResLoss
    from training.seqcores_trainer import SeqCoResTrainer

    sc = config["training"].get("seqcores", {})
    criterion = SeqCoResLoss(
        task_weight=sc.get("task_weight", 1.0),
        vq_weight=sc.get("vq_weight", 1.0),
        recon_weight=sc.get("recon_weight", 1.0),
        residual_penalty_weight=sc.get("residual_penalty_weight", 0.1),
        batch_entropy_weight=sc.get("batch_entropy_weight", 0.1),
        residual_annealing_start=sc.get("residual_annealing_start", 0),
        residual_annealing_end=sc.get("residual_annealing_end", 50),
    )
    return SeqCoResTrainer(model, criterion, config)


def _build_standard_trainer(model, config, model_type):
    return Trainer(model, config, model_type=model_type)


# 키: model_type  값: factory(model, config) → trainer
TRAINER_DISPATCH: dict[str, callable] = {
    "seqcores": _build_seqcores_trainer,
    # 새 전용 트레이너 추가 예시:
    # "my_new_model": lambda model, cfg: MyNewTrainer(model, cfg),
}


# ---------------------------------------------------------------------------
# 핵심 함수 (sweep 스크립트에서 직접 import해서 재사용 가능)
# ---------------------------------------------------------------------------

def build_and_train(
    config: dict,
    model_type: str,
    train_loader,
    test_loader,
    num_concepts: int,
    resume: bool = False,
    checkpoint_name: str = "best.pt",
) -> dict:
    """모델을 빌드하고 학습을 실행합니다.

    Args:
        config:         (수정된) 설정 dict. output_dir과 name이 올바르게 설정된 상태여야 합니다.
        model_type:     SUPPORTED_MODELS 중 하나.
        train_loader:   학습 데이터 로더.
        test_loader:    검증 데이터 로더.
        num_concepts:   데이터셋의 개념(속성) 수.
        resume:         체크포인트에서 이어서 학습할지 여부.
        checkpoint_name: 이어서 학습할 체크포인트 파일명 (기본값 "best.pt").

    Returns:
        학습 히스토리 dict.
    """
    set_seed(config["experiment"]["seed"])

    # num_concepts를 모델 설정에 주입
    config["_model_type"] = model_type
    for key in ("cores", "vcores"):
        if model_type == key:
            config["model"].setdefault(key, {})["num_concepts"] = num_concepts

    model = build_model(config, num_concepts)
    print_param_summary(model)

    # 트레이너 선택
    if model_type in TRAINER_DISPATCH:
        trainer = TRAINER_DISPATCH[model_type](model, config)
    else:
        trainer = _build_standard_trainer(model, config, model_type)

    # Resume
    if resume:
        ckpt_path = os.path.join(trainer.output_dir, "checkpoints", checkpoint_name)
        meta = load_checkpoint(trainer.raw_model
                               if hasattr(trainer, "raw_model") else model,
                               ckpt_path)
        if meta:
            trainer.current_epoch = meta.get("epoch", 0)
            trainer.best_loss = meta.get("best_loss", float("inf"))
            print(f"  epoch={trainer.current_epoch}부터 재개합니다.")

    history = trainer.train(train_loader, test_loader)

    del trainer, model
    free_memory()
    return history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="CoRes-Embedding 학습 스크립트",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--model", required=True, choices=SUPPORTED_MODELS,
                   help="학습할 모델 타입")
    p.add_argument("--config", default="configs/default.yaml",
                   help="기본 설정 파일 경로 (기본값: configs/default.yaml)")
    p.add_argument("--backbone", default=None,
                   help="backbone 아키텍처 오버라이드 (예: vit_b_16, resnet50)")
    p.add_argument("--epochs", type=int, default=None,
                   help="학습 에포크 수 오버라이드")
    p.add_argument("--latent-dim", type=int, default=None,
                   help="잠재 벡터 차원 오버라이드")
    p.add_argument("--device", default=None,
                   help="디바이스 오버라이드 (cuda / cpu)")
    p.add_argument("--name", default=None,
                   help="실험 이름 오버라이드 (출력 폴더명에 반영)")
    p.add_argument("--output-dir", default=None,
                   help="결과 저장 루트 폴더 오버라이드")
    p.add_argument("--resume", action="store_true",
                   help="기존 체크포인트에서 이어서 학습")
    p.add_argument("--checkpoint", default="best.pt",
                   help="Resume 시 사용할 체크포인트 파일명 (기본값: best.pt)")
    p.add_argument("--set", nargs="*", metavar="KEY=VALUE",
                   help="설정 임의 오버라이드. 예: --set training.lr=1e-3 model.latent_dim=128")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    config = load_config(args.config)

    # --- 오버라이드 적용 ---
    if args.backbone:
        config["model"]["backbone"] = args.backbone
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.latent_dim:
        config["model"]["latent_dim"] = args.latent_dim
    if args.device:
        config["experiment"]["device"] = args.device
    if args.name:
        config["experiment"]["name"] = args.name
    if args.output_dir:
        config["experiment"]["output_dir"] = args.output_dir
    if args.set:
        overrides = {}
        for kv in args.set:
            if "=" not in kv:
                print(f"[WARN] --set 인수 형식 오류: '{kv}' (KEY=VALUE 형식 필요)")
                continue
            k, v = kv.split("=", 1)
            # 숫자/불리언 자동 캐스팅
            for cast in (int, float):
                try:
                    v = cast(v); break
                except ValueError:
                    pass
            if v in ("true", "True"):
                v = True
            elif v in ("false", "False"):
                v = False
            overrides[k] = v
        config = apply_overrides(config, overrides)

    # GPU 자동 전환
    config["experiment"]["device"] = resolve_device(config["experiment"]["device"])

    backbone = config["model"].get("backbone", "resnet18")
    num_gpus = torch.cuda.device_count() if config["experiment"]["device"] == "cuda" else 0

    print("=" * 60)
    print("CoRes-Embedding 학습")
    print("=" * 60)
    print(f"  모델:        {args.model}")
    print(f"  Backbone:    {backbone}")
    print(f"  데이터셋:    {config['dataset']['name']}")
    print(f"  잠재 벡터:   {config['model']['latent_dim']}")
    print(f"  디바이스:    {config['experiment']['device']}"
          + (f"  (DataParallel ×{num_gpus})" if num_gpus > 1 else ""))
    print(f"  에포크:      {config['training']['epochs']}")
    print(f"  배치 크기:   {config['training']['batch_size']}")
    print(f"  출력 폴더:   {config['experiment']['output_dir']}/{config['experiment']['name']}")
    print("=" * 60)

    print("\n데이터셋 로딩 중...")
    train_loader, test_loader, num_concepts = get_dataloaders(config)
    print(f"  num_concepts={num_concepts}  학습={len(train_loader)} 배치  "
          f"검증={len(test_loader)} 배치")

    t0 = time.time()
    history = build_and_train(
        config,
        model_type=args.model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_concepts=num_concepts,
        resume=args.resume,
        checkpoint_name=args.checkpoint,
    )
    elapsed = time.time() - t0

    print(f"\n학습 완료! ({elapsed:.1f}s)")
    output_dir = os.path.join(
        config["experiment"]["output_dir"],
        config["experiment"]["name"],
        args.model,
    )
    print(f"결과 저장 위치: {output_dir}")

    # 히스토리 저장
    dump_json(os.path.join(output_dir, "history.json"), history)
    return history


if __name__ == "__main__":
    main()
