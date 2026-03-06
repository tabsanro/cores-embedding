"""
Backbone Sweep Experiment Script
=================================
다양한 backbone에서 baseline과 CoRes 모델을 비교하는 실험 스크립트.

사용법:
    # 기본 실행 (resnet18, resnet50에서 baseline/cores 비교)
    python scripts/run_backbone_sweep.py --backbones resnet18 resnet50

    # 모델 지정
    python scripts/run_backbone_sweep.py \\
        --backbones resnet18 resnet50 densenet121 \\
        --models baseline cores

    # 학습 후 평가까지 실행
    python scripts/run_backbone_sweep.py \\
        --backbones resnet18 resnet50 \\
        --models baseline cores \\
        --evaluate

    # 학습 없이 평가만 실행 (체크포인트가 있는 경우)
    python scripts/run_backbone_sweep.py \\
        --backbones resnet18 resnet50 \\
        --models baseline cores \\
        --eval-only

    # 실험 목록을 YAML 파일로 지정
    python scripts/run_backbone_sweep.py --sweep-config sweep.yaml

sweep.yaml 형식:
    backbones:
      - resnet18
      - resnet50
      - densenet121
    models:
      - baseline
      - cores
    epochs: 50          # 선택 사항 (기본값: config의 epochs 사용)
    evaluate: true      # 선택 사항
"""

import os
import sys
import json
import copy
import time
import argparse
import yaml
import random
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import get_dataloaders
from models import build_model
from training.trainer import Trainer


# ---------------------------------------------------------------------------
# Supported backbones (from backbone.py)
# ---------------------------------------------------------------------------
SUPPORTED_BACKBONES = [
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
    "densenet121", "densenet169", "densenet201", "densenet161",
    "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3",
    "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7",
    "vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32", "vit_h_14",
]

SUPPORTED_MODELS = ["baseline", "cores"]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_run_config(base_config: dict, backbone: str, model_type: str,
                    sweep_output_dir: str, epochs_override: int | None) -> dict:
    """base_config를 복사 후 backbone / 모델 특화 설정을 주입."""
    cfg = copy.deepcopy(base_config)
    cfg["model"]["backbone"] = backbone
    cfg["_model_type"] = model_type

    # 출력 경로를 sweep 내부로 재지정
    cfg["experiment"]["output_dir"] = os.path.join(sweep_output_dir, backbone)
    cfg["experiment"]["name"] = ""   # Trainer가 {output_dir}/{name}/{model_type} 으로 저장

    if epochs_override is not None:
        cfg["training"]["epochs"] = epochs_override

    if model_type == "cores":
        cfg["model"]["cores"]["num_concepts"] = -1  # will be set after dataloading

    return cfg


def checkpoint_path_of(sweep_output_dir: str, backbone: str,
                        model_type: str, filename: str = "best.pt") -> str:
    return os.path.join(sweep_output_dir, backbone, model_type, "checkpoints", filename)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_single(backbone: str, model_type: str, cfg: dict,
                 train_loader, test_loader, num_concepts: int) -> dict:
    """단일 (backbone, model_type) 조합 학습 후 학습 히스토리 반환."""

    set_seed(cfg["experiment"]["seed"])

    if model_type == "cores":
        cfg["model"]["cores"]["num_concepts"] = num_concepts
    elif model_type == "vcores":
        cfg["model"]["vcores"]["num_concepts"] = num_concepts

    model = build_model(cfg, num_concepts)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: total={total_params:,}  trainable={trainable_params:,}")

    trainer = Trainer(model, cfg, model_type=model_type)
    history = trainer.train(train_loader, test_loader)

    # ✅ 명시적 메모리 해제
    del trainer
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()

    return history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_single(backbone: str, model_type: str, cfg: dict,
                    train_loader, test_loader, num_concepts: int,
                    checkpoint: str = "best.pt") -> dict:
    """단일 (backbone, model_type) 조합 평가."""

    from evaluation.perturbation import PerturbationEvaluator
    from evaluation.fewshot import FewShotEvaluator
    from evaluation.manifold import ManifoldEvaluator

    device = cfg["experiment"].get("device", "cpu")

    # 모델 로드
    if model_type == "cores":
        cfg["model"]["cores"]["num_concepts"] = num_concepts
    model = build_model(cfg, num_concepts)

    ckpt_file = checkpoint_path_of(
        cfg["experiment"]["output_dir"],   # already = sweep_output_dir/backbone
        "",  # name is empty, so Trainer path is output_dir/model_type
        model_type, checkpoint,
    )
    # Trainer 저장 경로: {output_dir}/{name}/{model_type}/checkpoints/
    # name="" 이면 그냥 {output_dir}/{model_type}/checkpoints/
    ckpt_file = os.path.join(
        cfg["experiment"]["output_dir"], model_type, "checkpoints", checkpoint
    )

    if os.path.exists(ckpt_file):
        ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded: {ckpt_file}")
    else:
        print(f"  [WARN] No checkpoint at {ckpt_file}, evaluating with random weights")

    results = {}

    print("  -> PerturbationEvaluator")
    try:
        ev = PerturbationEvaluator(model, device=device)
        results["perturbation"] = ev.run_full_evaluation(test_loader, cfg)
    except Exception as e:
        results["perturbation"] = {"error": str(e)}
        print(f"     [ERROR] {e}")

    print("  -> FewShotEvaluator")
    try:
        ev = FewShotEvaluator(model, device=device)
        results["fewshot"] = ev.run_full_evaluation(train_loader, test_loader, cfg)
    except Exception as e:
        results["fewshot"] = {"error": str(e)}
        print(f"     [ERROR] {e}")

    print("  -> ManifoldEvaluator")
    try:
        ev = ManifoldEvaluator(model, device=device)
        results["manifold"] = ev.run_full_evaluation(test_loader, cfg)
    except Exception as e:
        results["manifold"] = {"error": str(e)}
        print(f"     [ERROR] {e}")

    return results


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _safe_get(d, *keys, default=None):
    """중첩 dict에서 안전하게 값 추출."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
        if d is None:
            return default
    return d


def build_summary_table(all_results: dict) -> dict:
    """all_results[backbone][model_type] 구조에서 요약 테이블 생성."""
    rows = []
    for backbone, models in all_results.items():
        for model_type, res in models.items():
            if "error" in res:
                rows.append({"backbone": backbone, "model": model_type,
                             "status": "FAILED", "error": res["error"]})
                continue

            eval_res = res.get("evaluation", {})
            row = {
                "backbone": backbone,
                "model": model_type,
                "status": "OK",
                "train_time_s": res.get("train_time_s"),
                "best_val_loss": res.get("best_val_loss"),
                # few-shot: mean accuracy across shots
                "fewshot_acc_mean": _safe_get(
                    eval_res, "fewshot", "mean_accuracy"
                ),
                # perturbation: robustness score
                "perturbation_robustness": _safe_get(
                    eval_res, "perturbation", "robustness_score"
                ),
            }
            rows.append(row)
    return rows


def save_summary(sweep_output_dir: str, all_results: dict, table_rows: list):
    """결과 JSON 및 Markdown 테이블 저장."""
    summary_dir = os.path.join(sweep_output_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    # JSON
    def _json_safe(obj):
        if isinstance(obj, dict):
            return {str(k): _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_json_safe(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    results_path = os.path.join(summary_dir, "all_results.json")
    with open(results_path, "w") as f:
        json.dump(_json_safe(all_results), f, indent=2)

    # Markdown 테이블
    md_lines = [
        "# Backbone Sweep 실험 결과\n",
        f"생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "| Backbone | Model | Status | Train Time (s) | Best Val Loss"
        " | FewShot Acc | Perturbation Robustness |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in table_rows:
        def _fmt(v, fmt=".4f"):
            return f"{v:{fmt}}" if isinstance(v, float) else (str(v) if v is not None else "-")

        md_lines.append(
            f"| {r['backbone']} | {r['model']} | {r['status']}"
            f" | {_fmt(r.get('train_time_s'), '.1f')}"
            f" | {_fmt(r.get('best_val_loss'))}"
            f" | {_fmt(r.get('fewshot_acc_mean'))}"
            f" | {_fmt(r.get('perturbation_robustness'))} |"
        )

    md_path = os.path.join(summary_dir, "summary.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))

    print(f"\n[Summary] {results_path}")
    print(f"[Summary] {md_path}")
    return summary_dir


def plot_comparison(sweep_output_dir: str, table_rows: list):
    """backbone별 baseline vs cores 비교 바 차트 생성."""
    summary_dir = os.path.join(sweep_output_dir, "summary", "figures")
    os.makedirs(summary_dir, exist_ok=True)

    metrics = [
        ("best_val_loss", "Best Val Loss (↓)", True),
        ("fewshot_acc_mean", "Few-shot Acc Mean (↑)", False),
        ("perturbation_robustness", "Perturbation Robustness (↑)", False),
        ("train_time_s", "Train Time (s)", True),
    ]

    ok_rows = [r for r in table_rows if r["status"] == "OK"]
    if not ok_rows:
        print("[Plot] 성공한 실험 결과가 없어 그래프를 생성하지 않습니다.")
        return

    backbones = sorted({r["backbone"] for r in ok_rows})
    model_types = sorted({r["model"] for r in ok_rows})
    palette = sns.color_palette("Set2", len(model_types))

    for metric_key, metric_label, lower_is_better in metrics:
        vals = {
            (r["backbone"], r["model"]): r.get(metric_key)
            for r in ok_rows
        }
        if all(v is None for v in vals.values()):
            continue

        x = np.arange(len(backbones))
        width = 0.8 / len(model_types)
        offsets = np.linspace(-(len(model_types) - 1) / 2, (len(model_types) - 1) / 2,
                              len(model_types)) * width

        fig, ax = plt.subplots(figsize=(max(6, len(backbones) * 1.5 + 2), 5))
        for i, (mt, color) in enumerate(zip(model_types, palette)):
            heights = [vals.get((bb, mt)) for bb in backbones]
            heights_plot = [h if h is not None else 0 for h in heights]
            bars = ax.bar(x + offsets[i], heights_plot, width, label=mt,
                          color=color, alpha=0.85, edgecolor="white")
            for bar, h in zip(bars, heights):
                if h is None:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.002, "N/A",
                            ha="center", va="bottom", fontsize=7, color="grey")

        ax.set_xlabel("Backbone")
        ax.set_ylabel(metric_label)
        ax.set_title(f"Backbone Sweep: {metric_label}")
        ax.set_xticks(x)
        ax.set_xticklabels(backbones, rotation=30, ha="right")
        ax.legend(title="Model")
        plt.tight_layout()

        fname = metric_key.replace("/", "_")
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(summary_dir, f"{fname}.{ext}"), dpi=150)
        plt.close(fig)
        print(f"[Plot] {metric_key} -> {summary_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-backbone sweep: baseline vs CoRes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", default="configs/default.yaml",
                        help="기본 설정 파일 경로")
    parser.add_argument("--sweep-config", default=None,
                        help="sweep 전용 YAML 파일 (backbones/models 목록 포함)")

    parser.add_argument("--backbones", nargs="+", default=None,
                        metavar="ARCH",
                        help=f"비교할 backbone 목록. 지원: {', '.join(SUPPORTED_BACKBONES)}")
    parser.add_argument("--models", nargs="+", default=None,
                        choices=SUPPORTED_MODELS,
                        help="비교할 모델 종류 (default: baseline cores)")

    parser.add_argument("--epochs", type=int, default=None,
                        help="에폭 수 override (미지정 시 config 값 사용)")
    parser.add_argument("--device", type=str, default=None,
                        help="cuda / cpu override")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="결과 저장 루트 디렉터리 (default: outputs/backbone_sweep_<timestamp>)")

    parser.add_argument("--evaluate", action="store_true",
                        help="학습 후 평가 실행")
    parser.add_argument("--eval-only", action="store_true",
                        help="학습 없이 평가만 실행 (체크포인트 필요)")
    parser.add_argument("--checkpoint", default="best.pt",
                        help="평가에 사용할 체크포인트 파일명 (default: best.pt)")

    parser.add_argument("--skip-existing", action="store_true",
                        help="이미 체크포인트가 존재하면 학습 건너뜀")
    parser.add_argument("--dry-run", action="store_true",
                        help="실험 목록만 출력하고 실제 실행하지 않음")

    return parser.parse_args()


def load_sweep_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()

    # ── 기본 config 로드 ─────────────────────────────────────────────────────
    with open(args.config) as f:
        base_config = yaml.safe_load(f)

    # ── sweep 파라미터 결정 ───────────────────────────────────────────────────
    sweep_cfg: dict = {}
    if args.sweep_config:
        sweep_cfg = load_sweep_config(args.sweep_config)

    backbones: list[str] = (
        args.backbones
        or sweep_cfg.get("backbones")
        or ["resnet18", "resnet50"]
    )
    models: list[str] = (
        args.models
        or sweep_cfg.get("models")
        or ["baseline", "cores"]
    )
    epochs_override: int | None = (
        args.epochs
        or sweep_cfg.get("epochs")
    )
    do_evaluate: bool = args.evaluate or sweep_cfg.get("evaluate", False)
    eval_only: bool = args.eval_only or sweep_cfg.get("eval_only", False)

    # 유효성 검사
    invalid_bb = [b for b in backbones if b not in SUPPORTED_BACKBONES]
    if invalid_bb:
        print(f"[ERROR] 지원하지 않는 backbone: {invalid_bb}")
        print(f"  지원 목록: {', '.join(SUPPORTED_BACKBONES)}")
        sys.exit(1)

    invalid_m = [m for m in models if m not in SUPPORTED_MODELS]
    if invalid_m:
        print(f"[ERROR] 지원하지 않는 model: {invalid_m}")
        sys.exit(1)

    # ── 출력 디렉터리 설정 ────────────────────────────────────────────────────
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

    # ── device 결정 ──────────────────────────────────────────────────────────
    device = (
        args.device
        or sweep_cfg.get("device")
        or base_config["experiment"].get("device", "cuda")
    )
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA 미사용 가능, CPU로 전환")
        device = "cpu"
    base_config["experiment"]["device"] = device

    # ── 실험 계획 출력 ────────────────────────────────────────────────────────
    total_runs = len(backbones) * len(models)
    print("=" * 65)
    print("  Backbone Sweep Experiment")
    print("=" * 65)
    print(f"  Backbones ({len(backbones)}): {', '.join(backbones)}")
    print(f"  Models    ({len(models)}): {', '.join(models)}")
    print(f"  Total runs: {total_runs}")
    print(f"  Epochs: {epochs_override or base_config['training']['epochs']}")
    print(f"  Device: {device}")
    print(f"  Output: {sweep_output_dir}")
    print(f"  Evaluate: {do_evaluate or eval_only}")
    print("=" * 65)

    # sweep 설정 저장
    sweep_meta = {
        "timestamp": timestamp,
        "backbones": backbones,
        "models": models,
        "epochs": epochs_override or base_config["training"]["epochs"],
        "device": device,
        "output_dir": sweep_output_dir,
        "base_config": args.config,
        "evaluate": do_evaluate or eval_only,
    }
    with open(os.path.join(sweep_output_dir, "sweep_config.yaml"), "w") as f:
        yaml.dump(sweep_meta, f, default_flow_style=False, allow_unicode=True)

    if args.dry_run:
        print("\n[Dry-run] 아래 실험이 실행될 예정입니다:")
        for idx, (bb, mt) in enumerate(
            [(bb, mt) for bb in backbones for mt in models], 1
        ):
            print(f"  {idx:>3}. {bb:30s}  {mt}")
        return

    # ── 데이터 로딩 (공용) ────────────────────────────────────────────────────
    print("\n[Data] 데이터셋 로딩 중...")
    train_loader, test_loader, num_concepts = get_dataloaders(base_config)
    print(f"  num_concepts={num_concepts}  "
          f"train={len(train_loader)} batches  "
          f"test={len(test_loader)} batches")

    # ── 실험 루프 ─────────────────────────────────────────────────────────────
    all_results: dict = {}
    run_idx = 0

    for backbone in backbones:
        all_results[backbone] = {}

        for model_type in models:
            run_idx += 1
            tag = f"[{run_idx}/{total_runs}] backbone={backbone}  model={model_type}"
            print(f"\n{'─'*65}")
            print(f"  {tag}")
            print(f"{'─'*65}")

            cfg = make_run_config(
                base_config, backbone, model_type,
                sweep_output_dir, epochs_override
            )
            run_result: dict = {}

            # ── 학습 ──────────────────────────────────────────────────────────
            ckpt_file = os.path.join(
                sweep_output_dir, backbone, model_type, "checkpoints", args.checkpoint
            )

            if eval_only:
                print("  [skip training] --eval-only 지정됨")
            elif args.skip_existing and os.path.exists(ckpt_file):
                print(f"  [skip training] 체크포인트 존재: {ckpt_file}")
            else:
                t0 = time.time()
                try:
                    history = train_single(
                        backbone, model_type, cfg,
                        train_loader, test_loader, num_concepts
                    )
                    elapsed = time.time() - t0
                    run_result["history"] = history
                    run_result["train_time_s"] = elapsed

                    # best val loss 추출
                    val_losses = [
                        ep.get("loss_total", ep.get("loss", float("inf")))
                        for ep in history.get("val", [])
                    ]
                    if val_losses:
                        run_result["best_val_loss"] = float(min(val_losses))

                    print(f"  학습 완료: {elapsed:.1f}s  "
                          f"best_val_loss={run_result.get('best_val_loss', 'N/A')}")
                except Exception as e:
                    run_result["error"] = str(e)
                    print(f"  [ERROR] 학습 실패: {e}")
                    traceback.print_exc()

            # ── 평가 ──────────────────────────────────────────────────────────
            if (do_evaluate or eval_only) and "error" not in run_result:
                print(f"  평가 시작...")
                try:
                    eval_res = evaluate_single(
                        backbone, model_type, cfg,
                        train_loader, test_loader, num_concepts,
                        checkpoint=args.checkpoint,
                    )
                    run_result["evaluation"] = eval_res
                except Exception as e:
                    run_result["evaluation_error"] = str(e)
                    print(f"  [ERROR] 평가 실패: {e}")
                    traceback.print_exc()

            all_results[backbone][model_type] = run_result

            # 중간 결과 즉시 저장 (장기 실험 중 크래시 대비)
            interim_path = os.path.join(sweep_output_dir, "results_interim.json")
            _dump_json(interim_path, all_results)

    # ── 최종 요약 생성 ────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  Sweep 완료! 결과 요약 생성 중...")
    print(f"{'='*65}")

    table_rows = build_summary_table(all_results)
    summary_dir = save_summary(sweep_output_dir, all_results, table_rows)

    # 비교 그래프 (평가 결과가 있는 경우에만 의미 있지만, val loss는 항상 그림)
    plot_comparison(sweep_output_dir, table_rows)

    # 콘솔 요약 테이블
    _print_summary_table(table_rows)

    print(f"\n모든 결과: {sweep_output_dir}")
    return all_results


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _dump_json(path: str, obj):
    def _safe(o):
        if isinstance(o, dict):
            return {str(k): _safe(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_safe(v) for v in o]
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return o

    with open(path, "w") as f:
        json.dump(_safe(obj), f, indent=2)


def _print_summary_table(rows: list):
    header = (
        f"{'Backbone':<22} {'Model':<10} {'Status':<8} "
        f"{'Train(s)':>9} {'ValLoss':>9} {'FewAcc':>8} {'Robust':>8}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for r in rows:
        def _f(v, fmt=".4f"):
            return f"{v:{fmt}}" if isinstance(v, (int, float)) else "-"

        print(
            f"{r['backbone']:<22} {r['model']:<10} {r['status']:<8} "
            f"{_f(r.get('train_time_s'), '9.1f')} "
            f"{_f(r.get('best_val_loss'), '9.4f')} "
            f"{_f(r.get('fewshot_acc_mean'), '8.4f')} "
            f"{_f(r.get('perturbation_robustness'), '8.4f')}"
        )
    print(sep)


if __name__ == "__main__":
    main()
