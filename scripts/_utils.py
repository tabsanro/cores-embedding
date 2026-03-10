"""
scripts/_utils.py — 3개 스크립트 공통 유틸리티

새 기능을 추가할 때 이 파일에 공유 로직을 넣으세요.
"""

from __future__ import annotations

import gc
import json
import os
import random
import sys

import numpy as np
import torch
import yaml

# 프로젝트 루트를 sys.path에 추가 (scripts/에서 import할 때 필요)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ---------------------------------------------------------------------------
# 재현성
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """모든 난수 생성기에 시드를 설정합니다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# 디바이스 / 메모리
# ---------------------------------------------------------------------------

def resolve_device(device_str: str) -> str:
    """'cuda'를 요청했으나 GPU가 없으면 'cpu'로 자동 전환합니다."""
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA를 사용할 수 없습니다. CPU로 전환합니다.")
        return "cpu"
    return device_str


def free_memory() -> None:
    """GPU 캐시와 Python GC를 해제합니다."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# ---------------------------------------------------------------------------
# 설정 파일
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """YAML 설정 파일을 읽어 dict로 반환합니다."""
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(config: dict, overrides: dict) -> dict:
    """최상위 평탄 key=value 오버라이드를 설정에 적용합니다.

    중첩 키는 점(.) 으로 구분합니다. 예: "training.epochs=50"
    """
    for key, value in overrides.items():
        parts = key.split(".")
        d = config
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value
    return config


# ---------------------------------------------------------------------------
# JSON 직렬화
# ---------------------------------------------------------------------------

def json_safe(obj):
    """numpy / torch Tensor를 JSON 직렬화 가능한 타입으로 변환합니다."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().item() if obj.numel() == 1 else obj.detach().cpu().tolist()
    return obj


def dump_json(path: str, obj) -> None:
    """객체를 JSON으로 직렬화하여 파일에 저장합니다."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(json_safe(obj), f, indent=2)


# ---------------------------------------------------------------------------
# 체크포인트
# ---------------------------------------------------------------------------

def load_checkpoint(model: torch.nn.Module, path: str) -> dict:
    """체크포인트를 불러와 모델에 적용합니다.

    Returns:
        체크포인트에서 읽은 메타데이터 dict (epoch, best_loss 등).
        파일이 없으면 빈 dict를 반환합니다.
    """
    if not os.path.exists(path):
        print(f"[WARN] 체크포인트를 찾을 수 없습니다: {path}")
        return {}
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  체크포인트 로드: {path}")
    return {k: v for k, v in ckpt.items() if k != "model_state_dict"}


# ---------------------------------------------------------------------------
# 모델 파라미터 요약
# ---------------------------------------------------------------------------

def print_param_summary(model: torch.nn.Module) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  파라미터: 전체={total:,}  학습가능={trainable:,}")
