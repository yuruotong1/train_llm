import os
from pathlib import Path

import torch
from model.model_minimind import MiniMindConfig


ROOT_DIR = Path(__file__).resolve().parent.parent
CORE_DIR = ROOT_DIR / "core"
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUT_DIR = ROOT_DIR / "out"
MODEL_DIR = ROOT_DIR / "model"
DOCS_DIR = CORE_DIR / "docs"


DATASET_FILES = {
    "pretrain": "pretrain_t2t_mini.jsonl",
    "sft": "sft_t2t_mini.jsonl",
    "dpo": "dpo.jsonl",
}


DEFAULTS = {
    "pretrain": {
        "epochs": 2,
        "seq_len": 512,
        "lr": 5e-4,
        "effective_batch_size": 48,
        "micro_batch_size": 4,
        "checkpoint_name": "pretrain",
    },
    "sft": {
        "epochs": 3,
        "seq_len": 512,
        "lr": 2e-5,
        "effective_batch_size": 32,
        "micro_batch_size": 8,
        "checkpoint_name": "sft",
    },
    "dpo": {
        "epochs": 1,
        "seq_len": 512,
        "lr": 5e-7,
        "beta": 0.1,
        "effective_batch_size": 8,
        "micro_batch_size": 2,
        "checkpoint_name": "dpo",
    },
}


def configure_domestic_mirrors() -> None:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("MODELSCOPE_DOMAIN", "www.modelscope.cn")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def ensure_workspace() -> None:
    configure_domestic_mirrors()
    for path in (DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, OUT_DIR, DOCS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def build_config(max_seq_len: int | None = None) -> MiniMindConfig:
    kwargs = {}
    if max_seq_len is not None:
        kwargs["max_position_embeddings"] = max(2048, max_seq_len)
    return MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs)


def checkpoint_path(stage: str) -> Path:
    suffix = build_config().hidden_size
    return OUT_DIR / f"{stage}_{suffix}.pth"


def state_path(stage: str) -> Path:
    return OUT_DIR / f"{stage}_state.pt"


def device_name() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
