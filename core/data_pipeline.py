import json
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
from modelscope.hub.file_download import dataset_file_download
from tqdm import tqdm
from transformers import AutoTokenizer

from core.runtime import (
    DATASET_FILES,
    MODEL_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    ROOT_DIR,
    configure_domestic_mirrors,
    ensure_workspace,
)


DATASET_ID = "gongjy/minimind_dataset"


def dataset_raw_path(stage: str) -> Path:
    return RAW_DATA_DIR / DATASET_FILES[stage]


def dataset_meta_path(stage: str) -> Path:
    return PROCESSED_DATA_DIR / f"{stage}_meta.json"


def dataset_tensor_path(stage: str, name: str) -> Path:
    return PROCESSED_DATA_DIR / f"{stage}_{name}.bin"


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def download_or_copy_dataset(stage: str) -> Path:
    ensure_workspace()
    configure_domestic_mirrors()
    file_name = DATASET_FILES[stage]
    target_path = dataset_raw_path(stage)
    if target_path.exists():
        return target_path

    try:
        downloaded = dataset_file_download(
            dataset_id=DATASET_ID,
            file_path=file_name,
            local_dir=str(RAW_DATA_DIR),
        )
        return Path(downloaded)
    except Exception:
        legacy_path = ROOT_DIR / "dataset" / file_name
        if legacy_path.exists():
            shutil.copy2(legacy_path, target_path)
            return target_path
        raise


def load_tokenizer():
    configure_domestic_mirrors()
    return AutoTokenizer.from_pretrained(str(MODEL_DIR))


def _normalize_message(message: dict) -> dict:
    normalized = dict(message)
    for key in ("tool_calls", "tools"):
        if key in normalized and isinstance(normalized[key], str) and normalized[key]:
            try:
                normalized[key] = json.loads(normalized[key])
            except json.JSONDecodeError:
                pass
    if normalized.get("reasoning_content") is None:
        normalized.pop("reasoning_content", None)
    return normalized


def _preprocess_chat(conversations: list[dict]) -> list[dict]:
    if not conversations:
        return conversations
    if any(conv.get("tools") for conv in conversations):
        return [_normalize_message(conv) for conv in conversations]
    return [_normalize_message(conv) for conv in conversations]


def _postprocess_prompt(prompt: str) -> str:
    return prompt.replace("<think>\n\n</think>\n\n", "")


def _render_sft_prompt(tokenizer, conversations: list[dict]) -> str:
    messages = _preprocess_chat(conversations)
    tools = None
    for message in messages:
        if message.get("role") == "system" and message.get("tools"):
            tools = message["tools"]
            break
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=tools,
    )
    return _postprocess_prompt(prompt)


def _assistant_markers(tokenizer):
    bos_ids = tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False).input_ids
    eos_ids = tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False).input_ids
    return bos_ids, eos_ids


def _generate_assistant_labels(input_ids: list[int], bos_ids: list[int], eos_ids: list[int], max_length: int) -> list[int]:
    labels = [-100] * len(input_ids)
    index = 0
    while index < len(input_ids):
        if input_ids[index:index + len(bos_ids)] == bos_ids:
            start = index + len(bos_ids)
            end = start
            while end < len(input_ids):
                if input_ids[end:end + len(eos_ids)] == eos_ids:
                    break
                end += 1
            for pos in range(start, min(end + len(eos_ids), max_length)):
                labels[pos] = input_ids[pos]
            index = end + len(eos_ids) if end < len(input_ids) else len(input_ids)
        else:
            index += 1
    return labels


def _generate_loss_mask(input_ids: list[int], bos_ids: list[int], eos_ids: list[int], max_length: int) -> list[int]:
    mask = [0] * len(input_ids)
    index = 0
    while index < len(input_ids):
        if input_ids[index:index + len(bos_ids)] == bos_ids:
            start = index + len(bos_ids)
            end = start
            while end < len(input_ids):
                if input_ids[end:end + len(eos_ids)] == eos_ids:
                    break
                end += 1
            for pos in range(start, min(end + len(eos_ids), max_length)):
                mask[pos] = 1
            index = end + len(eos_ids) if end < len(input_ids) else len(input_ids)
        else:
            index += 1
    return mask


def preprocess_pretrain(seq_len: int) -> dict:
    source = download_or_copy_dataset("pretrain")
    tokenizer = load_tokenizer()
    count = count_jsonl(source)
    print(f"[pretrain] 共 {count} 条，tokenize 后写入 .bin（此步骤只需做一次，之后训练直接读二进制）")
    input_ids_mm = np.memmap(dataset_tensor_path("pretrain", "input_ids"), dtype=np.int32, mode="w+", shape=(count, seq_len))
    labels_mm = np.memmap(dataset_tensor_path("pretrain", "labels"), dtype=np.int32, mode="w+", shape=(count, seq_len))

    for row_idx, sample in enumerate(tqdm(iter_jsonl(source), total=count, desc="pretrain tokenize")):
        tokens = tokenizer(
            str(sample["text"]),
            add_special_tokens=False,
            truncation=True,
            max_length=seq_len - 2,
        ).input_ids
        tokens = [tokenizer.bos_token_id] + tokens + [tokenizer.eos_token_id]
        padded = tokens + [tokenizer.pad_token_id] * (seq_len - len(tokens))
        labels = list(padded)
        labels = [-100 if token == tokenizer.pad_token_id else token for token in labels]
        input_ids_mm[row_idx] = np.asarray(padded, dtype=np.int32)
        labels_mm[row_idx] = np.asarray(labels, dtype=np.int32)

    input_ids_mm.flush()
    labels_mm.flush()
    meta = {
        "stage": "pretrain",
        "count": count,
        "seq_len": seq_len,
        "dtype": "int32",
        "files": {
            "input_ids": dataset_tensor_path("pretrain", "input_ids").name,
            "labels": dataset_tensor_path("pretrain", "labels").name,
        },
    }
    dataset_meta_path("pretrain").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def preprocess_sft(seq_len: int) -> dict:
    source = download_or_copy_dataset("sft")
    tokenizer = load_tokenizer()
    bos_ids, eos_ids = _assistant_markers(tokenizer)
    count = count_jsonl(source)
    print(f"[sft] 共 {count} 条，tokenize 后写入 .bin")
    input_ids_mm = np.memmap(dataset_tensor_path("sft", "input_ids"), dtype=np.int32, mode="w+", shape=(count, seq_len))
    labels_mm = np.memmap(dataset_tensor_path("sft", "labels"), dtype=np.int32, mode="w+", shape=(count, seq_len))

    for row_idx, sample in enumerate(tqdm(iter_jsonl(source), total=count, desc="sft tokenize")):
        prompt = _render_sft_prompt(tokenizer, sample["conversations"])
        input_ids = tokenizer(prompt, add_special_tokens=False).input_ids[:seq_len]
        input_ids += [tokenizer.pad_token_id] * (seq_len - len(input_ids))
        labels = _generate_assistant_labels(input_ids, bos_ids, eos_ids, seq_len)
        input_ids_mm[row_idx] = np.asarray(input_ids, dtype=np.int32)
        labels_mm[row_idx] = np.asarray(labels, dtype=np.int32)

    input_ids_mm.flush()
    labels_mm.flush()
    meta = {
        "stage": "sft",
        "count": count,
        "seq_len": seq_len,
        "dtype": "int32",
        "files": {
            "input_ids": dataset_tensor_path("sft", "input_ids").name,
            "labels": dataset_tensor_path("sft", "labels").name,
        },
    }
    dataset_meta_path("sft").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def preprocess_dpo(seq_len: int) -> dict:
    source = download_or_copy_dataset("dpo")
    tokenizer = load_tokenizer()
    bos_ids, eos_ids = _assistant_markers(tokenizer)
    count = count_jsonl(source)
    print(f"[dpo] 共 {count} 条，tokenize 后写入 .bin")
    shapes = (count, seq_len - 1)
    tensors = {
        "x_chosen": np.memmap(dataset_tensor_path("dpo", "x_chosen"), dtype=np.int32, mode="w+", shape=shapes),
        "y_chosen": np.memmap(dataset_tensor_path("dpo", "y_chosen"), dtype=np.int32, mode="w+", shape=shapes),
        "mask_chosen": np.memmap(dataset_tensor_path("dpo", "mask_chosen"), dtype=np.int32, mode="w+", shape=shapes),
        "x_rejected": np.memmap(dataset_tensor_path("dpo", "x_rejected"), dtype=np.int32, mode="w+", shape=shapes),
        "y_rejected": np.memmap(dataset_tensor_path("dpo", "y_rejected"), dtype=np.int32, mode="w+", shape=shapes),
        "mask_rejected": np.memmap(dataset_tensor_path("dpo", "mask_rejected"), dtype=np.int32, mode="w+", shape=shapes),
    }

    for row_idx, sample in enumerate(tqdm(iter_jsonl(source), total=count, desc="dpo tokenize")):
        chosen_prompt = _postprocess_prompt(
            tokenizer.apply_chat_template(sample["chosen"], tokenize=False, add_generation_prompt=False)
        )
        rejected_prompt = _postprocess_prompt(
            tokenizer.apply_chat_template(sample["rejected"], tokenize=False, add_generation_prompt=False)
        )
        chosen_ids = tokenizer(
            chosen_prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=seq_len,
            padding="max_length",
        ).input_ids
        rejected_ids = tokenizer(
            rejected_prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=seq_len,
            padding="max_length",
        ).input_ids
        chosen_mask = _generate_loss_mask(chosen_ids, bos_ids, eos_ids, seq_len)
        rejected_mask = _generate_loss_mask(rejected_ids, bos_ids, eos_ids, seq_len)

        tensors["x_chosen"][row_idx] = np.asarray(chosen_ids[:-1], dtype=np.int32)
        tensors["y_chosen"][row_idx] = np.asarray(chosen_ids[1:], dtype=np.int32)
        tensors["mask_chosen"][row_idx] = np.asarray(chosen_mask[1:], dtype=np.int32)
        tensors["x_rejected"][row_idx] = np.asarray(rejected_ids[:-1], dtype=np.int32)
        tensors["y_rejected"][row_idx] = np.asarray(rejected_ids[1:], dtype=np.int32)
        tensors["mask_rejected"][row_idx] = np.asarray(rejected_mask[1:], dtype=np.int32)

    for tensor in tensors.values():
        tensor.flush()

    meta = {
        "stage": "dpo",
        "count": count,
        "seq_len": seq_len - 1,
        "dtype": "int32",
        "files": {name: dataset_tensor_path("dpo", name).name for name in tensors},
    }
    dataset_meta_path("dpo").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def prepare_all(pretrain_seq_len: int, sft_seq_len: int, dpo_seq_len: int) -> dict:
    raw_paths = {stage: download_or_copy_dataset(stage) for stage in DATASET_FILES}
    counts = {stage: count_jsonl(path) for stage, path in raw_paths.items()}
    processed = {
        "pretrain": preprocess_pretrain(pretrain_seq_len),
        "sft": preprocess_sft(sft_seq_len),
        "dpo": preprocess_dpo(dpo_seq_len),
    }
    return {"counts": counts, "processed": processed}
