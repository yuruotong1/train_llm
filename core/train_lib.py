import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from core.runtime import (
    DEFAULTS,
    MODEL_DIR,
    OUT_DIR,
    PROCESSED_DATA_DIR,
    build_config,
    checkpoint_path,
    configure_domestic_mirrors,
    device_name,
    ensure_workspace,
    state_path,
)
from model.model_minimind import MiniMindForCausalLM


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_tokenizer():
    configure_domestic_mirrors()
    return AutoTokenizer.from_pretrained(str(MODEL_DIR))


def load_stage_meta(stage: str) -> dict:
    meta_path = PROCESSED_DATA_DIR / f"{stage}_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"未找到 {meta_path}，请先执行 uv run python trainer/1_prepare_data.py")
    return json.loads(meta_path.read_text(encoding="utf-8"))


class SequenceBinDataset(Dataset):
    def __init__(self, stage: str):
        self.meta = load_stage_meta(stage)
        self.input_ids = np.memmap(
            PROCESSED_DATA_DIR / self.meta["files"]["input_ids"],
            dtype=np.int32,
            mode="r",
            shape=(self.meta["count"], self.meta["seq_len"]),
        )
        self.labels = np.memmap(
            PROCESSED_DATA_DIR / self.meta["files"]["labels"],
            dtype=np.int32,
            mode="r",
            shape=(self.meta["count"], self.meta["seq_len"]),
        )

    def __len__(self) -> int:
        return self.meta["count"]

    def __getitem__(self, index: int):
        return (
            torch.tensor(self.input_ids[index], dtype=torch.long),
            torch.tensor(self.labels[index], dtype=torch.long),
        )


class DPOBinDataset(Dataset):
    def __init__(self):
        self.meta = load_stage_meta("dpo")
        self.tensors = {
            name: np.memmap(
                PROCESSED_DATA_DIR / file_name,
                dtype=np.int32,
                mode="r",
                shape=(self.meta["count"], self.meta["seq_len"]),
            )
            for name, file_name in self.meta["files"].items()
        }

    def __len__(self) -> int:
        return self.meta["count"]

    def __getitem__(self, index: int):
        return {name: torch.tensor(mm[index], dtype=torch.long) for name, mm in self.tensors.items()}


def _autocast(device: str):
    if device.startswith("cuda"):
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def _format_duration(seconds: float) -> str:
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}时{minutes}分{secs}秒"
    if minutes:
        return f"{minutes}分{secs}秒"
    return f"{secs}秒"


def _save_stage(
    stage: str, model, optimizer, current_epoch: int, total_epochs: int, final_loss: float, elapsed_seconds: float
) -> Path:
    ckpt_path = checkpoint_path(stage)
    state_file = state_path(stage)
    raw_state = {k: v.detach().cpu().half() for k, v in model.state_dict().items()}
    torch.save(raw_state, ckpt_path)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "current_epoch": current_epoch,
            "total_epochs": total_epochs,
            "final_loss": final_loss,
            "completed": current_epoch >= total_epochs - 1,
            "elapsed_seconds": elapsed_seconds,
        },
        state_file,
    )
    return ckpt_path


def _check_skip(stage: str) -> tuple[Path, float] | None:
    ckpt = checkpoint_path(stage)
    sf = state_path(stage)
    if ckpt.exists() and sf.exists():
        saved = torch.load(sf, map_location="cpu", weights_only=True)
        if saved.get("completed"):
            elapsed = _format_duration(saved.get("elapsed_seconds", 0.0))
            print(f"[{stage}] checkpoint 已存在（loss={saved['final_loss']:.4f}，用时 {elapsed}），跳过训练")
            return ckpt, saved["final_loss"]
    return None


def _try_resume(stage: str, model, optimizer, device: str) -> tuple[int, float]:
    ckpt = checkpoint_path(stage)
    sf = state_path(stage)
    if not (ckpt.exists() and sf.exists()):
        return 0, 0.0
    saved = torch.load(sf, map_location="cpu", weights_only=True)
    if saved.get("completed"):
        return 0, 0.0
    epoch = saved.get("current_epoch", -1)
    if epoch < 0:
        return 0, 0.0
    state_dict = torch.load(ckpt, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    opt_state = torch.load(sf, map_location=device)
    optimizer.load_state_dict(opt_state["optimizer"])
    start = epoch + 1
    prior_elapsed = saved.get("elapsed_seconds", 0.0)
    print(f"[{stage}] 检测到未完成的训练，从 epoch {start} 恢复（已用时 {_format_duration(prior_elapsed)}）")
    return start, prior_elapsed


def load_model(stage: str | None = None, device: str | None = None, max_seq_len: int = 512):
    ensure_workspace()
    device = device or device_name()
    tokenizer = load_tokenizer()
    model = MiniMindForCausalLM(build_config(max_seq_len=max_seq_len))
    if stage is not None:
        ckpt_path = checkpoint_path(stage)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"未找到 {ckpt_path}，请先完成对应训练阶段")
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
    return model.to(device), tokenizer


def _build_loader(dataset: Dataset, micro_batch_size: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=micro_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def _train_sequence_stage(stage: str, init_from: str | None = None) -> tuple[Path, float]:
    ensure_workspace()
    if result := _check_skip(stage):
        return result

    seed_everything()
    cfg = DEFAULTS[stage]
    device = device_name()
    dataset = SequenceBinDataset(stage)
    model, _ = load_model(init_from, device=device, max_seq_len=cfg["seq_len"])
    model.train()

    micro_batch_size = cfg["micro_batch_size"]
    accumulation_steps = max(1, cfg["effective_batch_size"] // micro_batch_size)
    loader = _build_loader(dataset, micro_batch_size)
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))
    final_loss = 0.0

    start_epoch, prior_elapsed = _try_resume(stage, model, optimizer, device)
    train_start = time.time()

    for epoch in range(start_epoch, cfg["epochs"]):
        epoch_start = time.time()
        epoch_losses = []
        optimizer.zero_grad(set_to_none=True)
        for step, (input_ids, labels) in enumerate(loader, start=1):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            with _autocast(device):
                outputs = model(input_ids, labels=labels)
                aux_loss = outputs.aux_loss if outputs.aux_loss is not None else 0.0
                loss = outputs.loss + aux_loss
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if step % accumulation_steps == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            batch_loss = loss.item() * accumulation_steps
            epoch_losses.append(batch_loss)
            final_loss = batch_loss
            if step % 50 == 0 or step == len(loader):
                print(
                    f"[{stage}] epoch {epoch + 1}/{cfg['epochs']} step {step}/{len(loader)} "
                    f"loss={batch_loss:.4f}"
                )

        final_loss = float(sum(epoch_losses) / max(len(epoch_losses), 1))
        epoch_elapsed = time.time() - epoch_start
        total_elapsed = prior_elapsed + (time.time() - train_start)
        print(
            f"[{stage}] epoch {epoch + 1} average_loss={final_loss:.4f} "
            f"用时 {_format_duration(epoch_elapsed)}（累计 {_format_duration(total_elapsed)}）"
        )
        _save_stage(stage, model, optimizer, epoch, cfg["epochs"], final_loss, total_elapsed)

    print(f"[{stage}] 训练完成，总用时 {_format_duration(prior_elapsed + (time.time() - train_start))}")
    return checkpoint_path(stage), final_loss


def train_pretrain() -> tuple[Path, float]:
    return _train_sequence_stage("pretrain", init_from=None)


def train_sft() -> tuple[Path, float]:
    if not checkpoint_path("pretrain").exists():
        raise FileNotFoundError("未找到预训练 checkpoint，请先运行 uv run python trainer/2_pretrain.py")
    return _train_sequence_stage("sft", init_from="pretrain")


def _logits_to_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=2)
    return torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)


def _dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    ref_log_probs = (ref_log_probs * mask).sum(dim=1)
    policy_log_probs = (policy_log_probs * mask).sum(dim=1)
    batch_size = ref_log_probs.shape[0]
    chosen_ref = ref_log_probs[:batch_size // 2]
    rejected_ref = ref_log_probs[batch_size // 2:]
    chosen_policy = policy_log_probs[:batch_size // 2]
    rejected_policy = policy_log_probs[batch_size // 2:]
    logits = (chosen_policy - rejected_policy) - (chosen_ref - rejected_ref)
    return -F.logsigmoid(beta * logits).mean()


def train_dpo() -> tuple[Path, float]:
    ensure_workspace()
    if result := _check_skip("dpo"):
        return result

    seed_everything()
    cfg = DEFAULTS["dpo"]
    device = device_name()
    if not checkpoint_path("sft").exists():
        raise FileNotFoundError("未找到 SFT checkpoint，请先运行 uv run python trainer/3_sft.py")

    dataset = DPOBinDataset()
    loader = _build_loader(dataset, cfg["micro_batch_size"])
    accumulation_steps = max(1, cfg["effective_batch_size"] // cfg["micro_batch_size"])

    model, tokenizer = load_model("sft", device=device, max_seq_len=cfg["seq_len"])
    ref_model, _ = load_model("sft", device=device, max_seq_len=cfg["seq_len"])
    del tokenizer
    ref_model.eval()
    ref_model.requires_grad_(False)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))
    final_loss = 0.0

    start_epoch, prior_elapsed = _try_resume("dpo", model, optimizer, device)
    train_start = time.time()

    for epoch in range(start_epoch, cfg["epochs"]):
        epoch_start = time.time()
        epoch_losses = []
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader, start=1):
            x_chosen = batch["x_chosen"].to(device)
            y_chosen = batch["y_chosen"].to(device)
            mask_chosen = batch["mask_chosen"].to(device)
            x_rejected = batch["x_rejected"].to(device)
            y_rejected = batch["y_rejected"].to(device)
            mask_rejected = batch["mask_rejected"].to(device)
            x = torch.cat([x_chosen, x_rejected], dim=0)
            y = torch.cat([y_chosen, y_rejected], dim=0)
            mask = torch.cat([mask_chosen, mask_rejected], dim=0)

            with _autocast(device):
                with torch.no_grad():
                    ref_logits = ref_model(x).logits
                    ref_log_probs = _logits_to_log_probs(ref_logits, y)
                outputs = model(x)
                policy_log_probs = _logits_to_log_probs(outputs.logits, y)
                loss = _dpo_loss(ref_log_probs, policy_log_probs, mask, cfg["beta"])
                aux_loss = outputs.aux_loss if outputs.aux_loss is not None else 0.0
                loss = (loss + aux_loss) / accumulation_steps

            scaler.scale(loss).backward()

            if step % accumulation_steps == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            batch_loss = loss.item() * accumulation_steps
            epoch_losses.append(batch_loss)
            final_loss = batch_loss
            if step % 20 == 0 or step == len(loader):
                print(
                    f"[dpo] epoch {epoch + 1}/{cfg['epochs']} step {step}/{len(loader)} "
                    f"loss={batch_loss:.4f}"
                )

        final_loss = float(sum(epoch_losses) / max(len(epoch_losses), 1))
        epoch_elapsed = time.time() - epoch_start
        total_elapsed = prior_elapsed + (time.time() - train_start)
        print(
            f"[dpo] epoch {epoch + 1} average_loss={final_loss:.4f} "
            f"用时 {_format_duration(epoch_elapsed)}（累计 {_format_duration(total_elapsed)}）"
        )
        _save_stage("dpo", model, optimizer, epoch, cfg["epochs"], final_loss, total_elapsed)

    print(f"[dpo] 训练完成，总用时 {_format_duration(prior_elapsed + (time.time() - train_start))}")
    return checkpoint_path("dpo"), final_loss


@torch.inference_mode()
def generate_text(
    stage: str,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    device: str | None = None,
) -> str:
    device = device or device_name()
    model, tokenizer = load_model(stage, device=device, max_seq_len=2048)
    model.eval()
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    generated = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        eos_token_id=tokenizer.eos_token_id,
    )
    answer_ids = generated[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(answer_ids, skip_special_tokens=True).strip()


@torch.inference_mode()
def generate_chat_response(
    model,
    tokenizer,
    messages: list[dict],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    tools: list | None = None,
    open_thinking: bool = False,
):
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=tools or None,
        open_thinking=open_thinking,
    )
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generated = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
    )
    answer_ids = generated[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(answer_ids, skip_special_tokens=True).strip()


def load_stage_for_inference(stage: str, device: str | None = None):
    if stage not in {"pretrain", "sft", "dpo"}:
        raise ValueError("stage 仅支持 pretrain、sft、dpo")
    model, tokenizer = load_model(stage, device=device, max_seq_len=4096)
    model.eval()
    return model, tokenizer
