import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from core.runtime import DEFAULTS, ensure_workspace
from core.train_lib import generate_chat_response, load_stage_for_inference, train_dpo


COMPARE_PROMPTS = [
    "如何高效学习一个新的 Python 项目？",
    "请写一句更有同理心的安慰话语，给刚刚面试失利的人。",
]


def main():
    ensure_workspace()
    cfg = DEFAULTS["dpo"]
    print(
        "开始 DPO 对齐: "
        f"epochs={cfg['epochs']}, batch_size={cfg['effective_batch_size']}"
    )
    checkpoint, final_loss = train_dpo()
    print(f"DPO checkpoint 已保存到: {checkpoint}")
    print(f"最终 loss: {final_loss:.4f}")

    sft_model, sft_tokenizer = load_stage_for_inference("sft")
    dpo_model, dpo_tokenizer = load_stage_for_inference("dpo")
    print("SFT vs DPO 回答对比:")
    for idx, prompt in enumerate(COMPARE_PROMPTS, start=1):
        messages = [{"role": "user", "content": prompt}]
        sft_answer = generate_chat_response(sft_model, sft_tokenizer, messages, max_new_tokens=128)
        dpo_answer = generate_chat_response(dpo_model, dpo_tokenizer, messages, max_new_tokens=128)
        print(f"[对比{idx}] 问题: {prompt}")
        print(f"[对比{idx}] SFT: {sft_answer}")
        print(f"[对比{idx}] DPO: {dpo_answer}")

    print('✅ "DPO完成"')


if __name__ == "__main__":
    main()
