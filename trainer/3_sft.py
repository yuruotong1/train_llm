import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from core.runtime import DEFAULTS, ensure_workspace
from core.train_lib import generate_chat_response, load_stage_for_inference, train_sft


EXAMPLES = [
    "请用三句话介绍你自己。",
    "推荐两本适合程序员阅读的中文技术书，并说明理由。",
    "帮我写一段鼓励正在找工作的朋友的话。",
]


def main():
    ensure_workspace()
    cfg = DEFAULTS["sft"]
    print(
        "开始 SFT 微调: "
        f"epochs={cfg['epochs']}, batch_size={cfg['effective_batch_size']}"
    )
    checkpoint, final_loss = train_sft()
    print(f"SFT checkpoint 已保存到: {checkpoint}")
    print(f"最终 loss: {final_loss:.4f}")

    model, tokenizer = load_stage_for_inference("sft")
    print("推理示例:")
    for idx, prompt in enumerate(EXAMPLES, start=1):
        answer = generate_chat_response(
            model,
            tokenizer,
            [{"role": "user", "content": prompt}],
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
        )
        print(f"[示例{idx}] 问: {prompt}")
        print(f"[示例{idx}] 答: {answer}")

    print('✅ "SFT完成"')


if __name__ == "__main__":
    main()
