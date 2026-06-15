import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from core.runtime import ensure_workspace
from core.train_lib import generate_chat_response, load_stage_for_inference


def main():
    ensure_workspace()
    stage = "dpo"
    if len(sys.argv) > 1:
        stage = sys.argv[1].strip().lower()
    if stage not in {"pretrain", "sft", "dpo"}:
        raise ValueError("用法: uv run python trainer/5_chat.py [pretrain|sft|dpo]")

    model, tokenizer = load_stage_for_inference(stage)
    print(f"已加载 {stage} checkpoint，输入 quit 退出。")
    history = []

    while True:
        user_input = input("你: ").strip()
        if user_input.lower() == "quit":
            print("已退出。")
            break
        history.append({"role": "user", "content": user_input})
        answer = generate_chat_response(model, tokenizer, history, max_new_tokens=256)
        print(f"{stage}: {answer}")
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
