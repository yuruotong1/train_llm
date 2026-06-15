import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from core.runtime import DEFAULTS, ensure_workspace
from core.train_lib import train_pretrain


def main():
    ensure_workspace()
    cfg = DEFAULTS["pretrain"]
    print(
        "开始预训练: "
        f"epochs={cfg['epochs']}, batch_size={cfg['effective_batch_size']}, "
        f"lr={cfg['lr']}, seq_len={cfg['seq_len']}"
    )
    checkpoint, final_loss = train_pretrain()
    print(f"checkpoint 已保存到: {checkpoint}")
    print(f"最终 loss: {final_loss:.4f}")
    print('✅ "预训练完成"')


if __name__ == "__main__":
    main()
