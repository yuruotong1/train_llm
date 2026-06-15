import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from core.data_pipeline import prepare_all
from core.runtime import DEFAULTS, RAW_DATA_DIR, ensure_workspace


def main():
    ensure_workspace()
    print("开始准备数据...")
    result = prepare_all(
        pretrain_seq_len=DEFAULTS["pretrain"]["seq_len"],
        sft_seq_len=DEFAULTS["sft"]["seq_len"],
        dpo_seq_len=DEFAULTS["dpo"]["seq_len"],
    )

    print(f"数据目录: {RAW_DATA_DIR}")
    for stage, count in result["counts"].items():
        print(f"{stage} 数据条目数: {count}")
    print("预处理完成，已生成 bin 文件到 data/processed/")
    print('✅ "数据就绪"')


if __name__ == "__main__":
    main()
