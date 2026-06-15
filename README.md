# train_llm

基于 [MiniMind](https://github.com/jingyaogong/minimind) 的训练框架，`core/` 目录封装了 MiniMind 的模型与训练逻辑。六步走完预训练 → SFT → DPO 全流程。

## 环境安装

需要 Python ≥ 3.10 和 [uv](https://github.com/astral-sh/uv)。

```bash
uv sync
```

torch 需单独安装，版本与本机 CUDA 驱动必须匹配。

**第一步：查本机 CUDA 版本**

```bash
nvcc --version        # 查 CUDA toolkit 版本，认准 "release X.X" 那行
nvidia-smi            # 查驱动支持的最高 CUDA 版本（右上角 CUDA Version）
```

两个命令看的东西不同：
- `nvcc` 是实际安装的 CUDA toolkit，PyTorch 版本应与它对齐
- `nvidia-smi` 显示的是驱动上限，torch 版本不能超过它
- 没装 CUDA toolkit（`nvcc` 不存在）时，以 `nvidia-smi` 的版本为准即可

**第二步：按版本安装**

| CUDA 版本 | 安装命令 |
|---|---|
| 12.8 | `uv pip install torch --index-url https://download.pytorch.org/whl/cu128` |
| 12.6 / 12.7 | `uv pip install torch --index-url https://download.pytorch.org/whl/cu126` |
| 12.4 / 12.5 | `uv pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| 12.1 – 12.3 | `uv pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| 11.8 | `uv pip install torch --index-url https://download.pytorch.org/whl/cu118` |
| 无 GPU / CPU 调试 | `uv pip install torch --index-url https://download.pytorch.org/whl/cpu` |

**验证安装：**

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

输出 `True` 即表示 GPU 可用。

## 六步训练

### 第一步：准备数据

```bash
uv run python trainer/1_prepare_data.py
```

从 ModelScope 下载预训练 / SFT / DPO 数据集到 `data/`，完成 tokenize 并生成 `.bin` 文件。

---

### 第二步：预训练

```bash
uv run python trainer/2_pretrain.py
```

epochs=2，batch_size=48，lr=5e-4，seq_len=512。checkpoint 保存到 `out/`。

---

### 第三步：SFT 指令微调

```bash
uv run python trainer/3_sft.py
```

加载预训练 checkpoint，epochs=3，batch_size=32。训练完自动跑 3 条推理示例。

---

### 第四步：DPO 对齐

```bash
uv run python trainer/4_dpo.py
```

加载 SFT checkpoint，batch_size=8，epochs=1。训练完展示 SFT vs DPO 回答对比。

---

### 第五步：对话

```bash
uv run python trainer/5_chat.py [pretrain|sft|dpo]
```

默认加载 `dpo` checkpoint，输入 `quit` 退出。

---

### 第六步：启动 OpenAI 兼容服务

```bash
uv run python trainer/server.py [--stage dpo] [--host 0.0.0.0] [--port 8998]
```

支持 `/v1/chat/completions` 接口及 tool call，可直接接入任何 OpenAI 兼容客户端。

```bash
# 健康检查
curl http://localhost:8998/health
```

## 项目结构

```
minimind/
├── trainer/
│   ├── 1_prepare_data.py   # 数据下载 + 预处理
│   ├── 2_pretrain.py       # 预训练
│   ├── 3_sft.py            # SFT 微调
│   ├── 4_dpo.py            # DPO 对齐
│   ├── 5_chat.py           # 交互对话
│   └── server.py           # OpenAI 兼容服务
├── core/
│   ├── data_pipeline.py    # 数据下载 + tokenize
│   ├── runtime.py          # 路径 / 超参 / 镜像配置
│   └── train_lib.py        # 训练 / 推理逻辑
├── model/
│   ├── model_minimind.py   # 模型架构
│   ├── model_lora.py       # LoRA 支持
│   └── tokenizer.*         # 分词器
└── pyproject.toml
```

## 说明

- 数据和权重默认走国内镜像（HF Mirror + ModelScope），无需科学上网。
- 所有 checkpoint 保存在 `out/`，不进 git。
- `demo` 可选依赖（streamlit）：`uv sync --extra demo`。
