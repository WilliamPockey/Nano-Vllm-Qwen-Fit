<p align="center">
<img width="300" src="assets/logo.png">
</p>

# Nano-Vllm-Qwen-Fit

本项目源自 [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)，是在其基础上修改而来的分支。感谢原作者 [GeeeekExplorer](https://github.com/GeeeekExplorer) 的优秀工作！

本项目保留了原项目的核心能力：轻量、易读的 vLLM 离线推理实现（约 1,200 行 Python 代码），以及 Prefix Caching、Tensor Parallelism、Torch Compile、CUDA Graph 等优化，推理速度与 vLLM 相当。

## 📝 更新日志

### v1.2 (2026-09-01)

- **修改文件**：
  - `layers/layernorm.py` - 适配Qwen3.5-0.8B的GemmaRMSNorm和RMSNormGated

### v1.1 (2026-08-26)

- **新增模型支持**：添加了 `Qwen3-30B-A3B` 模型的支持
- **新增文件**：
  - `models/models.py` - 模型注册与加载逻辑
  - `models/qwen3_moe.py` - Qwen3 MoE 架构实现
- **修改文件**：
  - `engine/model_runner.py` - 适配新模型的推理引擎

### v1.0

- 初始版本，基于 [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 分支

## 🚧 开发中的新特性

- [ ] **新模型支持**
  - [x] Qwen3-30B-A3B
  - [ ] Qwen3.5-0.8B
- [ ] **分布式支持**
- [ ] **支持 LoRA 加载**

## Installation

直接从本仓库安装：

```bash
pip install git+https://github.com/WilliamPockey/Nano-Vllm-Qwen-Fit.git
```

或克隆后本地安装：

```bash
git clone git@github.com:WilliamPockey/Nano-Vllm-Qwen-Fit.git
cd Nano-Vllm-Qwen-Fit
pip install .
```

## Quick Start

下载模型权重（以 Qwen3-0.6B 为例）：

```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

用法参见 `example.py`。API 与 vLLM 接口保持一致，仅 `LLM.generate` 方法略有差异：

```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```
