<p align="center">
<img width="300" src="assets/logo.png">
</p>

# Nano-Vllm-Qwen-Fit

本项目源自 [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)，是在其基础上修改而来的分支。感谢原作者 [GeeeekExplorer](https://github.com/GeeeekExplorer) 的优秀工作！

本项目保留了原项目的核心能力：轻量、易读的 vLLM 离线推理实现（约 1,200 行 Python 代码），以及 Prefix Caching、Tensor Parallelism、Torch Compile、CUDA Graph 等优化，推理速度与 vLLM 相当。

## 📝 更新日志

### v1.5 (2026-11-11)

- **修改文件**：
  - `engine/model_runner.py` - 实现Qwen3.5的双轨缓存：KV池第一维改为full_attention层数（24层中仅6层），新增GDN状态池（conv窗口+fp32 ssm状态，约19.5MB/序列），按显存预算折算槽位数并回填config供调度器使用；`prepare_prefill`/`prepare_decode`新增gdn_slot、has_initial与CPU侧cu_seqlens的注入；qwen3_5_text强制eager并校验tp=1；权重加载跳过`mtp.*`与`model.visual.*`前缀
  - `layers/gdn_attention.py` - 补充GQA头展开（`repeat_interleave`，适配num_v_heads>num_k_heads的模型）；槽位与分段信息改读context中的CPU list，避免每层device→host同步；明确状态池采用FLA `[H,K,V]` 布局并补全形状约定注释
  - `utils/loader.py` - `load_model` 新增 `skip_prefixes` 参数，支持按前缀跳过权重（默认空，不影响原有模型）
  - `engine/scheduler.py` - 补上 `self.is_hybrid` 初始化
- **问题修复**：
  - 修复KV池块字节数仍按全量层数计算、导致显存预算偏差4倍的问题
  - 修复Scheduler使用未定义属性 `self.is_hybrid` 导致的 `AttributeError`
  - 修复 `load_model` 签名与调用方不一致导致的 `TypeError`

### v1.4 (2026-09-13)

- **新增文件**：
  - `config.py` - 支持Qwen3.5-0.8B的复合Config读取
  - `utils/context.py` - 增加GDN缓存所需字段
  - `engine/sequence.py` - 增加GDN_slot字段
  - `engine/block_manager.py` - 支持Qwen3.5-0.8B的关闭Prefix缓存复用
  - `engine/scheduler.py` - 增加GDN_Slot分配与释放

### v1.3 (2026-09-05)

- **修改文件**：
  - `layers/linear.py` - 适配Qwen3.5-0.8B的q_gate权重加载
- **新增文件**：
  - `layers/gdn_attention.py` - Qwen3.5-0.8B的GDNAttention模块及具体的Gated Delta Rule(!!!模块暂未实现并行，待优化!!!)

### v1.2 (2026-09-01)

- **修改文件**：
  - `layers/layernorm.py` - 适配Qwen3.5-0.8B的GemmaRMSNorm和RMSNormGated
  - `layers/rotary_embedding.py` - 适配Qwen3.5-0.8B的部分位置编码与MRoPE

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
