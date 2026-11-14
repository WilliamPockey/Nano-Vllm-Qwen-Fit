import torch
from torch import nn
import torch.distributed as dist

from nanovllm.layers.attention import Attention
from nanovllm.layers.gdn_attention import GDNAttention
from nanovllm.layers.layernorm import GemmaRMSNorm
from nanovllm.layers.linear import QKVParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.models.qwen3 import Qwen3MLP   # Qwen3.5 的 MLP 与 Qwen3 完全一致（gate/up/down, silu, 无 bias）

def _get_rope_params(config):
    """Qwen3.5 的 rope 超参藏在 rope_parameters 字典里，兼容两种写法。"""
    rp = getattr(config, "rope_parameters", None) or {}
    #尝试从字典里取"rope_theta"这个key.如果不存在，就用后面的getattr(...)作为默认值。
    rope_theta = rp.get("rope_theta", getattr(config, "rope_theta", 10_000_000))
    partial = rp.get("partial_rotary_factor", 1.0)
    return rope_theta, partial

class Qwen3_5Attention(nn.Module):
    """full_attention 层（24 层里每 4 层一个，共 6 层）。"""

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        assert not getattr(config, "attention_bias", False), "v1: attention_bias must be False"
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = getattr(config, "head_dim", config.hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        rope_theta, partial_rotary_factor = _get_rope_params(config)
        rotary_dim = int(self.head_dim * partial_rotary_factor)      # 256*0.25 = 64

        # q 段因 attn_output_gate 双倍宽（q_gate=True 已支持）：packed 后
        # [q|gate(2*2048) interleaved per-head | k(512) | v(512)]，tp=1 下整拷 q_proj
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size, self.head_dim,
            self.total_num_heads, self.total_num_kv_heads,
            bias=False, q_gate=True,
        )
        self.o_proj = RowParallelLinear(self.total_num_heads * self.head_dim, config.hidden_size, bias=False)
        self.rotary_emb = get_rope(
            self.head_dim, rotary_dim,
            max_position=config.max_position_embeddings, base=rope_theta,
        )
        self.attn = Attention(self.num_heads, self.head_dim, self.scaling, self.num_kv_heads)
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)   # per-head，作用于最后一维
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q_gate, k, v = qkv.split([2 * self.q_size, self.kv_size, self.kv_size], dim=-1)
        # per-head 交错 [q(256)|gate(256)]，不是 q 全段 + gate 全段（HF L743-746）
        q, gate = q_gate.view(-1, self.num_heads, self.head_dim * 2).chunk(2, dim=-1)
        q = self.q_norm(q)
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim))
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)                # partial rope：只旋转前 64 维
        o = self.attn(q, k, v)                                 # [N, num_heads, head_dim]
        o = o * gate.sigmoid()                                           # attention 之后、o_proj 之前过 gate
        return self.o_proj(o.flatten(1, -1))

class Qwen3_5DecoderLayer(nn.Module):

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.block_type = config.layer_types[layer_idx]
        if self.block_type == "linear_attention":
            self.linear_attn = GDNAttention(
                hidden_size=config.hidden_size,
                num_k_heads=config.linear_num_key_heads,
                num_v_heads=config.linear_num_value_heads,
                head_k_dim=config.linear_key_head_dim,
                head_v_dim=config.linear_value_head_dim,
                conv_kernel_size=config.linear_conv_kernel_dim,
                rms_norm_eps=config.rms_norm_eps,
            )
        elif self.block_type == "full_attention":
            self.self_attn = Qwen3_5Attention(config, layer_idx)
        else:
            raise ValueError(f"Invalid layer_type {self.block_type}")
        self.mlp = Qwen3MLP(config.hidden_size, config.intermediate_size, config.hidden_act)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions, hidden_states, residual) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        if self.block_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states)    # GDN 不吃 positions
        else:
            hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

class Qwen3_5LanguageModel(nn.Module):
    """= HF 的 Qwen3_5TextModel；参数前缀 model.language_model.*（Phase D 顶层镜像）。"""

    def __init__(self, config) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3_5DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3_5Model(nn.Module):
    """镜像 HF Qwen3_5Model 的层级（Phase C 在这里加 self.visual = ...）。"""

    def __init__(self, config) -> None:
        super().__init__()
        self.language_model = Qwen3_5LanguageModel(config)


class Qwen3_5ForCausalLM(nn.Module):
    """文本版顶层。类名/层级与 HF 的 Qwen3_5ForConditionalGeneration 文本子树一致，
    权重名全部原生命中：model.language_model.layers.{i}.{linear_attn|self_attn|mlp|input_layernorm|post_attention_layernorm}.*,
    model.language_model.{embed_tokens,norm}.weight；lm_head 因 tie 不存在于 safetensors。
    GDN 的 in_proj_qkv/z/b/a、conv1d.weight、dt_bias、A_log、norm.weight、out_proj.weight
    以及视觉的 model.visual.*、mtp.* 均无需映射（后两类由 loader skip_prefixes 跳过）。
    Phase C 会新增 Qwen3_5ForConditionalGeneration 接管注册键 "qwen3_5"。"""

    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),     # q 段含 gate，双倍宽（QKVParallelLinear.q_gate）
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config) -> None:
        super().__init__()
        self.model = Qwen3_5Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.language_model.embed_tokens.weight.data

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.model.language_model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)