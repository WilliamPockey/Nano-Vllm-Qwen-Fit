from .qwen3 import Qwen3ForCausalLM
from .qwen3_moe import Qwen3MoeForCausalLM
from .qwen3_5 import Qwen3_5ForCausalLM

model_dict = {
    "qwen3": Qwen3ForCausalLM,
    "qwen3_moe": Qwen3MoeForCausalLM,
    "qwen3_5": Qwen3_5ForCausalLM,
    "qwen3_5_text": Qwen3_5ForCausalLM,
}