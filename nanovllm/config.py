import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    # ---- Qwen3.5 hybrid (GDN + full attention) ----
    is_hybrid: bool = False
    num_gdn_slots: int = -1
    vision_config: AutoConfig | None = None
    image_token_id: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        hf_config = AutoConfig.from_pretrained(self.model)
        if getattr(hf_config, "model_type", None) == "qwen3_5":
            assert self.tensor_parallel_size == 1, "GDN v1 requires tp=1"
            self.is_hybrid = True
            self.vision_config = hf_config.vision_config
            self.image_token_id = getattr(hf_config, "image_token_id", -1)
            hf_config = hf_config.text_config  # model_type == "qwen3_5_text"
            self.enforce_eager = True
            self.max_num_batched_tokens = min(self.max_num_batched_tokens, 8192)
        self.hf_config = hf_config
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        
