from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    max_seq_length: int = 512
    hidden_size: int = 512
    num_layers: int = 6
    num_heads: int = 8
    intermediate_size: int = 2048
    dropout_rate: float = 0.1
    layer_norm_eps: float = 1e-6

    def validate(self):
        assert self.hidden_size % self.num_heads == 0, \
            "hidden_size must be divisible by num_heads"