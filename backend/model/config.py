import yaml
from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    max_seq_length: int
    ff_dim: int
    dropout: float
    learning_rate: float
    batch_size: int
    num_epochs: int


class Config:
    """
    Loads YAML config into structured dataclass
    """

    def __init__(self, path: str):
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        self.model = ModelConfig(**data)

    def to_dict(self):
        return asdict(self.model)