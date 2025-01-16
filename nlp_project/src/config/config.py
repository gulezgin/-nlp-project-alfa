from dataclasses import dataclass
from typing import List, Dict

@dataclass
class ModelConfig:
    model_name: str = "bert-base-multilingual-cased"
    max_length: int = 128
    batch_size: int = 16
    learning_rate: float = 2e-5
    epochs: int = 3
    num_labels: int = 3

@dataclass
class DataConfig:
    train_path: str = "data/train.csv"
    test_path: str = "data/test.csv"
    valid_path: str = "data/valid.csv"
    supported_languages: List[str] = ("en", "tr", "es", "fr") 