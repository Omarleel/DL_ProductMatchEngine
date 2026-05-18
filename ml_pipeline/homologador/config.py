from dataclasses import dataclass


@dataclass(frozen=True)
class Model2Config:
    max_tokens: int = 18000
    max_char_tokens: int = 160
    word_seq_len: int = 32
    char_seq_len: int = 96
    text_embedding_dim: int = 96
    char_embedding_dim: int = 32
    item_embedding_dim: int = 128
    unit_embedding_dim: int = 12
    type_embedding_dim: int = 8
    dropout_rate: float = 0.20
    l2_reg: float = 1e-5
    learning_rate: float = 8e-4