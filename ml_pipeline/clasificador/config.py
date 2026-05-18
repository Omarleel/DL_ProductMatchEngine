from dataclasses import dataclass

@dataclass(frozen=True)
class AttributeModelConfigV2:
    max_tokens: int = 25000
    max_char_tokens: int = 260
    word_seq_len: int = 48
    char_seq_len: int = 140
    text_embedding_dim: int = 128
    char_embedding_dim: int = 32
    unit_embedding_dim: int = 12
    type_embedding_dim: int = 8
    hint_embedding_dim: int = 20
    provider_embedding_dim: int = 20
    trunk_dim: int = 256
    dropout_rate: float = 0.20
    l2_reg: float = 1e-5
    learning_rate: float = 7e-4
