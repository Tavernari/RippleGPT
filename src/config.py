from dataclasses import dataclass

@dataclass
class RippleConfig:
    vocab_size: int = 65
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2
    bias: bool = False
    
    # Magic toggle
    # If True, removes Positional Embeddings entirely (Relying 100% on Ripple Field)
    use_absolute_pos_emb: bool = False 
