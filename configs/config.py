from dataclasses import dataclass

@dataclass
class GPTConfig:
    # Model Architecture
    block_size: int = 256      # Max sequence length
    vocab_size: int = 65       # Depends on dataset (e.g., char-level)
    n_layer: int = 6           # Number of transformer blocks
    n_head: int = 6            # Number of attention heads
    n_embd: int = 384          # Embedding dimension
    
    # Training Hyperparameters
    learning_rate: float = 3e-4
    max_iters: int = 5000
    eval_interval: int = 500
    eval_iters: int = 200
    batch_size: int = 64
    dropout: float = 0.2
    
    # Device
    device: str = 'cuda'       # 'cuda' or 'cpu'