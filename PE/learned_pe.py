import torch
import torch.nn as nn


class LearnedPositionalEncoding(nn.Module):
    """
    Learned Positional Encoding.

    A trainable embedding table: one d_model vector per position.
    Optimized end-to-end with the rest of the model during training.
    Used in: BERT, GPT-2, and many other transformer models.

    Properties:
        - Task-specific: learns optimal positional representations
        - No mathematical constraints on the encoding shape
        - Requires sufficient training data to generalize
        - Fixed maximum sequence length (cannot extrapolate beyond max_seq_len)
        - d_model * max_seq_len learnable parameters
    """

    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Embedding(max_seq_len, d_model)
        nn.init.normal_(self.pe.weight, mean=0.0, std=0.02)  # GPT-style init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            x with learned positional encoding added
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        pe = self.pe(positions).unsqueeze(0)  # (1, seq_len, d_model)
        x = x + pe
        return self.dropout(x)
