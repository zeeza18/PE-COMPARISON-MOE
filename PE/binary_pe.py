import math
import torch
import torch.nn as nn


class BinaryPositionalEncoding(nn.Module):
    """
    Binary Positional Encoding.

    Encodes position i as its binary bit pattern:
        bit_k(i) = (i >> k) & 1   for k = 0, 1, ..., ceil(log2(max_seq_len))

    Then projects from n_bits -> d_model via a learned linear layer.

    Properties:
        - No trigonometric operations (fast on all hardware)
        - Deterministic bit pattern — positions are uniquely identified
        - One learnable projection layer (n_bits x d_model)
        - Limited expressiveness due to discrete 0/1 encoding
        - Best for: short sequences, edge/resource-constrained devices
    """

    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        n_bits = math.ceil(math.log2(max_seq_len + 1))

        # Build binary encoding table: position i -> n_bits vector
        pe_binary = torch.zeros(max_seq_len, n_bits)
        for pos in range(max_seq_len):
            for bit in range(n_bits):
                pe_binary[pos, bit] = float((pos >> bit) & 1)

        self.register_buffer('pe_binary', pe_binary.unsqueeze(0))  # (1, max_seq_len, n_bits)

        # Learned projection: n_bits -> d_model
        self.projection = nn.Linear(n_bits, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            x with binary positional encoding added
        """
        seq_len = x.size(1)
        pe = self.projection(self.pe_binary[:, :seq_len, :])  # (1, seq_len, d_model)
        x = x + pe
        return self.dropout(x)
