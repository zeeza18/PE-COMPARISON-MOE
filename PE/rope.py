import torch
import torch.nn as nn


class RoPEPositionalEncoding(nn.Module):
    """
    Rotary Position Embedding (RoPE) — Su et al. (2021).
    "RoFormer: Enhanced Transformer with Rotary Position Embedding"

    Used in: LLaMA, GPT-NeoX, Falcon, Mistral, and most modern LLMs.

    Canonical RoPE rotates query/key vectors inside attention.
    This implementation applies the rotation to the full embedding vector
    so it fits the same additive interface as other PE methods, enabling
    fair comparison in a unified framework.

    Rotation formula:
        x_rotated = x * cos(theta_i) + rotate_half(x) * sin(theta_i)
        where theta_i = pos / 10000^(2i/d_model)

    Properties:
        - Zero learnable parameters
        - Encodes relative positions (not just absolute)
        - Naturally length-generalizable
        - State-of-the-art for language modeling tasks
    """

    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        # Rotation frequencies: theta_i = 1 / 10000^(2i/d_model)
        theta = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        positions = torch.arange(max_seq_len, dtype=torch.float)
        angles = torch.outer(positions, theta)  # (max_seq_len, d_model/2)

        self.register_buffer('cos', torch.cos(angles))  # (max_seq_len, d_model/2)
        self.register_buffer('sin', torch.sin(angles))  # (max_seq_len, d_model/2)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Returns [-x2, x1] where x = [x1, x2] split at halfway."""
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            x with RoPE rotation applied
        """
        seq_len = x.size(1)
        cos = self.cos[:seq_len, :]                        # (seq_len, d_model/2)
        sin = self.sin[:seq_len, :]                        # (seq_len, d_model/2)

        # Broadcast to (1, seq_len, d_model) by repeating for both halves
        cos = torch.cat([cos, cos], dim=-1).unsqueeze(0)  # (1, seq_len, d_model)
        sin = torch.cat([sin, sin], dim=-1).unsqueeze(0)  # (1, seq_len, d_model)

        x_rotated = x * cos + self._rotate_half(x) * sin
        return self.dropout(x_rotated)
