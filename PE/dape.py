import math
import torch
import torch.nn as nn


class DAPEPositionalEncoding(nn.Module):
    """
    Data-Adaptive Positional Encoding (DAPE) — simplified implementation.

    Inspired by Zheng et al. (NeurIPS 2024):
    "DAPE: Data-Adaptive Positional Encoding for Length Extrapolation"

    Core idea: adapt sinusoidal PE using a lightweight network conditioned
    on input sequence statistics (scale + shift per dimension).

        PE_DAPE(x, i) = (1 + alpha(x)) * PE_sin(i) + beta(x)

    where alpha and beta are learned from [seq_mean, seq_std, norm_length].

    Properties:
        - Dynamic: encoding adapts based on input content
        - Lightweight overhead: small 3->32->2*d_model network
        - Generalizes to variable-length sequences
        - Interpretable: alpha/beta quantify modulation amount

    Note: This is a simplified additive version for unified comparison.
    The original DAPE uses attention-based modulation for length extrapolation.
    """

    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        # Base sinusoidal encoding
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe_base', pe.unsqueeze(0))  # (1, max_seq_len, d_model)

        # Adaptation network: sequence stats -> (alpha, beta) per dimension
        # Input features: [global_mean, global_std, normalized_length]
        self.adapt_net = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 2 * d_model),  # alpha (d_model) + beta (d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            x with data-adaptive positional encoding added
        """
        batch_size, seq_len, _ = x.shape

        # Compute sequence-level statistics (detached — no grad through stats)
        with torch.no_grad():
            seq_stats = torch.stack([
                x.mean(dim=[1, 2]),
                x.std(dim=[1, 2]).clamp(min=1e-6),
                torch.full((batch_size,), seq_len / 512.0, device=x.device),
            ], dim=1)  # (batch, 3)

        # Adaptive scale and shift
        modulation = self.adapt_net(seq_stats)              # (batch, 2*d_model)
        alpha = modulation[:, :self.d_model].unsqueeze(1)  # (batch, 1, d_model)
        beta  = modulation[:, self.d_model:].unsqueeze(1)  # (batch, 1, d_model)

        pe_adaptive = (1 + alpha) * self.pe_base[:, :seq_len, :] + beta
        x = x + pe_adaptive
        return self.dropout(x)
