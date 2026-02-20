from .sinusoidal_pe import SinusoidalPositionalEncoding
from .binary_pe import BinaryPositionalEncoding
from .rope import RoPEPositionalEncoding
from .learned_pe import LearnedPositionalEncoding
from .dape import DAPEPositionalEncoding

PE_METHODS = {
    "sinusoidal": SinusoidalPositionalEncoding,
    "binary":     BinaryPositionalEncoding,
    "rope":       RoPEPositionalEncoding,
    "learned":    LearnedPositionalEncoding,
    "dape":       DAPEPositionalEncoding,
}
