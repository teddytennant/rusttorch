"""
RustTorch - High-performance PyTorch operations in Rust

This package provides drop-in replacements for performance-critical
PyTorch operations, implemented in Rust for improved safety and speed.
"""

from .rusttorch import (
    Tensor,
    add,
    mul,
    relu,
    __version__,
    __author__,
)

__all__ = [
    "Tensor",
    "add",
    "mul",
    "relu",
    "__version__",
    "__author__",
]
