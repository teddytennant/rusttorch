"""
RustTorch - High-performance tensor/nn/autograd in Rust

PyTorch-like API (own Tensor/Variable types). Experimental standalone lib.
See tests for autograd training examples. Not a torch drop-in.
"""

from .rusttorch import (
    Tensor,
    # Autograd / neural network primitives
    Variable,
    Parameter,
    Linear,
    MSELoss,
    SGD,
    Adam,
    # Element-wise operations
    add,
    mul,
    sub,
    div,
    add_scalar,
    mul_scalar,
    # Matrix operations
    matmul,
    transpose,
    reshape,
    # Reduction operations
    sum,
    mean,
    # Activation functions
    relu,
    sigmoid,
    tanh,
    gelu,
    leaky_relu,
    elu,
    selu,
    swish,
    mish,
    softplus,
    softsign,
    softmax,
    # Loss functions
    mse_loss,
    l1_loss,
    smooth_l1_loss,
    binary_cross_entropy_loss,
    cross_entropy_loss,
    # Optimizers
    sgd_update,
    sgd_momentum_update,
    adam_update,
    adamw_update,
    # Broadcasting operations
    add_broadcast,
    mul_broadcast,
    sub_broadcast,
    div_broadcast,
    # SIMD-optimized operations
    add_simd,
    mul_simd,
    relu_simd,
    mul_scalar_simd,
    fused_multiply_add,
    # Data loading and preprocessing
    load_csv,
    normalize,
    create_batches,
    shuffle_indices,
    train_val_test_split,
    # Metadata
    __version__,
    __author__,
)

__all__ = [
    "Tensor",
    "Variable",
    "Parameter",
    "Linear",
    "MSELoss",
    "SGD",
    "Adam",
    # Element-wise operations
    "add",
    "mul",
    "sub",
    "div",
    "add_scalar",
    "mul_scalar",
    # Matrix operations
    "matmul",
    "transpose",
    "reshape",
    # Reduction operations
    "sum",
    "mean",
    # Activation functions
    "relu",
    "sigmoid",
    "tanh",
    "gelu",
    "leaky_relu",
    "elu",
    "selu",
    "swish",
    "mish",
    "softplus",
    "softsign",
    "softmax",
    # Loss functions
    "mse_loss",
    "l1_loss",
    "smooth_l1_loss",
    "binary_cross_entropy_loss",
    "cross_entropy_loss",
    # Optimizers
    "sgd_update",
    "sgd_momentum_update",
    "adam_update",
    "adamw_update",
    # Broadcasting operations
    "add_broadcast",
    "mul_broadcast",
    "sub_broadcast",
    "div_broadcast",
    # SIMD-optimized operations
    "add_simd",
    "mul_simd",
    "relu_simd",
    "mul_scalar_simd",
    "fused_multiply_add",
    # Data loading and preprocessing
    "load_csv",
    "normalize",
    "create_batches",
    "shuffle_indices",
    "train_val_test_split",
    # Metadata
    "__version__",
    "__author__",
]
