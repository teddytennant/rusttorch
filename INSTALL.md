# RustTorch Installation Guide

RustTorch is a PyTorch extension that provides high-performance Rust implementations of common operations. This guide shows you how to install and use RustTorch alongside your existing PyTorch installation.

## Quick Start

### Option 1: Install from PyPI (Recommended - Coming Soon)

```bash
# Install PyTorch first (if not already installed)
pip install torch

# Install RustTorch extension
pip install rusttorch
```

### Option 2: Install from Source

This is the current recommended method:

```bash
# 1. Ensure PyTorch is installed
pip install torch

# 2. Install Rust toolchain (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# 3. Clone the repository
git clone https://github.com/yourusername/rusttorch.git
cd rusttorch

# 4. Install build dependencies
pip install maturin

# 5. Build and install the extension
cd rusttorch-py
maturin develop --release

# 6. Verify installation
python -c "import torch; import rusttorch; print('PyTorch:', torch.__version__); print('RustTorch:', rusttorch.__version__)"
```

## Using RustTorch

### Automatic Integration (Future Feature)

In the future, RustTorch will automatically accelerate PyTorch operations when installed:

```python
import torch

# PyTorch operations may automatically use RustTorch backend
x = torch.randn(1000, 1000)
y = torch.randn(1000, 1000)
result = torch.add(x, y)  # Accelerated by RustTorch if enabled
```

### Explicit Usage (Current Method)

Currently, use RustTorch operations explicitly:

```python
import torch
import rusttorch

# Create PyTorch tensors
x_torch = torch.randn(1000, 1000)
y_torch = torch.randn(1000, 1000)

# Convert to RustTorch for acceleration
x_rust = rusttorch.Tensor.from_numpy(x_torch.numpy())
y_rust = rusttorch.Tensor.from_numpy(y_torch.numpy())

# Use Rust-accelerated operations
result = rusttorch.add(x_rust, y_rust)
activated = rusttorch.relu(result)

# Convert back to PyTorch when needed
result_torch = torch.from_numpy(result.to_numpy())
```

### Direct RustTorch API

You can also use RustTorch's API directly:

```python
import rusttorch

# Create tensors
x = rusttorch.Tensor.zeros([1000, 1000])
y = rusttorch.Tensor.ones([1000, 1000])

# Perform operations
result = rusttorch.add(x, y)
activated = rusttorch.relu(result)

# Available operations:
# - Element-wise: add, mul, sub, div (+ scalar variants)
# - Reductions: sum, mean, max, min
# - Activations: relu, sigmoid, tanh, gelu, softmax, etc.
# - Matrix ops: matmul, transpose, reshape
# - Loss functions: mse_loss, cross_entropy, etc.
# - Optimizer updates: sgd_step, adam_step, etc.
```

## Performance Testing

Compare RustTorch performance with PyTorch:

```bash
cd benchmarks
python compare_pytorch.py
```

This will run benchmarks comparing:
- Element-wise operations
- Activation functions
- Matrix operations
- Reduction operations

## Troubleshooting

### Rust not found

If you get "cargo: command not found":
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

### Maturin build fails

Make sure you have Python development headers:
```bash
# Ubuntu/Debian
sudo apt-get install python3-dev

# macOS
xcode-select --install
```

### Import errors

Verify both packages are installed:
```bash
python -c "import torch; print('PyTorch OK')"
python -c "import rusttorch; print('RustTorch OK')"
```

## What Gets Installed

When you install RustTorch, you get:

1. **RustTorch Python package** (`rusttorch`) - Python bindings to Rust operations
2. **Native Rust library** - Compiled Rust code for high performance
3. **PyTorch dependency** - Ensures PyTorch is available

Your existing PyTorch installation remains unchanged. RustTorch works alongside it.

## Uninstalling

To remove RustTorch:

```bash
pip uninstall rusttorch
```

This removes RustTorch but leaves PyTorch intact.

## Next Steps

- Read [README.md](README.md) for feature overview
- Check [PERFORMANCE.md](PERFORMANCE.md) for performance benchmarks
- See [CONTRIBUTING.md](CONTRIBUTING.md) to contribute
- Review [RUSTTORCH_PLAN.md](RUSTTORCH_PLAN.md) for implementation details

## Support

- **Issues**: https://github.com/yourusername/rusttorch/issues
- **Documentation**: https://github.com/yourusername/rusttorch#readme
