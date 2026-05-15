# RustTorch Installation Guide

RustTorch is a PyTorch extension that provides high-performance Rust implementations of common operations. This guide shows you how to install and use RustTorch alongside your existing PyTorch installation.

## Quick Start

### Option 1: Install from Source (Current)

PyPI package not yet published. Build from source:

### Option 2: Install from Source

This is the current recommended method:

```bash
# 1. Ensure PyTorch is installed
pip install torch

# 2. Install Rust toolchain (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# 3. Clone the repository
git clone https://github.com/teddytennant/rusttorch.git
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

Use explicitly (no torch monkey-patch):

```python
import rusttorch as rt
from rusttorch import Variable, Linear, Adam, MSELoss
import numpy as np

# Low-level
x = rt.Tensor.zeros([32, 10])
y = rt.add(x, rt.ones([32, 10]))

# Autograd training loop
model = Linear(10, 2)
opt = Adam([p for p in model.parameters()], lr=0.01)
loss_fn = MSELoss()
# v = Variable.from_numpy(...); out = model.forward(v); ...
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

- `cargo test --workspace && cargo clippy --workspace --all-targets -- -D warnings`
- Read [README.md](README.md) and [RUSTTORCH_TODO.md](RUSTTORCH_TODO.md)

## Support

- **Issues**: https://github.com/teddytennant/rusttorch/issues
- **Documentation**: https://github.com/teddytennant/rusttorch#readme
