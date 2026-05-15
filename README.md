# RustTorch - High-Performance PyTorch Extension in Rust

![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

⚠️ **EXPERIMENTAL PROJECT** - This is an experimental fork of PyTorch with major components rewritten in Rust. It is **not intended for real-world use** and should be treated as a research/exploration project only.

**RustTorch** is a PyTorch extension that provides high-performance implementations of common operations in Rust. Install it alongside PyTorch to accelerate CPU-bound operations while maintaining full compatibility with the PyTorch ecosystem.

## Project Goals

- **Drop-in Performance**: Install alongside PyTorch for immediate speedups on CPU operations
- **Safety**: Leverage Rust's ownership model for memory safety and thread safety
- **Full Compatibility**: Works seamlessly with existing PyTorch code - no changes required
- **Selective Acceleration**: Use Rust-optimized ops where beneficial, fall back to PyTorch elsewhere

## Why Rust?

| Feature | Benefit |
|---------|---------|
| Memory Safety | Eliminate data races and memory leaks at compile time |
| Zero-Cost Abstractions | C++-level performance with high-level code |
| Fearless Concurrency | Safe parallel processing without locks |
| Modern Tooling | Cargo, comprehensive testing, excellent documentation |
| SIMD Support | First-class vectorization for numerical computing |

## Project Structure

```
rusttorch/
├── rusttorch-core/          # Core Rust implementation
│   ├── tensor/              # Tensor types and operations
│   ├── ops/                 # Mathematical operations
│   └── memory/              # Memory management
├── rusttorch-py/            # Python bindings (PyO3)
├── benchmarks/              # Performance comparisons
└── RUSTTORCH_PLAN.md        # Detailed implementation plan
```

## What's Being Rewritten

### Completed Features ✅

#### Tensor Operations
- **Element-wise ops**: add, sub, mul, div (with parallel execution for large tensors)
- **Scalar ops**: add_scalar, mul_scalar
- **Reductions**: sum, mean, max, min (global and dimension-specific)
- **Matrix ops**: matmul, transpose, reshape

#### Activation Functions
- **Basic**: ReLU, Leaky ReLU, Sigmoid, Tanh
- **Advanced**: GELU, SELU, ELU, Swish/SiLU, Mish
- **Smooth**: Softmax, Softplus, Softsign

#### Loss Functions
- **Regression**: MSE, L1 (MAE), Smooth L1 (Huber)
- **Classification**: Binary Cross-Entropy, Cross-Entropy

#### Optimizers
- **SGD**: Standard and with Momentum
- **Adam**: Standard Adam and AdamW (with weight decay)

### In Development 🚧
- **GPU Support**: CUDA backend (on `gpu-device-abstraction` branch; main is CPU-only)
- **Polish**: CI, more Python wrappers, property tests, full BumpArena integration, vmap/grad functional API (blocked on slice/concat)

### Current Scope
Full autograd, 20+ nn modules (Conv, norms, attention, Transformer, ResNet, GPT2), optimizers, data loaders, broadcasting (explicit), batched matmul. CPU-only on main; GPU work on side branch. Experimental — not a PyTorch drop-in.

## Technology Stack

- **Rust**: 1.70+ (latest stable features)
- **PyO3**: Python bindings for seamless integration
- **ndarray**: Multi-dimensional array library
- **rayon**: Data parallelism
- **criterion**: Performance benchmarking

## Performance Goals

Target operations aim for:
- **1.2x-2x** speedup vs PyTorch C++ backend on CPU
- **Zero** memory leaks or data races
- **100%** API compatibility for implemented operations

## Status

Core complete: tensors, elementwise+parallel+SIMD-named, reductions, matmul (batched), 13 activations, 5 losses, SGD/Adam, full autograd (Variable), 20+ nn modules (Linear/Conv/Transformer/ResNet/GPT2/etc), data loaders, safetensors, memory arena (Bump), Python bindings for training loop.

**Experimental** — use for research or as reference. See RUSTTORCH_TODO.md for remaining (GPU, more py bindings, CI, property tests, vmap). 668+ tests pass, clippy -D warnings clean.

## Testing Strategy

- **Unit Tests**: Every operation tested in isolation
- **Integration Tests**: Python API compatibility
- **Benchmarks**: Continuous performance comparison
- **Property Testing**: Random test generation for edge cases

## Installation

### Install from Source

See [INSTALL.md](INSTALL.md). PyPI package not yet published.

### Install from Source

**Prerequisites:**
- PyTorch already installed (`pip install torch`)
- Rust toolchain 1.70+ (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)
- Python 3.10+
- Maturin (`pip install maturin`)

**Build Steps:**
```bash
# Clone the repository
git clone https://github.com/yourusername/rusttorch.git
cd rusttorch

# Build and install the extension
cd rusttorch-py
maturin develop --release

# Verify installation (should show PyTorch and RustTorch versions)
python -c "import torch; import rusttorch; print(f'PyTorch: {torch.__version__}'); print(f'RustTorch: {rusttorch.__version__}')"
```

**Run Tests:**
```bash
# Rust unit tests
cd rusttorch-core
cargo test

# Benchmarks
cd ../benchmarks
python compare_pytorch.py
```

## Usage

RustTorch is designed to work alongside PyTorch. You can use PyTorch as normal and selectively use RustTorch for performance-critical operations.

### Method 1: Use PyTorch Normally (with RustTorch in background)

```python
import torch

# Use PyTorch as normal - RustTorch acceleration is automatic for supported ops
x = torch.randn(1000, 1000)
y = torch.randn(1000, 1000)
result = torch.add(x, y)  # May use RustTorch backend if enabled
```

### Method 2: Explicit RustTorch Operations

```python
import torch
import rusttorch

# Convert PyTorch tensors to RustTorch for explicit acceleration
x_torch = torch.randn(1000, 1000)
y_torch = torch.randn(1000, 1000)

# Use RustTorch for CPU-bound operations
x_rust = rusttorch.Tensor.from_numpy(x_torch.numpy())
y_rust = rusttorch.Tensor.from_numpy(y_torch.numpy())
result = rusttorch.add(x_rust, y_rust)  # Rust-accelerated

# Convert back to PyTorch when needed
result_torch = torch.from_numpy(result.to_numpy())
```

### Method 3: Direct RustTorch API

```python
import rusttorch

# Use RustTorch's API directly for new code
x = rusttorch.Tensor.zeros([1000, 1000])
y = rusttorch.Tensor.ones([1000, 1000])
result = rusttorch.add(x, y)
activated = rusttorch.relu(result)
```

## Contributing

This is an experimental project. Contributions are welcome! Focus areas:
- Core tensor operations
- Performance optimizations
- Documentation
- Testing

## License

This project follows PyTorch's BSD-style license. See original [PyTorch LICENSE](LICENSE) for details.

## Status

**Experimental research project**. Full autograd + 20+ nn + training + GPT2 implemented and tested (668+ tests). Not a PyTorch replacement or drop-in. See RUSTTORCH_TODO.md for open items (GPU branch, py wrappers, CI, property tests). Older perf claims aspirational.

## Original PyTorch

This project is based on [PyTorch](https://github.com/pytorch/pytorch), the premier deep learning framework. All credit for the original design and implementation goes to the PyTorch team and contributors.

For the original PyTorch documentation, visit:
- [PyTorch.org](https://pytorch.org/)
- [PyTorch GitHub](https://github.com/pytorch/pytorch)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

## Contact & Resources

- **Documentation**: See [RUSTTORCH_PLAN.md](RUSTTORCH_PLAN.md) for detailed implementation plan
- **Issues**: Report bugs or request features via GitHub Issues
- **Discussions**: Share ideas and questions in GitHub Discussions

---

**Note**: This project is in early development. APIs and features are subject to change.
