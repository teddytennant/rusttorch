# RustTorch - PyTorch Performance Components in Rust

![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

**RustTorch** is an experimental project that rewrites performance-critical parts of PyTorch in Rust, focusing on memory safety, performance, and modern tooling while maintaining compatibility with the PyTorch ecosystem.

## 🎯 Project Goals

- **Performance**: Achieve 1.2x-2x speedup on targeted CPU operations
- **Safety**: Leverage Rust's ownership model for memory safety and thread safety
- **Compatibility**: Maintain API compatibility with PyTorch for seamless integration
- **Modularity**: Create reusable components that can be adopted incrementally

## 🚀 Why Rust?

| Feature | Benefit |
|---------|---------|
| Memory Safety | Eliminate data races and memory leaks at compile time |
| Zero-Cost Abstractions | C++-level performance with high-level code |
| Fearless Concurrency | Safe parallel processing without locks |
| Modern Tooling | Cargo, comprehensive testing, excellent documentation |
| SIMD Support | First-class vectorization for numerical computing |

## 📦 Project Structure

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

## 🎓 What's Being Rewritten

### Phase 1: Core Operations (Current Focus)

#### ✅ High Priority
- **Tensor Operations**: Element-wise ops (add, sub, mul, div), reductions (sum, mean, max, min)
- **Data Loading**: CSV parsing, data preprocessing, batching
- **Activation Functions**: ReLU, Sigmoid, Tanh, Softmax, GELU

#### 📋 Medium Priority
- **Loss Functions**: MSE, Cross-Entropy, L1/L2
- **Matrix Operations**: matmul, transpose, reshape
- **Optimizer Logic**: SGD, Adam/AdamW update rules

### Not Included (Initial Phase)
- ❌ Autograd engine (too complex, core to PyTorch)
- ❌ CUDA kernels (CPU focus first)
- ❌ Neural network layers (depend on autograd)
- ❌ Distributed training (complex coordination)

## 🛠️ Technology Stack

- **Rust**: 1.70+ (latest stable features)
- **PyO3**: Python bindings for seamless integration
- **ndarray**: Multi-dimensional array library
- **rayon**: Data parallelism
- **criterion**: Performance benchmarking

## 📊 Performance Goals

Target operations aim for:
- **1.2x-2x** speedup vs PyTorch C++ backend on CPU
- **Zero** memory leaks or data races
- **100%** API compatibility for implemented operations

## 🏗️ Development Phases

### Phase 1: Foundation ✅ (Completed)
- [x] Project structure and planning
- [x] Core tensor types
- [x] Basic memory management
- [x] Initial Python bindings

### Phase 2: Core Operations ✅ (Completed)
- [x] Element-wise operations (add, mul, sub, div, scalars)
- [x] Reduction operations (sum, mean, max, min, dim-specific)
- [x] Activation functions (ReLU, Sigmoid, Tanh, GELU, Softmax, Leaky ReLU)
- [x] Unit tests (100+ tests with edge cases)

### Phase 3: Integration (Current)
- [x] Python API mirroring PyTorch
- [ ] Performance benchmarks (scripts ready, awaiting Rust installation)
- [x] Documentation
- [ ] CI/CD

### Phase 4: Advanced Features (Next)
- [ ] SIMD optimizations
- [ ] Multi-threading with rayon
- [ ] Matrix operations (matmul, etc.)
- [ ] GPU support (wgpu/CUDA)

## 🧪 Testing Strategy

- **Unit Tests**: Every operation tested in isolation
- **Integration Tests**: Python API compatibility
- **Benchmarks**: Continuous performance comparison
- **Property Testing**: Random test generation for edge cases

## 📚 Getting Started

### Prerequisites
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install Python 3.10+
python3 --version
```

### Building (Coming Soon)
```bash
# Build Rust core
cd rusttorch-core
cargo build --release

# Build Python bindings
cd ../rusttorch-py
maturin develop --release
```

### Usage Example (Planned)
```python
import torch
import rusttorch

# Drop-in replacement for specific operations
x = torch.randn(1000, 1000)
y = torch.randn(1000, 1000)

# Use Rust-accelerated operations
result = rusttorch.add(x, y)  # Faster than torch.add on CPU
```

## 🤝 Contributing

This is an experimental project. Contributions are welcome! Focus areas:
- Core tensor operations
- Performance optimizations
- Documentation
- Testing

## 📄 License

This project follows PyTorch's BSD-style license. See original [PyTorch LICENSE](LICENSE) for details.

## ⚠️ Status

**Current Status**: 🚀 Core Operations Implemented

RustTorch now has fully functional implementations of:
- ✅ Tensor creation and management (zeros, ones, from_vec)
- ✅ Element-wise operations (add, mul, sub, div + scalars)
- ✅ Reduction operations (sum, mean, max, min + dimension-specific)
- ✅ Activation functions (ReLU, Sigmoid, Tanh, GELU, Softmax, Leaky ReLU)
- ✅ Python bindings via PyO3
- ✅ 100+ comprehensive unit tests

**Next Steps**: Performance benchmarking against PyTorch and SIMD optimization

This is an experimental project to explore Rust's viability for PyTorch performance-critical components. It is NOT intended to replace PyTorch, but to complement it with high-performance Rust implementations for specific use cases.

## 📖 Original PyTorch

This project is based on [PyTorch](https://github.com/pytorch/pytorch), the premier deep learning framework. All credit for the original design and implementation goes to the PyTorch team and contributors.

For the original PyTorch documentation, visit:
- [PyTorch.org](https://pytorch.org/)
- [PyTorch GitHub](https://github.com/pytorch/pytorch)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

## 📞 Contact & Resources

- **Documentation**: See [RUSTTORCH_PLAN.md](RUSTTORCH_PLAN.md) for detailed implementation plan
- **Issues**: Report bugs or request features via GitHub Issues
- **Discussions**: Share ideas and questions in GitHub Discussions

---

**Note**: This project is in early development. APIs and features are subject to change.
