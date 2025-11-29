# Contributing to RustTorch

Thank you for your interest in contributing to RustTorch! This document provides guidelines for contributing.

## Project Goals

RustTorch aims to:
1. Provide high-performance CPU implementations of PyTorch operations
2. Maintain memory safety through Rust's ownership model
3. Offer a PyTorch-compatible API for easy adoption
4. Demonstrate Rust's viability for numerical computing

## Areas for Contribution

### High Priority
- **Core Operations**: Implement tensor operations (element-wise, reductions, linear algebra)
- **Activation Functions**: ReLU, Sigmoid, Tanh, GELU, etc.
- **Performance Optimization**: SIMD, parallelization, memory layout
- **Testing**: Unit tests, integration tests, property-based tests
- **Benchmarking**: Performance comparisons with PyTorch

### Medium Priority
- **Data Loading**: Efficient CSV/data parsers
- **Loss Functions**: MSE, Cross-Entropy, L1/L2
- **Documentation**: Examples, tutorials, API docs
- **Python API**: Enhancing PyO3 bindings

## Development Setup

1. Fork and clone the repository
2. Install Rust: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
3. Install Python 3.10+
4. Install Maturin: `pip install maturin`
5. See [BUILDING.md](BUILDING.md) for detailed build instructions

## License

By contributing, you agree that your contributions will be licensed under the BSD-3-Clause license.
