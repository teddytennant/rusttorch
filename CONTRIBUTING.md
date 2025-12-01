# Contributing to RustTorch

Thank you for your interest in contributing to RustTorch! This document provides guidelines for contributing to this PyTorch extension.

## Development Setup

### Prerequisites

1. **PyTorch**: Install PyTorch first
   ```bash
   pip install torch
   ```

2. **Rust Toolchain**: Install Rust 1.70 or later
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

3. **Maturin**: Install the build tool
   ```bash
   pip install maturin
   ```

### Building from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/rusttorch.git
cd rusttorch

# Build Rust core library
cd rusttorch-core
cargo build --release
cargo test

# Build Python extension
cd ../rusttorch-py
maturin develop --release
```

## Testing

### Rust Tests
```bash
cd rusttorch-core
cargo test
cargo test --release  # Release mode tests
```

### Benchmarks
```bash
cd benchmarks
python compare_pytorch.py
```

## Code Style

### Rust Code
- Follow standard Rust conventions (`cargo fmt`)
- Run clippy before submitting: `cargo clippy --all-targets`
- Write tests for new operations
- Document public APIs with doc comments

### Python Code
- Follow PEP 8 style guide
- Add type hints where appropriate
- Write docstrings for public functions

## Submitting Changes

1. **Fork the repository** on GitHub
2. **Create a feature branch** from `main`
   ```bash
   git checkout -b feature/my-new-feature
   ```
3. **Make your changes** with clear, focused commits
4. **Run tests** to ensure nothing breaks
5. **Submit a pull request** with a clear description

## Areas for Contribution

### High Priority
- **SIMD Optimizations**: Explicit vectorization for numerical operations
- **Broadcasting Support**: NumPy-style shape broadcasting
- **Additional Operations**: Implement more PyTorch operations in Rust
- **Performance Testing**: Comprehensive benchmarks

### Medium Priority
- **Documentation**: Improve API docs and examples
- **Error Handling**: Better error messages and validation
- **GPU Support**: CUDA or wgpu backends

### Lower Priority
- **CI/CD**: Automated testing and building
- **Platform Support**: Testing on different OS/architectures

## Questions?

- **Documentation**: See [README.md](README.md) and [RUSTTORCH_PLAN.md](RUSTTORCH_PLAN.md)
- **Issues**: Open an issue on GitHub
- **Discussions**: Start a discussion in GitHub Discussions

Thank you for contributing to RustTorch!
