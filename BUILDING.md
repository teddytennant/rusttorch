# Building RustTorch

This guide explains how to build RustTorch from source.

## Prerequisites

### 1. Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

Verify installation:
```bash
rustc --version
cargo --version
```

### 2. Install Python 3.10+

```bash
python3 --version  # Should be 3.10 or higher
```

### 3. Install Maturin (for Python bindings)

```bash
pip install maturin
```

## Building the Rust Core

Build the core Rust library:

```bash
cd rusttorch-core
cargo build --release
```

Run tests:
```bash
cargo test
```

Run benchmarks:
```bash
cargo bench
```

## Building Python Bindings

Build and install the Python package in development mode:

```bash
cd rusttorch-py
maturin develop --release
```

For a production wheel:
```bash
maturin build --release
```

## Running Tests

### Rust Tests
```bash
cd rusttorch-core
cargo test
```

### Python Tests (coming soon)
```bash
cd rusttorch-py
pytest
```

## Running Benchmarks

### Rust Benchmarks
```bash
cd rusttorch-core
cargo bench
```

### Python Performance Comparison
```bash
python benchmarks/compare_pytorch.py
```

## Development Workflow

For active development:

1. Make changes to Rust code
2. Rebuild Python bindings:
   ```bash
   cd rusttorch-py
   maturin develop
   ```
3. Test in Python:
   ```bash
   python -c "import rusttorch; print(rusttorch.Tensor.zeros([2, 3]))"
   ```

## Troubleshooting

### Rust not found
Make sure Rust is installed and in your PATH:
```bash
source $HOME/.cargo/env
```

### Maturin build fails
Ensure you have the latest version:
```bash
pip install --upgrade maturin
```

### Python can't find rusttorch
Make sure you ran `maturin develop` in the rusttorch-py directory.

## Build Configurations

### Debug Build (faster compilation, slower runtime)
```bash
maturin develop
```

### Release Build (slower compilation, faster runtime)
```bash
maturin develop --release
```

### With specific Python version
```bash
maturin develop --release -i python3.11
```

## Platform-Specific Notes

### Linux
No special requirements.

### macOS
May need to install additional build tools:
```bash
xcode-select --install
```

### Windows
Requires Visual Studio Build Tools with C++ support.

## Next Steps

After building, see the README.md for usage examples and the RUSTTORCH_PLAN.md for implementation details.
