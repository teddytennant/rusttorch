# RustTorch Benchmarks

Performance comparison benchmarks for RustTorch vs PyTorch.

## Running Benchmarks

Compare RustTorch performance against PyTorch:

```bash
python compare_pytorch.py
```

This will benchmark:
- Element-wise operations (add, mul, sub, div)
- Activation functions (relu, sigmoid, tanh, gelu)
- Matrix operations (matmul, transpose)
- Reduction operations (sum, mean, max, min)

## Results

The benchmark will show:
- Execution time for each operation
- Speedup factor (RustTorch vs PyTorch)
- Memory usage comparison

## Requirements

- PyTorch installed (`pip install torch`)
- RustTorch installed (`cd rusttorch-py && maturin develop --release`)
