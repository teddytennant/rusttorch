"""
Benchmark RustTorch vs PyTorch performance

This script compares the performance of RustTorch operations
against native PyTorch CPU operations.

Author: Theodore Tennant (@teddytennant)
"""

import time
import torch
import numpy as np
import sys
from typing import Callable, Any, Tuple

# Try to import RustTorch if available
RUSTTORCH_AVAILABLE = False
try:
    import rusttorch
    RUSTTORCH_AVAILABLE = True
except ImportError:
    print("Warning: RustTorch not available. Only PyTorch benchmarks will run.")
    print("To build RustTorch: cd rusttorch-py && maturin develop --release\n")


def benchmark_function(func: Callable, *args, iterations: int = 1000, warmup: int = 10) -> float:
    """Benchmark a function with warmup and multiple iterations

    Args:
        func: Function to benchmark
        *args: Arguments to pass to the function
        iterations: Number of iterations to run
        warmup: Number of warmup iterations

    Returns:
        Average time per iteration in seconds
    """
    # Warmup
    for _ in range(warmup):
        func(*args)

    # Actual benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        result = func(*args)
    end = time.perf_counter()

    return (end - start) / iterations


def format_speedup(pytorch_time: float, rusttorch_time: float) -> str:
    """Format speedup comparison"""
    if rusttorch_time == 0:
        return "N/A"
    speedup = pytorch_time / rusttorch_time
    if speedup > 1:
        return f"{speedup:.2f}x faster"
    else:
        return f"{1/speedup:.2f}x slower"


def benchmark_elementwise_ops():
    """Benchmark element-wise operations"""
    print("=" * 80)
    print(" Benchmarking Element-wise Operations")
    print("=" * 80)

    sizes = [100, 500, 1000]
    iterations = 100

    for size in sizes:
        print(f"\n{'Tensor size:':<20} {size}x{size} ({size*size:,} elements)")
        print("-" * 80)

        # PyTorch tensors
        a_torch = torch.randn(size, size)
        b_torch = torch.randn(size, size)

        ops = [
            ("add", torch.add),
            ("mul", torch.mul),
            ("sub", torch.sub),
            ("div", torch.div),
        ]

        for op_name, op_func in ops:
            pytorch_time = benchmark_function(op_func, a_torch, b_torch, iterations=iterations)
            print(f"  {op_name:<15} PyTorch: {pytorch_time*1000:>8.4f} ms", end="")

            if RUSTTORCH_AVAILABLE:
                # Convert to RustTorch tensors (placeholder for when implemented)
                # a_rust = rusttorch.Tensor.from_numpy(a_torch.numpy())
                # b_rust = rusttorch.Tensor.from_numpy(b_torch.numpy())
                # rusttorch_time = benchmark_function(getattr(rusttorch, op_name), a_rust, b_rust, iterations=iterations)
                # print(f"    RustTorch: {rusttorch_time*1000:>8.4f} ms    {format_speedup(pytorch_time, rusttorch_time)}")
                print("    RustTorch: Not yet implemented")
            else:
                print()

        # Scalar operations
        scalar = 2.5
        scalar_ops = [
            ("add_scalar", lambda t: torch.add(t, scalar)),
            ("mul_scalar", lambda t: torch.mul(t, scalar)),
        ]

        for op_name, op_func in scalar_ops:
            pytorch_time = benchmark_function(op_func, a_torch, iterations=iterations)
            print(f"  {op_name:<15} PyTorch: {pytorch_time*1000:>8.4f} ms", end="")

            if RUSTTORCH_AVAILABLE:
                print("    RustTorch: Not yet implemented")
            else:
                print()


def benchmark_activations():
    """Benchmark activation functions"""
    print("\n" + "=" * 80)
    print(" Benchmarking Activation Functions")
    print("=" * 80)

    sizes = [100, 500, 1000]
    iterations = 100

    for size in sizes:
        print(f"\n{'Tensor size:':<20} {size}x{size} ({size*size:,} elements)")
        print("-" * 80)

        x_torch = torch.randn(size, size)

        activations = [
            ("relu", torch.relu),
            ("sigmoid", torch.sigmoid),
            ("tanh", torch.tanh),
            ("gelu", torch.nn.functional.gelu),
        ]

        for act_name, act_func in activations:
            pytorch_time = benchmark_function(act_func, x_torch, iterations=iterations)
            print(f"  {act_name:<15} PyTorch: {pytorch_time*1000:>8.4f} ms", end="")

            if RUSTTORCH_AVAILABLE:
                print("    RustTorch: Not yet implemented")
            else:
                print()

        # Softmax along dimension
        pytorch_time = benchmark_function(torch.nn.functional.softmax, x_torch, 1, iterations=iterations)
        print(f"  {'softmax':<15} PyTorch: {pytorch_time*1000:>8.4f} ms", end="")

        if RUSTTORCH_AVAILABLE:
            print("    RustTorch: Not yet implemented")
        else:
            print()


def benchmark_reductions():
    """Benchmark reduction operations"""
    print("\n" + "=" * 80)
    print(" Benchmarking Reduction Operations")
    print("=" * 80)

    sizes = [100, 500, 1000]
    iterations = 100

    for size in sizes:
        print(f"\n{'Tensor size:':<20} {size}x{size} ({size*size:,} elements)")
        print("-" * 80)

        x_torch = torch.randn(size, size)

        # Global reductions
        reductions = [
            ("sum", torch.sum),
            ("mean", torch.mean),
            ("max", torch.max),
            ("min", torch.min),
        ]

        for red_name, red_func in reductions:
            pytorch_time = benchmark_function(red_func, x_torch, iterations=iterations)
            print(f"  {red_name:<15} PyTorch: {pytorch_time*1000:>8.4f} ms", end="")

            if RUSTTORCH_AVAILABLE:
                print("    RustTorch: Not yet implemented")
            else:
                print()

        # Dimension-specific reductions
        dim_reductions = [
            ("sum_dim", lambda t: torch.sum(t, dim=0)),
            ("mean_dim", lambda t: torch.mean(t, dim=0)),
        ]

        for red_name, red_func in dim_reductions:
            pytorch_time = benchmark_function(red_func, x_torch, iterations=iterations)
            print(f"  {red_name:<15} PyTorch: {pytorch_time*1000:>8.4f} ms", end="")

            if RUSTTORCH_AVAILABLE:
                print("    RustTorch: Not yet implemented")
            else:
                print()


def benchmark_matrix_ops():
    """Benchmark matrix operations"""
    print("\n" + "=" * 80)
    print(" Benchmarking Matrix Operations")
    print("=" * 80)

    # Matrix multiplication with different sizes
    matmul_sizes = [(64, 64, 64), (128, 128, 128), (256, 256, 256)]
    iterations = 50

    print("\nMatrix Multiplication:")
    print("-" * 80)
    for m, k, n in matmul_sizes:
        print(f"\n{'Size:':<20} {m}x{k} @ {k}x{n} = {m}x{n}")

        a_torch = torch.randn(m, k)
        b_torch = torch.randn(k, n)

        pytorch_time = benchmark_function(torch.matmul, a_torch, b_torch, iterations=iterations)
        print(f"  {'matmul':<15} PyTorch: {pytorch_time*1000:>8.4f} ms", end="")

        if RUSTTORCH_AVAILABLE:
            print("    RustTorch: Not yet implemented")
        else:
            print()

    # Transpose
    sizes = [100, 500, 1000]
    print("\n\nTranspose:")
    print("-" * 80)
    for size in sizes:
        print(f"\n{'Tensor size:':<20} {size}x{size}")

        x_torch = torch.randn(size, size)

        pytorch_time = benchmark_function(torch.transpose, x_torch, 0, 1, iterations=100)
        print(f"  {'transpose':<15} PyTorch: {pytorch_time*1000:>8.4f} ms", end="")

        if RUSTTORCH_AVAILABLE:
            print("    RustTorch: Not yet implemented")
        else:
            print()

    # Reshape
    print("\n\nReshape:")
    print("-" * 80)
    for size in sizes:
        print(f"\n{'Tensor size:':<20} {size}x{size} -> {size*size//2}x2")

        x_torch = torch.randn(size, size)
        new_shape = (size * size // 2, 2)

        pytorch_time = benchmark_function(torch.reshape, x_torch, new_shape, iterations=100)
        print(f"  {'reshape':<15} PyTorch: {pytorch_time*1000:>8.4f} ms", end="")

        if RUSTTORCH_AVAILABLE:
            print("    RustTorch: Not yet implemented")
        else:
            print()


def print_summary():
    """Print summary and instructions"""
    print("\n" + "=" * 80)
    print(" Summary")
    print("=" * 80)
    print("\nBenchmark Categories Tested:")
    print("  1. Element-wise operations (add, mul, sub, div, scalar ops)")
    print("  2. Activation functions (relu, sigmoid, tanh, gelu, softmax)")
    print("  3. Reduction operations (sum, mean, max, min, dim-specific)")
    print("  4. Matrix operations (matmul, transpose, reshape)")
    print("\nTensor Sizes: 100x100, 500x500, 1000x1000")
    print("Matrix Multiply: 64x64, 128x128, 256x256")
    print("Iterations: 50-100 per benchmark (with 10 warmup iterations)")

    if not RUSTTORCH_AVAILABLE:
        print("\n" + "!" * 80)
        print(" RustTorch Not Available")
        print("!" * 80)
        print("\nTo enable RustTorch benchmarks:")
        print("  1. cd rusttorch-py")
        print("  2. maturin develop --release")
        print("  3. Re-run this script")
        print("\nNote: Building may take several minutes on first run.")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    print("=" * 80)
    print(" RustTorch vs PyTorch Performance Benchmarks")
    print("=" * 80)
    print(f"\nPyTorch Version: {torch.__version__}")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"RustTorch Available: {RUSTTORCH_AVAILABLE}")
    print()

    try:
        benchmark_elementwise_ops()
        benchmark_activations()
        benchmark_reductions()
        benchmark_matrix_ops()
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\n\nError during benchmarking: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print_summary()
