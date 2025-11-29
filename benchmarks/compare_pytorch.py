"""
Benchmark RustTorch vs PyTorch performance

This script compares the performance of RustTorch operations
against native PyTorch CPU operations.
"""

import time
import torch
import numpy as np

# Uncomment when RustTorch is built
# import rusttorch


def benchmark_function(func, *args, iterations=1000, warmup=10):
    """Benchmark a function with warmup and multiple iterations"""
    # Warmup
    for _ in range(warmup):
        func(*args)

    # Actual benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        func(*args)
    end = time.perf_counter()

    return (end - start) / iterations


def benchmark_elementwise_ops():
    """Benchmark element-wise operations"""
    print("=" * 60)
    print("Benchmarking Element-wise Operations")
    print("=" * 60)

    sizes = [100, 1000, 10000]

    for size in sizes:
        print(f"\nTensor size: {size}x{size}")

        # PyTorch tensors
        a = torch.randn(size, size)
        b = torch.randn(size, size)

        # Addition
        pytorch_time = benchmark_function(torch.add, a, b, iterations=100)
        print(f"  PyTorch add: {pytorch_time*1000:.4f} ms")

        # Multiplication
        pytorch_time = benchmark_function(torch.mul, a, b, iterations=100)
        print(f"  PyTorch mul: {pytorch_time*1000:.4f} ms")

        # TODO: Add RustTorch benchmarks when library is built
        # rusttorch_time = benchmark_function(rusttorch.add, a, b, iterations=100)
        # speedup = pytorch_time / rusttorch_time
        # print(f"  RustTorch add: {rusttorch_time*1000:.4f} ms (speedup: {speedup:.2f}x)")


def benchmark_activations():
    """Benchmark activation functions"""
    print("\n" + "=" * 60)
    print("Benchmarking Activation Functions")
    print("=" * 60)

    sizes = [1000, 5000, 10000]

    for size in sizes:
        print(f"\nTensor size: {size}x{size}")

        x = torch.randn(size, size)

        # ReLU
        pytorch_time = benchmark_function(torch.relu, x, iterations=100)
        print(f"  PyTorch ReLU: {pytorch_time*1000:.4f} ms")

        # Sigmoid
        pytorch_time = benchmark_function(torch.sigmoid, x, iterations=100)
        print(f"  PyTorch Sigmoid: {pytorch_time*1000:.4f} ms")

        # Tanh
        pytorch_time = benchmark_function(torch.tanh, x, iterations=100)
        print(f"  PyTorch Tanh: {pytorch_time*1000:.4f} ms")


def benchmark_reductions():
    """Benchmark reduction operations"""
    print("\n" + "=" * 60)
    print("Benchmarking Reduction Operations")
    print("=" * 60)

    sizes = [1000, 5000, 10000]

    for size in sizes:
        print(f"\nTensor size: {size}x{size}")

        x = torch.randn(size, size)

        # Sum
        pytorch_time = benchmark_function(torch.sum, x, iterations=100)
        print(f"  PyTorch sum: {pytorch_time*1000:.4f} ms")

        # Mean
        pytorch_time = benchmark_function(torch.mean, x, iterations=100)
        print(f"  PyTorch mean: {pytorch_time*1000:.4f} ms")

        # Max
        pytorch_time = benchmark_function(torch.max, x, iterations=100)
        print(f"  PyTorch max: {pytorch_time*1000:.4f} ms")


if __name__ == "__main__":
    print("RustTorch Performance Benchmarks")
    print("=" * 60)
    print("\nNOTE: RustTorch comparisons will be available after building the library")
    print("To build: cd rusttorch-py && maturin develop --release\n")

    benchmark_elementwise_ops()
    benchmark_activations()
    benchmark_reductions()

    print("\n" + "=" * 60)
    print("Benchmarks complete!")
    print("=" * 60)
