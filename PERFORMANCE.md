# RustTorch Performance Guide

**Author**: Theodore Tennant (@teddytennant)
**Last Updated**: November 29, 2025

## Overview

This document provides comprehensive performance information for RustTorch, including benchmarking methodology, optimization techniques, profiling guidance, and expected performance characteristics.

## Table of Contents

1. [Benchmarking](#benchmarking)
2. [Performance Targets](#performance-targets)
3. [Profiling Guide](#profiling-guide)
4. [Optimization Strategies](#optimization-strategies)
5. [Memory Performance](#memory-performance)
6. [Known Bottlenecks](#known-bottlenecks)
7. [Future Optimizations](#future-optimizations)

---

## Benchmarking

### Running Benchmarks

#### Rust Benchmarks (Criterion)

```bash
# Run all benchmarks
cd rusttorch-core
cargo bench

# Run specific benchmark group
cargo bench --bench tensor_ops -- elementwise_ops

# Run with baseline comparison
cargo bench --bench tensor_ops -- --save-baseline main
# ... make changes ...
cargo bench --bench tensor_ops -- --baseline main
```

**Benchmark Coverage:**
- Tensor creation (zeros, ones, from_vec)
- Element-wise operations (add, mul, sub, div, scalar ops)
- Reduction operations (sum, mean, max, min, dimensional)
- Activation functions (relu, sigmoid, tanh, gelu, softmax)
- Tensor sizes: 100x100, 500x500, 1000x1000

#### Python Comparison Benchmarks

```bash
# Compare RustTorch vs PyTorch
cd benchmarks
python compare_pytorch.py
```

This script benchmarks:
- Element-wise operations
- Activation functions
- Reduction operations
- Side-by-side comparison when RustTorch Python bindings are built

### Benchmark Methodology

All benchmarks follow these principles:

1. **Warmup Phase**: 10 iterations to warm up CPU caches
2. **Measurement Phase**: 100 iterations for statistical significance
3. **Black Box**: Results are passed through `black_box()` to prevent optimization
4. **Consistent Sizes**: Tests use 100², 500², and 1000² element tensors
5. **Random Data**: Input data is randomized to prevent cache effects

---

## Performance Targets

### Primary Goals

| Metric | Target | Status |
|--------|--------|--------|
| Element-wise ops vs PyTorch CPU | 1.2x - 2.0x faster | To be measured |
| Memory safety overhead | 0% (compile-time) | ✅ Achieved |
| Allocation efficiency | Minimize copies | 🔄 In progress |
| Cache efficiency | Maximize locality | ✅ Contiguous layout |
| Parallel scalability | Linear to 8 cores | 🔜 Planned |

### Operation-Specific Targets

#### Element-wise Operations
- **Target**: 1.5x faster than PyTorch CPU
- **Rationale**: Simple operations benefit from Rust's zero-cost abstractions
- **Key Optimizations**: SIMD vectorization, loop unrolling

#### Activation Functions
- **Target**: 1.2x - 1.8x faster than PyTorch CPU
- **Rationale**: Transcendental functions (sigmoid, tanh) have overhead
- **Key Optimizations**: Fast approximations, vectorization

#### Reductions
- **Target**: 1.3x - 2.0x faster than PyTorch CPU
- **Rationale**: Cache-friendly sequential access patterns
- **Key Optimizations**: Parallel reduction, tree-based algorithms

---

## Profiling Guide

### Rust Profiling

#### CPU Profiling with perf

```bash
# Install perf (Linux)
sudo apt-get install linux-tools-generic

# Build with debug symbols
cd rusttorch-core
cargo build --release

# Profile a benchmark
perf record --call-graph=dwarf cargo bench --bench tensor_ops -- add/1000
perf report

# Generate flamegraph
git clone https://github.com/flamegraph-rs/flamegraph
cargo install flamegraph
cargo flamegraph --bench tensor_ops -- add/1000
```

#### Memory Profiling with Valgrind

```bash
# Install valgrind
sudo apt-get install valgrind

# Profile memory usage
valgrind --tool=massif cargo bench --bench tensor_ops -- add/100
ms_print massif.out.<PID>

# Check for memory leaks
valgrind --leak-check=full cargo test
```

#### Criterion Reports

Criterion generates HTML reports with:
- Time vs iteration count
- Statistical analysis (mean, median, std dev)
- Outlier detection
- Throughput estimates

**Location**: `rusttorch-core/target/criterion/`

### Identifying Hotspots

Common hotspots to investigate:

1. **Memory Allocation**
   - Look for: `alloc()`, `Vec::with_capacity()`
   - Solution: Pre-allocate, use object pools

2. **Data Copying**
   - Look for: `.clone()`, `.to_vec()`, `.to_owned()`
   - Solution: Use references, implement in-place operations

3. **Scalar Operations in Loops**
   - Look for: Element-by-element access in large loops
   - Solution: SIMD vectorization, batch processing

4. **Cache Misses**
   - Look for: Random access patterns, large stride jumps
   - Solution: Reorganize data layout, use cache blocking

---

## Optimization Strategies

### 1. SIMD Vectorization

**Current Status**: Partial (ndarray provides some)
**Target**: Manual SIMD for hot paths

```rust
// Example: Manual SIMD with std::simd (when stable)
use std::simd::*;

pub fn add_simd(a: &[f32], b: &[f32], result: &mut [f32]) {
    let chunks = a.len() / 4;

    for i in 0..chunks {
        let idx = i * 4;
        let va = f32x4::from_slice(&a[idx..idx+4]);
        let vb = f32x4::from_slice(&b[idx..idx+4]);
        let vr = va + vb;
        vr.copy_to_slice(&mut result[idx..idx+4]);
    }

    // Handle remainder
    for i in (chunks * 4)..a.len() {
        result[i] = a[i] + b[i];
    }
}
```

**Expected Speedup**: 2-4x for element-wise operations

### 2. Parallel Processing with Rayon

**Current Status**: Available but not utilized
**Target**: Parallelize operations on large tensors

```rust
use rayon::prelude::*;

pub fn add_parallel(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // Use parallel iterators for large tensors
    const PARALLEL_THRESHOLD: usize = 10_000;

    if a.numel() > PARALLEL_THRESHOLD {
        // Parallel implementation
        let data: Vec<f32> = a.data()
            .par_iter()
            .zip(b.data().par_iter())
            .map(|(x, y)| x + y)
            .collect();
        Ok(Tensor::from_vec(data, a.shape()))
    } else {
        // Sequential for small tensors (avoid overhead)
        add_sequential(a, b)
    }
}
```

**Expected Speedup**: Near-linear with core count for large tensors

### 3. Memory Pool Allocation

**Current Status**: Not implemented
**Target**: Reduce allocation overhead

```rust
// Concept: Reusable tensor memory pool
pub struct TensorPool {
    small: Vec<Vec<f32>>,   // < 1KB
    medium: Vec<Vec<f32>>,  // 1KB - 1MB
    large: Vec<Vec<f32>>,   // > 1MB
}

impl TensorPool {
    pub fn get(&mut self, size: usize) -> Vec<f32> {
        // Return pre-allocated buffer or allocate new
    }

    pub fn return_buffer(&mut self, buffer: Vec<f32>) {
        // Add buffer back to pool
    }
}
```

**Expected Speedup**: 5-10x for repeated small allocations

### 4. In-Place Operations

**Current Status**: Not implemented
**Target**: Avoid unnecessary copies

```rust
// Current: Creates new tensor
let result = add(&a, &b);

// Future: Modify in-place
let mut result = a.clone();
result.add_(&b);  // In-place addition (save one allocation)
```

**Expected Speedup**: 1.5-2x for operation chains

### 5. Cache-Friendly Layouts

**Current Status**: Contiguous row-major (good)
**Future**: Tiled/blocked layouts for matrix ops

```rust
// Blocked matrix multiply for better cache utilization
const BLOCK_SIZE: usize = 64;

for i in (0..n).step_by(BLOCK_SIZE) {
    for j in (0..n).step_by(BLOCK_SIZE) {
        for k in (0..n).step_by(BLOCK_SIZE) {
            // Multiply BLOCK_SIZE x BLOCK_SIZE submatrices
            multiply_block(a, b, c, i, j, k, BLOCK_SIZE);
        }
    }
}
```

**Expected Speedup**: 2-3x for large matrix operations

---

## Memory Performance

### Current Memory Characteristics

| Operation | Memory Allocation | Notes |
|-----------|------------------|-------|
| Tensor creation | 1x allocation | Contiguous Vec<T> |
| Element-wise op | 1x allocation | Output tensor |
| Reduction | Minimal | Returns scalar |
| Clone | 1x allocation | Full copy |
| Slice view | 0 allocations | ❌ Not implemented |

### Memory Optimization Priorities

1. **Views/Slicing** (Highest impact)
   - Current: All operations create new tensors
   - Target: Zero-copy views for indexing/slicing
   - Savings: 50-90% memory reduction in typical workflows

2. **Small Tensor Optimization**
   - Current: Heap allocation for all sizes
   - Target: Stack allocation for small tensors (<= 128 bytes)
   - Savings: Faster allocation/deallocation

3. **Copy-on-Write**
   - Current: Clone always copies
   - Target: Share data until mutation
   - Savings: Memory and time for read-heavy workloads

### Memory Profiling Results

To be measured:
- [ ] Peak memory usage for common operations
- [ ] Allocation count per operation
- [ ] Cache hit rates
- [ ] Memory bandwidth utilization

---

## Known Bottlenecks

### 1. Memory Allocation

**Impact**: High
**Operations Affected**: All creating new tensors
**Symptoms**: Allocation shows up in profiler
**Mitigation**: Memory pooling (planned)

### 2. No Broadcasting

**Impact**: Medium
**Operations Affected**: Element-wise ops with different shapes
**Symptoms**: Operations fail that work in PyTorch
**Mitigation**: Implement broadcasting (Phase 4)

### 3. No Views

**Impact**: High
**Operations Affected**: Indexing, slicing, transpose
**Symptoms**: Unnecessary copies
**Mitigation**: Implement zero-copy views (Phase 4)

### 4. Scalar Loop Overhead

**Impact**: Medium (for small tensors)
**Operations Affected**: Element-wise operations
**Symptoms**: Slower than expected for small inputs
**Mitigation**: SIMD vectorization (in progress)

### 5. No GPU Support

**Impact**: Critical (for large-scale ML)
**Operations Affected**: All
**Symptoms**: Can't match GPU PyTorch
**Mitigation**: GPU backend (long-term goal)

---

## Future Optimizations

### Short-term (Phase 3)

- [x] Comprehensive benchmarking suite
- [ ] Criterion baseline comparisons
- [ ] Profile hot paths with perf
- [ ] Identify top 3 bottlenecks
- [ ] Basic SIMD for element-wise ops

### Medium-term (Phase 4)

- [ ] Parallel processing with rayon (>10k elements)
- [ ] Memory pooling for allocations
- [ ] In-place operation variants
- [ ] Zero-copy views and slicing
- [ ] Broadcasting support

### Long-term (Future)

- [ ] GPU support (CUDA/ROCm/WebGPU)
- [ ] Custom allocator integration
- [ ] Advanced SIMD (AVX-512, NEON)
- [ ] Distributed operations
- [ ] JIT compilation for op fusion

---

## Benchmarking Results

### To Be Completed

Once RustTorch Python bindings are fully integrated:

1. Run `python benchmarks/compare_pytorch.py`
2. Collect results for all operation categories
3. Create comparison tables
4. Identify operations where RustTorch excels
5. Identify areas needing optimization

**Template for Results:**

```
Operation: add (1000x1000 tensors)
PyTorch CPU:  X.XX ms
RustTorch:    Y.YY ms
Speedup:      Z.ZZx
```

---

## References

### Tools

- [Criterion.rs](https://github.com/bheisler/criterion.rs) - Rust benchmarking
- [perf](https://perf.wiki.kernel.org/) - Linux profiler
- [Valgrind](https://valgrind.org/) - Memory profiler
- [cargo-flamegraph](https://github.com/flamegraph-rs/flamegraph) - Flamegraph generation

### Resources

- [The Rust Performance Book](https://nnethercote.github.io/perf-book/)
- [Rayon Parallel Programming](https://docs.rs/rayon/)
- [SIMD in Rust](https://rust-lang.github.io/packed_simd/)
- [PyTorch Performance Tuning](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

---

**Author**: Theodore Tennant (teddytennant@icloud.com)
**License**: BSD-3-Clause (following PyTorch)
**Repository**: https://github.com/teddytennant/rusttorch
