# Known Issues and Future Work

**Author**: Theodore Tennant (@teddytennant)
**Last Updated**: November 29, 2025

This document tracks known limitations, issues, and future work for RustTorch.

## Current Limitations

### 1. Batched Matrix Multiplication Not Implemented

**Status**: Not Implemented
**Priority**: Medium
**Affects**: Matrix operations (matmul)

**Description**:
Matrix multiplication currently only supports 2D tensors. Higher dimensional tensors (3D, 4D) that require batched matrix multiplication are not yet supported.

**Example**:
```rust
// This works
let a = Tensor::ones(&[2, 3], DType::Float32);
let b = Tensor::ones(&[3, 4], DType::Float32);
let c = matmul(&a, &b).unwrap(); // ✅ OK

// This doesn't work yet
let a = Tensor::ones(&[batch, 2, 3], DType::Float32);
let b = Tensor::ones(&[batch, 3, 4], DType::Float32);
let c = matmul(&a, &b); // ❌ Error: "Batched matrix multiplication not yet implemented"
```

**Workaround**:
Iterate over batches manually and call matmul on each 2D slice.

**Implementation Notes**:
- Need to handle batch dimensions correctly
- Should support broadcasting across batch dimensions
- PyTorch supports complex broadcasting rules for matmul
- Performance considerations for large batches

**Related Files**:
- `rusttorch-core/src/ops/matrix.rs:87-94`

---

### 2. No Broadcasting Support

**Status**: Not Implemented
**Priority**: High
**Affects**: Element-wise operations

**Description**:
Element-wise operations require tensors to have exactly the same shape. PyTorch's broadcasting rules are not yet implemented.

**Example**:
```rust
// This works
let a = Tensor::ones(&[2, 3], DType::Float32);
let b = Tensor::ones(&[2, 3], DType::Float32);
let c = add(&a, &b).unwrap(); // ✅ OK

// This doesn't work yet (should broadcast)
let a = Tensor::ones(&[2, 3], DType::Float32);
let b = Tensor::ones(&[3], DType::Float32);  // Should broadcast to [2, 3]
let c = add(&a, &b); // ❌ Error: Shape mismatch
```

**Workaround**:
Manually expand tensors to matching shapes before operations.

**Implementation Notes**:
- NumPy/PyTorch broadcasting rules are complex
- Need to handle dimension alignment (trailing dimensions)
- Memory efficiency considerations
- See: https://numpy.org/doc/stable/user/basics.broadcasting.html

**Related Files**:
- `rusttorch-core/src/ops/elementwise.rs`
- `rusttorch-core/src/tensor/shape.rs`

---

### 3. No Zero-Copy Views/Slicing

**Status**: Not Implemented
**Priority**: High
**Affects**: All operations, memory usage

**Description**:
All operations create new tensors. There's no support for zero-copy views or slicing, leading to unnecessary memory allocations.

**Example**:
```rust
let a = Tensor::ones(&[100, 100], DType::Float32);

// This should be a view (no copy), but creates a new tensor
let slice = a.slice(0..50, 0..50); // ❌ Not implemented

// Transpose creates a new tensor (should be a view)
let b = transpose(&a); // ❌ Allocates new memory
```

**Workaround**:
None currently - all operations allocate.

**Implementation Notes**:
- Need to distinguish between owned tensors and views
- Views need lifetime tracking
- Copy-on-write semantics for mutations
- Complex but critical for performance

**Related Files**:
- `rusttorch-core/src/tensor/mod.rs:123-126` (TensorView stub)

---

### 4. No In-Place Operations

**Status**: Not Implemented
**Priority**: Medium
**Affects**: Memory efficiency

**Description**:
All operations create new tensors. In-place operations (e.g., `add_`, `mul_`) would reduce allocations.

**Example**:
```rust
let mut a = Tensor::ones(&[1000, 1000], DType::Float32);
let b = Tensor::ones(&[1000, 1000], DType::Float32);

// Currently requires allocation
let c = add(&a, &b).unwrap(); // Allocates new tensor

// Desired: in-place operation
a.add_(&b); // ❌ Not implemented - would modify a directly
```

**Workaround**:
Accept the extra allocations.

**Implementation Notes**:
- Need mutable tensor operations
- Requires careful handling with Arc (check ref count)
- Should error if tensor is shared (ref count > 1)

**Related Files**:
- `rusttorch-core/src/ops/elementwise.rs`

---

### 5. Limited Python Bindings

**Status**: Partially Implemented
**Priority**: Medium
**Affects**: Python API

**Description**:
Python bindings exist but don't expose all operations yet. Matrix operations (matmul, transpose, reshape) need to be added.

**Example**:
```python
import rusttorch

# These work
t = rusttorch.Tensor.zeros([2, 3])
result = rusttorch.add(t1, t2)
activated = rusttorch.relu(t)

# These don't work yet
c = rusttorch.matmul(a, b)  # ❌ Not exposed
t_transposed = rusttorch.transpose(t)  # ❌ Not exposed
reshaped = rusttorch.reshape(t, [6, 1])  # ❌ Not exposed
```

**Workaround**:
Use Rust API directly or wait for Python bindings.

**Implementation Notes**:
- Need to add PyO3 wrappers for new operations
- Should match PyTorch's Python API
- Error handling for Python exceptions

**Related Files**:
- `rusttorch-py/src/lib.rs`

---

### 6. No SIMD Vectorization

**Status**: Not Implemented
**Priority**: High (Performance)
**Affects**: Element-wise operations performance

**Description**:
Operations don't use explicit SIMD instructions, relying only on ndarray's implicit vectorization and compiler auto-vectorization.

**Impact**:
Missing 2-4x potential speedup on element-wise operations.

**Implementation Notes**:
- Use `std::simd` (when stable) or `packed_simd`
- Target AVX2/AVX-512 on x86, NEON on ARM
- Need fallback for non-SIMD platforms
- See PERFORMANCE.md for implementation examples

**Related Files**:
- `rusttorch-core/src/ops/elementwise.rs`
- `rusttorch-core/src/ops/activation.rs`

---

### 7. No Parallel Processing

**Status**: Not Implemented
**Priority**: High (Performance)
**Affects**: Large tensor operations

**Description**:
Operations are single-threaded. Large tensors could benefit from parallel processing with Rayon.

**Impact**:
Missing near-linear speedup with core count for large operations.

**Implementation Notes**:
- Use Rayon's parallel iterators
- Set threshold for parallelization (e.g., 10k elements)
- Avoid overhead for small tensors
- See PERFORMANCE.md for implementation examples

**Related Files**:
- `rusttorch-core/src/ops/elementwise.rs`
- `rusttorch-core/src/ops/reduction.rs`

---

### 8. No GPU Support

**Status**: Not Implemented
**Priority**: Low (Future)
**Affects**: Performance on GPU workloads

**Description**:
RustTorch is CPU-only. No CUDA, ROCm, or WebGPU support.

**Impact**:
Cannot compete with PyTorch GPU for large-scale ML training.

**Implementation Notes**:
- Could use `wgpu` for cross-platform GPU
- Or CUDA bindings for NVIDIA-specific
- Major architectural change required
- Long-term goal

---

## Feature Requests

### Memory Pool Allocator

**Priority**: Medium
**Impact**: Reduce allocation overhead

Implement a memory pool for frequently allocated tensor sizes to reduce malloc/free overhead.

### Copy-on-Write Tensors

**Priority**: Medium
**Impact**: Memory efficiency

Implement copy-on-write semantics so cloning tensors doesn't copy data until mutation.

### Loss Functions

**Priority**: Medium
**Impact**: Completeness

Implement common loss functions:
- MSE (Mean Squared Error)
- Cross-Entropy
- L1/L2 Loss
- Binary Cross-Entropy

### Advanced Activation Functions

**Priority**: Low
**Impact**: Completeness

Add more activation variants:
- Swish
- Mish
- Hard Swish
- ELU, SELU

### Optimizer Update Rules

**Priority**: Low
**Impact**: Training support

Implement optimizer updates:
- SGD with momentum
- Adam
- AdamW
- RMSprop

---

## How to Report Issues

When reporting issues or requesting features:

1. Check this file first to see if it's already known
2. Create a GitHub issue with:
   - Clear description
   - Code example (if applicable)
   - Expected vs actual behavior
   - System information (OS, Rust version)

3. Use appropriate labels:
   - `bug` - Something is broken
   - `enhancement` - New feature
   - `performance` - Performance improvement
   - `documentation` - Docs need improvement

---

## Contributing

Want to fix one of these issues? Great!

1. Comment on the issue (or create one) to claim it
2. Fork the repository
3. Create a feature branch
4. Implement with tests
5. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

**Repository**: https://github.com/teddytennant/rusttorch
**Author**: Theodore Tennant (teddytennant@icloud.com)
