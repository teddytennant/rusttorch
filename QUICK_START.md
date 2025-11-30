# RustTorch Quick Start Guide

## Installation

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone the repository
cd /workspace/rusttorch

# Build Rust core
cd rusttorch-core
cargo build --release
cargo test

# Build Python bindings
cd ../rusttorch-py
pip install maturin
maturin develop --release

# Verify installation
python -c "import rusttorch; print('RustTorch version:', rusttorch.__version__)"
```

## Basic Usage

### Creating Tensors

```python
import rusttorch as rt

# Create tensors
zeros = rt.Tensor.zeros([3, 4])      # 3x4 tensor of zeros
ones = rt.Tensor.ones([2, 2])        # 2x2 tensor of ones

# Check properties
print(zeros.shape())  # [3, 4]
print(zeros.ndim())   # 2
print(zeros.numel())  # 12
```

### Element-wise Operations

```python
# Basic operations
a = rt.Tensor.ones([100, 100])
b = rt.Tensor.ones([100, 100])

c = rt.add(a, b)        # Addition
d = rt.mul(a, b)        # Multiplication
e = rt.sub(a, b)        # Subtraction
f = rt.div(a, b)        # Division

# Scalar operations
g = rt.add_scalar(a, 5.0)   # Add 5 to all elements
h = rt.mul_scalar(a, 2.0)   # Multiply all by 2

# Large tensors automatically use parallel execution
large = rt.Tensor.ones([500, 500])  # 250k elements
result = rt.add(large, large)        # Uses Rayon parallelization
```

### Matrix Operations

```python
# Matrix multiplication
x = rt.Tensor.ones([2, 3])
y = rt.Tensor.ones([3, 4])
z = rt.matmul(x, y)  # 2x4 result

# Transpose
x_t = rt.transpose(x)  # 3x2 result

# Reshape
flat = rt.reshape(z, [8])  # Flatten to 1D
```

### Activation Functions

```python
import rusttorch as rt

x = rt.Tensor.from_vec([-2.0, -1.0, 0.0, 1.0, 2.0], [5])

# Basic activations
relu_out = rt.relu(x)
sigmoid_out = rt.sigmoid(x)
tanh_out = rt.tanh(x)

# Advanced activations
gelu_out = rt.gelu(x)
swish_out = rt.swish(x)
mish_out = rt.mish(x)

# Parametric activations
leaky_out = rt.leaky_relu(x, alpha=0.01)
elu_out = rt.elu(x, alpha=1.0)

# Self-normalizing
selu_out = rt.selu(x)

# Smooth activations
softplus_out = rt.softplus(x)
softsign_out = rt.softsign(x)

# Softmax (with dimension)
batch = rt.Tensor.ones([32, 10])  # Batch of 32, 10 classes
probs = rt.softmax(batch, dim=1)  # Softmax along class dimension
```

### Loss Functions

```python
# Mean Squared Error
pred = rt.Tensor.from_vec([1.0, 2.0, 3.0], [3])
target = rt.Tensor.from_vec([1.5, 2.5, 2.5], [3])
mse = rt.mse_loss(pred, target)
print(f"MSE Loss: {mse}")

# L1 Loss (Mean Absolute Error)
l1 = rt.l1_loss(pred, target)
print(f"L1 Loss: {l1}")

# Smooth L1 Loss (Huber)
smooth = rt.smooth_l1_loss(pred, target, beta=1.0)
print(f"Smooth L1 Loss: {smooth}")

# Binary Cross-Entropy
pred_probs = rt.Tensor.from_vec([0.9, 0.1, 0.8, 0.2], [4])
targets = rt.Tensor.from_vec([1.0, 0.0, 1.0, 0.0], [4])
bce = rt.binary_cross_entropy_loss(pred_probs, targets, epsilon=1e-7)
print(f"BCE Loss: {bce}")

# Cross-Entropy (multi-class)
pred_probs = rt.Tensor.from_vec([0.7, 0.2, 0.1, 0.1, 0.6, 0.3], [2, 3])
targets = rt.Tensor.from_vec([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [2, 3])
ce = rt.cross_entropy_loss(pred_probs, targets, epsilon=1e-7)
print(f"CE Loss: {ce}")
```

### Optimizers

#### SGD
```python
# Standard SGD
params = rt.Tensor.ones([1000])
grads = rt.Tensor.from_vec([0.01] * 1000, [1000])

new_params = rt.sgd_update(params, grads, learning_rate=0.1)
```

#### SGD with Momentum
```python
# Initialize velocity
velocity = rt.Tensor.zeros([1000])

# Training loop
for step in range(100):
    # Compute gradients (manual for now)
    grads = compute_gradients(params)  # Your gradient computation

    # Update with momentum
    params, velocity = rt.sgd_momentum_update(
        params, grads, velocity,
        learning_rate=0.01,
        momentum=0.9
    )
```

#### Adam
```python
# Initialize moment estimates
m = rt.Tensor.zeros([1000])  # First moment
v = rt.Tensor.zeros([1000])  # Second moment

# Training loop
for timestep in range(1, 1001):  # timestep starts at 1
    grads = compute_gradients(params)

    params, m, v = rt.adam_update(
        params, grads, m, v,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        timestep=timestep
    )
```

#### AdamW (Adam with Weight Decay)
```python
m = rt.Tensor.zeros([1000])
v = rt.Tensor.zeros([1000])

for timestep in range(1, 1001):
    grads = compute_gradients(params)

    params, m, v = rt.adamw_update(
        params, grads, m, v,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.01,  # L2 regularization
        timestep=timestep
    )
```

### Reductions

```python
x = rt.Tensor.ones([10, 20])

# Global reductions
total = rt.sum(x)    # Sum of all elements
avg = rt.mean(x)     # Mean of all elements

print(f"Total: {total}")  # 200.0
print(f"Average: {avg}")  # 1.0
```

## Complete Training Example

```python
import rusttorch as rt
import numpy as np

# Hyperparameters
input_size = 100
output_size = 10
batch_size = 32
learning_rate = 0.001
epochs = 100

# Initialize parameters (simple linear layer)
weights = rt.Tensor.ones([input_size, output_size])
bias = rt.Tensor.zeros([output_size])

# Adam optimizer states
m_w = rt.Tensor.zeros([input_size, output_size])
v_w = rt.Tensor.zeros([input_size, output_size])
m_b = rt.Tensor.zeros([output_size])
v_b = rt.Tensor.zeros([output_size])

# Training loop
for epoch in range(epochs):
    for batch in range(100):
        timestep = epoch * 100 + batch + 1

        # Forward pass (simplified - you'd compute this)
        # x @ weights + bias
        logits = compute_forward(weights, bias)  # Your forward pass
        probs = rt.softmax(logits, dim=1)

        # Compute loss
        loss = rt.cross_entropy_loss(probs, targets, epsilon=1e-7)

        # Compute gradients (manual - no autograd yet)
        grad_w, grad_b = compute_gradients()  # Your backward pass

        # Update weights with Adam
        weights, m_w, v_w = rt.adam_update(
            weights, grad_w, m_w, v_w,
            learning_rate, 0.9, 0.999, 1e-8, timestep
        )

        # Update bias with Adam
        bias, m_b, v_b = rt.adam_update(
            bias, grad_b, m_b, v_b,
            learning_rate, 0.9, 0.999, 1e-8, timestep
        )

        if batch % 10 == 0:
            print(f"Epoch {epoch}, Batch {batch}, Loss: {loss:.4f}")
```

## Performance Tips

### Parallel Execution
```python
# Small tensors: sequential execution (faster for small sizes)
small = rt.Tensor.ones([50, 50])  # 2,500 elements
result = rt.add(small, small)      # Sequential

# Large tensors: automatic parallel execution
large = rt.Tensor.ones([200, 200])  # 40,000 elements
result = rt.add(large, large)       # Parallel with Rayon

# Threshold is 10,000 elements
```

### Memory Efficiency
```python
# RustTorch uses Arc (reference counting)
# Cloning is cheap - just increments counter
x = rt.Tensor.ones([1000, 1000])
y = x  # Cheap clone via Arc

# But operations create new tensors
# TODO: In-place operations coming in future release
```

## Troubleshooting

### Import Error
```bash
# If "import rusttorch" fails, rebuild:
cd rusttorch-py
maturin develop --release
```

### Shape Mismatch Errors
```python
# Element-wise ops require same shape (no broadcasting yet)
a = rt.Tensor.ones([2, 3])
b = rt.Tensor.ones([2, 3])  # ✅ Same shape
c = rt.Tensor.ones([3, 2])  # ❌ Different shape

result = rt.add(a, b)  # ✅ Works
result = rt.add(a, c)  # ❌ Panic: shapes must match
```

### Type Errors
```python
# Some operations require floating-point tensors
x = rt.Tensor.ones([10])      # Float32 by default
y = rt.sigmoid(x)             # ✅ Works

# Division only for floats
x_int = rt.Tensor.zeros([10], dtype=rt.DType.Int32)
y = rt.div(x_int, x_int)      # ❌ Panic: division requires floats
```

## Feature Comparison

| Feature | Status | Notes |
|---------|--------|-------|
| Tensor creation | ✅ | zeros, ones, from_vec |
| Element-wise ops | ✅ | add, sub, mul, div, scalars |
| Matrix ops | ✅ | matmul, transpose, reshape |
| Activations | ✅ | 13 functions |
| Loss functions | ✅ | 5 functions |
| Optimizers | ✅ | SGD, Adam, AdamW |
| Parallel execution | ✅ | Rayon for large tensors |
| Broadcasting | ❌ | Coming soon |
| Autograd | ❌ | Manual gradients only |
| GPU support | ❌ | CPU only |

## Next Steps

- Read [PHASE5_COMPLETION.md](PHASE5_COMPLETION.md) for implementation details
- Check [PERFORMANCE.md](PERFORMANCE.md) for benchmarking guide
- See [RUSTTORCH_PLAN.md](RUSTTORCH_PLAN.md) for roadmap

## Support

- **Issues**: Report bugs on GitHub Issues
- **Discussions**: Ask questions in GitHub Discussions
- **Email**: teddytennant@icloud.com

---

*Happy coding with RustTorch!*
