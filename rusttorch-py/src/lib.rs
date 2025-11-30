//! RustTorch Python Bindings
//!
//! PyO3-based Python bindings for the RustTorch library.
//! Provides a PyTorch-like API for high-performance tensor operations.

use pyo3::prelude::*;
use pyo3::types::PyList;
use numpy::{PyArray1, PyReadonlyArray2, PyReadonlyArrayDyn};
use rusttorch_core::{Tensor as RustTensor, DType};

/// Python wrapper for Rust Tensor
#[pyclass(name = "Tensor")]
struct PyTensor {
    inner: RustTensor,
}

#[pymethods]
impl PyTensor {
    /// Create a new tensor filled with zeros
    #[staticmethod]
    fn zeros(py: Python, shape: &PyList) -> PyResult<Self> {
        let shape_vec: Vec<usize> = shape.extract()?;
        Ok(PyTensor {
            inner: RustTensor::zeros(&shape_vec, DType::Float32),
        })
    }

    /// Create a new tensor filled with ones
    #[staticmethod]
    fn ones(py: Python, shape: &PyList) -> PyResult<Self> {
        let shape_vec: Vec<usize> = shape.extract()?;
        Ok(PyTensor {
            inner: RustTensor::ones(&shape_vec, DType::Float32),
        })
    }

    /// Create tensor from NumPy array (2D float32)
    #[staticmethod]
    fn from_numpy(py: Python, array: PyReadonlyArrayDyn<f32>) -> PyResult<Self> {
        let shape: Vec<usize> = array.shape().to_vec();
        let data: Vec<f32> = array.as_slice()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be contiguous"))?
            .to_vec();
        Ok(PyTensor {
            inner: RustTensor::from_vec(data, &shape),
        })
    }

    /// Get the shape of the tensor
    fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }

    /// Get the number of dimensions
    fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    /// Get the total number of elements
    fn numel(&self) -> usize {
        self.inner.numel()
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!("Tensor(shape={:?}, dtype={})", self.inner.shape(), self.inner.dtype())
    }
}

/// Add two tensors element-wise
#[pyfunction]
fn add(a: &PyTensor, b: &PyTensor) -> PyTensor {
    PyTensor {
        inner: rusttorch_core::ops::add(&a.inner, &b.inner),
    }
}

/// Multiply two tensors element-wise
#[pyfunction]
fn mul(a: &PyTensor, b: &PyTensor) -> PyTensor {
    PyTensor {
        inner: rusttorch_core::ops::mul(&a.inner, &b.inner),
    }
}

/// ReLU activation function
#[pyfunction]
fn relu(tensor: &PyTensor) -> PyTensor {
    PyTensor {
        inner: rusttorch_core::ops::relu(&tensor.inner),
    }
}

/// Matrix multiplication
#[pyfunction]
fn matmul(a: &PyTensor, b: &PyTensor) -> PyResult<PyTensor> {
    match rusttorch_core::ops::matmul(&a.inner, &b.inner) {
        Ok(result) => Ok(PyTensor { inner: result }),
        Err(e) => Err(pyo3::exceptions::PyValueError::new_err(e)),
    }
}

/// Transpose a tensor
#[pyfunction]
fn transpose(tensor: &PyTensor) -> PyTensor {
    PyTensor {
        inner: rusttorch_core::ops::transpose(&tensor.inner),
    }
}

/// Reshape a tensor
#[pyfunction]
fn reshape(tensor: &PyTensor, shape: &PyList) -> PyResult<PyTensor> {
    let shape_vec: Vec<usize> = shape.extract()?;
    match rusttorch_core::ops::reshape(&tensor.inner, &shape_vec) {
        Ok(result) => Ok(PyTensor { inner: result }),
        Err(e) => Err(pyo3::exceptions::PyValueError::new_err(e)),
    }
}

/// Sum reduction
#[pyfunction]
fn sum(tensor: &PyTensor) -> f64 {
    rusttorch_core::ops::sum(&tensor.inner)
}

/// Mean reduction
#[pyfunction]
fn mean(tensor: &PyTensor) -> f64 {
    rusttorch_core::ops::mean(&tensor.inner)
}

/// Sigmoid activation
#[pyfunction]
fn sigmoid(tensor: &PyTensor) -> PyTensor {
    PyTensor {
        inner: rusttorch_core::ops::sigmoid(&tensor.inner),
    }
}

/// Tanh activation
#[pyfunction]
fn tanh(tensor: &PyTensor) -> PyTensor {
    PyTensor {
        inner: rusttorch_core::ops::tanh(&tensor.inner),
    }
}

/// GELU activation
#[pyfunction]
fn gelu(tensor: &PyTensor) -> PyTensor {
    PyTensor {
        inner: rusttorch_core::ops::gelu(&tensor.inner),
    }
}

/// Leaky ReLU activation
#[pyfunction]
fn leaky_relu(tensor: &PyTensor, alpha: f32) -> PyTensor {
    PyTensor {
        inner: rusttorch_core::ops::leaky_relu(&tensor.inner, alpha),
    }
}

/// ELU activation
#[pyfunction]
fn elu(tensor: &PyTensor, alpha: f32) -> PyTensor {
    PyTensor {
        inner: rusttorch_core::ops::elu(&tensor.inner, alpha),
    }
}

/// SELU activation
#[pyfunction]
fn selu(tensor: &PyTensor) -> PyTensor {
    PyTensor {
        inner: rusttorch_core::ops::selu(&tensor.inner),
    }
}

/// Swish/SiLU activation
#[pyfunction]
fn swish(tensor: &PyTensor) -> PyTensor {
    PyTensor {
        inner: rusttorch_core::ops::swish(&tensor.inner),
    }
}

/// Mish activation
#[pyfunction]
fn mish(tensor: &PyTensor) -> PyTensor {
    PyTensor {
        inner: rusttorch_core::ops::mish(&tensor.inner),
    }
}

/// Softplus activation
#[pyfunction]
fn softplus(tensor: &PyTensor) -> PyTensor {
    PyTensor {
        inner: rusttorch_core::ops::softplus(&tensor.inner),
    }
}

/// Softsign activation
#[pyfunction]
fn softsign(tensor: &PyTensor) -> PyTensor {
    PyTensor {
        inner: rusttorch_core::ops::softsign(&tensor.inner),
    }
}

/// Softmax activation
#[pyfunction]
fn softmax(tensor: &PyTensor, dim: usize) -> PyTensor {
    PyTensor {
        inner: rusttorch_core::ops::softmax(&tensor.inner, dim),
    }
}

// ============================================================================
// Loss Functions
// ============================================================================

/// Mean Squared Error loss
#[pyfunction]
fn mse_loss(predictions: &PyTensor, targets: &PyTensor) -> PyResult<f64> {
    rusttorch_core::ops::mse_loss(&predictions.inner, &targets.inner)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))
}

/// Mean Absolute Error (L1) loss
#[pyfunction]
fn l1_loss(predictions: &PyTensor, targets: &PyTensor) -> PyResult<f64> {
    rusttorch_core::ops::l1_loss(&predictions.inner, &targets.inner)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))
}

/// Smooth L1 loss (Huber loss)
#[pyfunction]
fn smooth_l1_loss(predictions: &PyTensor, targets: &PyTensor, beta: f64) -> PyResult<f64> {
    rusttorch_core::ops::smooth_l1_loss(&predictions.inner, &targets.inner, beta)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))
}

/// Binary Cross-Entropy loss
#[pyfunction]
fn binary_cross_entropy_loss(predictions: &PyTensor, targets: &PyTensor, epsilon: f64) -> PyResult<f64> {
    rusttorch_core::ops::binary_cross_entropy_loss(&predictions.inner, &targets.inner, epsilon)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))
}

/// Cross-Entropy loss
#[pyfunction]
fn cross_entropy_loss(predictions: &PyTensor, targets: &PyTensor, epsilon: f64) -> PyResult<f64> {
    rusttorch_core::ops::cross_entropy_loss(&predictions.inner, &targets.inner, epsilon)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))
}

// ============================================================================
// Optimizers
// ============================================================================

/// SGD parameter update
#[pyfunction]
fn sgd_update(params: &PyTensor, gradients: &PyTensor, learning_rate: f64) -> PyResult<PyTensor> {
    rusttorch_core::ops::sgd_update(&params.inner, &gradients.inner, learning_rate)
        .map(|result| PyTensor { inner: result })
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))
}

/// SGD with Momentum parameter update
#[pyfunction]
fn sgd_momentum_update(
    params: &PyTensor,
    gradients: &PyTensor,
    velocity: &PyTensor,
    learning_rate: f64,
    momentum: f64,
) -> PyResult<(PyTensor, PyTensor)> {
    rusttorch_core::ops::sgd_momentum_update(
        &params.inner,
        &gradients.inner,
        &velocity.inner,
        learning_rate,
        momentum,
    )
    .map(|(new_params, new_velocity)| {
        (
            PyTensor { inner: new_params },
            PyTensor { inner: new_velocity },
        )
    })
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))
}

/// Adam optimizer parameter update
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn adam_update(
    params: &PyTensor,
    gradients: &PyTensor,
    m: &PyTensor,
    v: &PyTensor,
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    timestep: usize,
) -> PyResult<(PyTensor, PyTensor, PyTensor)> {
    rusttorch_core::ops::adam_update(
        &params.inner,
        &gradients.inner,
        &m.inner,
        &v.inner,
        learning_rate,
        beta1,
        beta2,
        epsilon,
        timestep,
    )
    .map(|(new_params, new_m, new_v)| {
        (
            PyTensor { inner: new_params },
            PyTensor { inner: new_m },
            PyTensor { inner: new_v },
        )
    })
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))
}

/// AdamW optimizer parameter update
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn adamw_update(
    params: &PyTensor,
    gradients: &PyTensor,
    m: &PyTensor,
    v: &PyTensor,
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    weight_decay: f64,
    timestep: usize,
) -> PyResult<(PyTensor, PyTensor, PyTensor)> {
    rusttorch_core::ops::adamw_update(
        &params.inner,
        &gradients.inner,
        &m.inner,
        &v.inner,
        learning_rate,
        beta1,
        beta2,
        epsilon,
        weight_decay,
        timestep,
    )
    .map(|(new_params, new_m, new_v)| {
        (
            PyTensor { inner: new_params },
            PyTensor { inner: new_m },
            PyTensor { inner: new_v },
        )
    })
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))
}

// ============================================================================
// Additional Element-wise Operations
// ============================================================================

/// Subtract two tensors element-wise
#[pyfunction]
fn sub(a: &PyTensor, b: &PyTensor) -> PyTensor {
    PyTensor {
        inner: rusttorch_core::ops::sub(&a.inner, &b.inner),
    }
}

/// Divide two tensors element-wise
#[pyfunction]
fn div(a: &PyTensor, b: &PyTensor) -> PyTensor {
    PyTensor {
        inner: rusttorch_core::ops::div(&a.inner, &b.inner),
    }
}

/// Add scalar to tensor
#[pyfunction]
fn add_scalar(tensor: &PyTensor, scalar: f32) -> PyTensor {
    PyTensor {
        inner: rusttorch_core::ops::add_scalar(&tensor.inner, scalar),
    }
}

/// Multiply tensor by scalar
#[pyfunction]
fn mul_scalar(tensor: &PyTensor, scalar: f32) -> PyTensor {
    PyTensor {
        inner: rusttorch_core::ops::mul_scalar(&tensor.inner, scalar),
    }
}

// ============================================================================
// Broadcasting Operations
// ============================================================================

/// Element-wise addition with broadcasting
#[pyfunction]
fn add_broadcast(a: &PyTensor, b: &PyTensor) -> PyResult<PyTensor> {
    rusttorch_core::ops::add_broadcast(&a.inner, &b.inner)
        .map(|result| PyTensor { inner: result })
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))
}

/// Element-wise multiplication with broadcasting
#[pyfunction]
fn mul_broadcast(a: &PyTensor, b: &PyTensor) -> PyResult<PyTensor> {
    rusttorch_core::ops::mul_broadcast(&a.inner, &b.inner)
        .map(|result| PyTensor { inner: result })
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))
}

/// Element-wise subtraction with broadcasting
#[pyfunction]
fn sub_broadcast(a: &PyTensor, b: &PyTensor) -> PyResult<PyTensor> {
    rusttorch_core::ops::sub_broadcast(&a.inner, &b.inner)
        .map(|result| PyTensor { inner: result })
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))
}

/// Element-wise division with broadcasting
#[pyfunction]
fn div_broadcast(a: &PyTensor, b: &PyTensor) -> PyResult<PyTensor> {
    rusttorch_core::ops::div_broadcast(&a.inner, &b.inner)
        .map(|result| PyTensor { inner: result })
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))
}

// ============================================================================
// SIMD-Optimized Operations
// ============================================================================

/// SIMD-optimized element-wise addition
#[pyfunction]
fn add_simd(a: &PyTensor, b: &PyTensor) -> PyTensor {
    PyTensor {
        inner: rusttorch_core::ops::add_simd(&a.inner, &b.inner),
    }
}

/// SIMD-optimized element-wise multiplication
#[pyfunction]
fn mul_simd(a: &PyTensor, b: &PyTensor) -> PyTensor {
    PyTensor {
        inner: rusttorch_core::ops::mul_simd(&a.inner, &b.inner),
    }
}

/// SIMD-optimized ReLU activation
#[pyfunction]
fn relu_simd(tensor: &PyTensor) -> PyTensor {
    PyTensor {
        inner: rusttorch_core::ops::relu_simd(&tensor.inner),
    }
}

/// SIMD-optimized scalar multiplication
#[pyfunction]
fn mul_scalar_simd(tensor: &PyTensor, scalar: f32) -> PyTensor {
    PyTensor {
        inner: rusttorch_core::ops::mul_scalar_simd(&tensor.inner, scalar),
    }
}

/// Fused multiply-add (SIMD optimized)
#[pyfunction]
fn fused_multiply_add(a: &PyTensor, b: &PyTensor, c: &PyTensor) -> PyTensor {
    PyTensor {
        inner: rusttorch_core::ops::fused_multiply_add(&a.inner, &b.inner, &c.inner),
    }
}

// ============================================================================
// Data Loading and Preprocessing
// ============================================================================

/// Load data from CSV file
#[pyfunction]
fn load_csv(path: String, has_header: bool, delimiter: char) -> PyResult<PyTensor> {
    match rusttorch_core::data::load_csv(path, has_header, delimiter) {
        Ok(tensor) => Ok(PyTensor { inner: tensor }),
        Err(e) => Err(pyo3::exceptions::PyIOError::new_err(e)),
    }
}

/// Normalize tensor data (z-score normalization)
#[pyfunction]
fn normalize(tensor: &PyTensor) -> (PyTensor, f64, f64) {
    let (normalized, mean, std) = rusttorch_core::data::normalize(&tensor.inner);
    (PyTensor { inner: normalized }, mean, std)
}

/// Create batches from dataset
#[pyfunction]
fn create_batches(data: &PyTensor, batch_size: usize, drop_last: bool) -> Vec<PyTensor> {
    rusttorch_core::data::create_batches(&data.inner, batch_size, drop_last)
        .into_iter()
        .map(|tensor| PyTensor { inner: tensor })
        .collect()
}

/// Shuffle indices for random batching
#[pyfunction]
fn shuffle_indices(num_samples: usize) -> Vec<usize> {
    rusttorch_core::data::shuffle_indices(num_samples)
}

/// Split data into train/validation/test sets
#[pyfunction]
fn train_val_test_split(
    data: &PyTensor,
    train_ratio: f64,
    val_ratio: f64,
    shuffle: bool,
) -> (PyTensor, PyTensor, PyTensor) {
    let (train, val, test) =
        rusttorch_core::data::train_val_test_split(&data.inner, train_ratio, val_ratio, shuffle);
    (
        PyTensor { inner: train },
        PyTensor { inner: val },
        PyTensor { inner: test },
    )
}

/// Python module definition
#[pymodule]
fn rusttorch(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyTensor>()?;

    // Element-wise operations
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(mul, m)?)?;
    m.add_function(wrap_pyfunction!(sub, m)?)?;
    m.add_function(wrap_pyfunction!(div, m)?)?;
    m.add_function(wrap_pyfunction!(add_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(mul_scalar, m)?)?;

    // Matrix operations
    m.add_function(wrap_pyfunction!(matmul, m)?)?;
    m.add_function(wrap_pyfunction!(transpose, m)?)?;
    m.add_function(wrap_pyfunction!(reshape, m)?)?;

    // Reduction operations
    m.add_function(wrap_pyfunction!(sum, m)?)?;
    m.add_function(wrap_pyfunction!(mean, m)?)?;

    // Activation functions
    m.add_function(wrap_pyfunction!(relu, m)?)?;
    m.add_function(wrap_pyfunction!(sigmoid, m)?)?;
    m.add_function(wrap_pyfunction!(tanh, m)?)?;
    m.add_function(wrap_pyfunction!(gelu, m)?)?;
    m.add_function(wrap_pyfunction!(leaky_relu, m)?)?;
    m.add_function(wrap_pyfunction!(elu, m)?)?;
    m.add_function(wrap_pyfunction!(selu, m)?)?;
    m.add_function(wrap_pyfunction!(swish, m)?)?;
    m.add_function(wrap_pyfunction!(mish, m)?)?;
    m.add_function(wrap_pyfunction!(softplus, m)?)?;
    m.add_function(wrap_pyfunction!(softsign, m)?)?;
    m.add_function(wrap_pyfunction!(softmax, m)?)?;

    // Loss functions
    m.add_function(wrap_pyfunction!(mse_loss, m)?)?;
    m.add_function(wrap_pyfunction!(l1_loss, m)?)?;
    m.add_function(wrap_pyfunction!(smooth_l1_loss, m)?)?;
    m.add_function(wrap_pyfunction!(binary_cross_entropy_loss, m)?)?;
    m.add_function(wrap_pyfunction!(cross_entropy_loss, m)?)?;

    // Optimizers
    m.add_function(wrap_pyfunction!(sgd_update, m)?)?;
    m.add_function(wrap_pyfunction!(sgd_momentum_update, m)?)?;
    m.add_function(wrap_pyfunction!(adam_update, m)?)?;
    m.add_function(wrap_pyfunction!(adamw_update, m)?)?;

    // Broadcasting operations
    m.add_function(wrap_pyfunction!(add_broadcast, m)?)?;
    m.add_function(wrap_pyfunction!(mul_broadcast, m)?)?;
    m.add_function(wrap_pyfunction!(sub_broadcast, m)?)?;
    m.add_function(wrap_pyfunction!(div_broadcast, m)?)?;

    // SIMD-optimized operations
    m.add_function(wrap_pyfunction!(add_simd, m)?)?;
    m.add_function(wrap_pyfunction!(mul_simd, m)?)?;
    m.add_function(wrap_pyfunction!(relu_simd, m)?)?;
    m.add_function(wrap_pyfunction!(mul_scalar_simd, m)?)?;
    m.add_function(wrap_pyfunction!(fused_multiply_add, m)?)?;

    // Data loading and preprocessing
    m.add_function(wrap_pyfunction!(load_csv, m)?)?;
    m.add_function(wrap_pyfunction!(normalize, m)?)?;
    m.add_function(wrap_pyfunction!(create_batches, m)?)?;
    m.add_function(wrap_pyfunction!(shuffle_indices, m)?)?;
    m.add_function(wrap_pyfunction!(train_val_test_split, m)?)?;

    // Module metadata
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__author__", "Theodore Tennant (@teddytennant)")?;

    Ok(())
}
