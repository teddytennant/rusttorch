//! RustTorch Python Bindings
//!
//! PyO3-based Python bindings for the RustTorch library.
//! Provides a PyTorch-like API for high-performance tensor operations.

use numpy::PyReadonlyArrayDyn;
use pyo3::prelude::*;
use pyo3::types::PyList;
use rusttorch_core::{DType, Tensor as RustTensor};

/// Helper to convert TensorError to PyValueError
fn tensor_err(e: impl std::fmt::Display) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(format!("{}", e))
}

/// Python wrapper for Rust Tensor
#[pyclass(name = "Tensor")]
struct PyTensor {
    inner: RustTensor,
}

#[pymethods]
impl PyTensor {
    #[staticmethod]
    fn zeros(_py: Python, shape: &PyList) -> PyResult<Self> {
        let shape_vec: Vec<usize> = shape.extract()?;
        Ok(PyTensor {
            inner: RustTensor::zeros(&shape_vec, DType::Float32),
        })
    }

    #[staticmethod]
    fn ones(_py: Python, shape: &PyList) -> PyResult<Self> {
        let shape_vec: Vec<usize> = shape.extract()?;
        Ok(PyTensor {
            inner: RustTensor::ones(&shape_vec, DType::Float32),
        })
    }

    #[staticmethod]
    fn from_numpy(_py: Python, array: PyReadonlyArrayDyn<f32>) -> PyResult<Self> {
        let shape: Vec<usize> = array.shape().to_vec();
        let data: Vec<f32> = array
            .as_slice()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be contiguous"))?
            .to_vec();
        Ok(PyTensor {
            inner: RustTensor::from_vec(data, &shape),
        })
    }

    fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }
    fn ndim(&self) -> usize {
        self.inner.ndim()
    }
    fn numel(&self) -> usize {
        self.inner.numel()
    }

    fn __repr__(&self) -> String {
        format!(
            "Tensor(shape={:?}, dtype={})",
            self.inner.shape(),
            self.inner.dtype()
        )
    }
}

// Element-wise operations (all now return Result)
#[pyfunction]
fn add(a: &PyTensor, b: &PyTensor) -> PyResult<PyTensor> {
    Ok(PyTensor {
        inner: rusttorch_core::ops::add(&a.inner, &b.inner).map_err(tensor_err)?,
    })
}

#[pyfunction]
fn mul(a: &PyTensor, b: &PyTensor) -> PyResult<PyTensor> {
    Ok(PyTensor {
        inner: rusttorch_core::ops::mul(&a.inner, &b.inner).map_err(tensor_err)?,
    })
}

#[pyfunction]
fn sub(a: &PyTensor, b: &PyTensor) -> PyResult<PyTensor> {
    Ok(PyTensor {
        inner: rusttorch_core::ops::sub(&a.inner, &b.inner).map_err(tensor_err)?,
    })
}

#[pyfunction]
fn div(a: &PyTensor, b: &PyTensor) -> PyResult<PyTensor> {
    Ok(PyTensor {
        inner: rusttorch_core::ops::div(&a.inner, &b.inner).map_err(tensor_err)?,
    })
}

#[pyfunction]
fn add_scalar(tensor: &PyTensor, scalar: f32) -> PyTensor {
    PyTensor {
        inner: rusttorch_core::ops::add_scalar(&tensor.inner, scalar),
    }
}

#[pyfunction]
fn mul_scalar(tensor: &PyTensor, scalar: f32) -> PyTensor {
    PyTensor {
        inner: rusttorch_core::ops::mul_scalar(&tensor.inner, scalar),
    }
}

// Activation functions (most return Result)
#[pyfunction]
fn relu(tensor: &PyTensor) -> PyTensor {
    PyTensor {
        inner: rusttorch_core::ops::relu(&tensor.inner),
    }
}

#[pyfunction]
fn sigmoid(tensor: &PyTensor) -> PyResult<PyTensor> {
    Ok(PyTensor {
        inner: rusttorch_core::ops::sigmoid(&tensor.inner).map_err(tensor_err)?,
    })
}

#[pyfunction]
fn tanh(tensor: &PyTensor) -> PyResult<PyTensor> {
    Ok(PyTensor {
        inner: rusttorch_core::ops::tanh(&tensor.inner).map_err(tensor_err)?,
    })
}

#[pyfunction]
fn gelu(tensor: &PyTensor) -> PyResult<PyTensor> {
    Ok(PyTensor {
        inner: rusttorch_core::ops::gelu(&tensor.inner).map_err(tensor_err)?,
    })
}

#[pyfunction]
fn leaky_relu(tensor: &PyTensor, alpha: f32) -> PyTensor {
    PyTensor {
        inner: rusttorch_core::ops::leaky_relu(&tensor.inner, alpha),
    }
}

#[pyfunction]
fn elu(tensor: &PyTensor, alpha: f32) -> PyResult<PyTensor> {
    Ok(PyTensor {
        inner: rusttorch_core::ops::elu(&tensor.inner, alpha).map_err(tensor_err)?,
    })
}

#[pyfunction]
fn selu(tensor: &PyTensor) -> PyResult<PyTensor> {
    Ok(PyTensor {
        inner: rusttorch_core::ops::selu(&tensor.inner).map_err(tensor_err)?,
    })
}

#[pyfunction]
fn swish(tensor: &PyTensor) -> PyResult<PyTensor> {
    Ok(PyTensor {
        inner: rusttorch_core::ops::swish(&tensor.inner).map_err(tensor_err)?,
    })
}

#[pyfunction]
fn mish(tensor: &PyTensor) -> PyResult<PyTensor> {
    Ok(PyTensor {
        inner: rusttorch_core::ops::mish(&tensor.inner).map_err(tensor_err)?,
    })
}

#[pyfunction]
fn softplus(tensor: &PyTensor) -> PyResult<PyTensor> {
    Ok(PyTensor {
        inner: rusttorch_core::ops::softplus(&tensor.inner).map_err(tensor_err)?,
    })
}

#[pyfunction]
fn softsign(tensor: &PyTensor) -> PyResult<PyTensor> {
    Ok(PyTensor {
        inner: rusttorch_core::ops::softsign(&tensor.inner).map_err(tensor_err)?,
    })
}

#[pyfunction]
fn softmax(tensor: &PyTensor, dim: usize) -> PyResult<PyTensor> {
    Ok(PyTensor {
        inner: rusttorch_core::ops::softmax(&tensor.inner, dim).map_err(tensor_err)?,
    })
}

// Matrix operations
#[pyfunction]
fn matmul(a: &PyTensor, b: &PyTensor) -> PyResult<PyTensor> {
    Ok(PyTensor {
        inner: rusttorch_core::ops::matmul(&a.inner, &b.inner).map_err(tensor_err)?,
    })
}

#[pyfunction]
fn transpose(tensor: &PyTensor) -> PyTensor {
    PyTensor {
        inner: rusttorch_core::ops::transpose(&tensor.inner),
    }
}

#[pyfunction]
fn reshape(tensor: &PyTensor, shape: &PyList) -> PyResult<PyTensor> {
    let shape_vec: Vec<usize> = shape.extract()?;
    Ok(PyTensor {
        inner: rusttorch_core::ops::reshape(&tensor.inner, &shape_vec).map_err(tensor_err)?,
    })
}

// Reduction operations
#[pyfunction]
fn sum(tensor: &PyTensor) -> f64 {
    rusttorch_core::ops::sum(&tensor.inner)
}

#[pyfunction]
fn mean(tensor: &PyTensor) -> PyResult<f64> {
    rusttorch_core::ops::mean(&tensor.inner).map_err(tensor_err)
}

// Loss functions
#[pyfunction]
fn mse_loss(predictions: &PyTensor, targets: &PyTensor) -> PyResult<f64> {
    rusttorch_core::ops::mse_loss(&predictions.inner, &targets.inner).map_err(tensor_err)
}

#[pyfunction]
fn l1_loss(predictions: &PyTensor, targets: &PyTensor) -> PyResult<f64> {
    rusttorch_core::ops::l1_loss(&predictions.inner, &targets.inner).map_err(tensor_err)
}

#[pyfunction]
fn smooth_l1_loss(predictions: &PyTensor, targets: &PyTensor, beta: f64) -> PyResult<f64> {
    rusttorch_core::ops::smooth_l1_loss(&predictions.inner, &targets.inner, beta)
        .map_err(tensor_err)
}

#[pyfunction]
fn binary_cross_entropy_loss(
    predictions: &PyTensor,
    targets: &PyTensor,
    epsilon: f64,
) -> PyResult<f64> {
    rusttorch_core::ops::binary_cross_entropy_loss(&predictions.inner, &targets.inner, epsilon)
        .map_err(tensor_err)
}

#[pyfunction]
fn cross_entropy_loss(predictions: &PyTensor, targets: &PyTensor, epsilon: f64) -> PyResult<f64> {
    rusttorch_core::ops::cross_entropy_loss(&predictions.inner, &targets.inner, epsilon)
        .map_err(tensor_err)
}

// Optimizers
#[pyfunction]
fn sgd_update(params: &PyTensor, gradients: &PyTensor, learning_rate: f64) -> PyResult<PyTensor> {
    Ok(PyTensor {
        inner: rusttorch_core::ops::sgd_update(&params.inner, &gradients.inner, learning_rate)
            .map_err(tensor_err)?,
    })
}

#[pyfunction]
fn sgd_momentum_update(
    params: &PyTensor,
    gradients: &PyTensor,
    velocity: &PyTensor,
    learning_rate: f64,
    momentum: f64,
) -> PyResult<(PyTensor, PyTensor)> {
    let (p, v) = rusttorch_core::ops::sgd_momentum_update(
        &params.inner,
        &gradients.inner,
        &velocity.inner,
        learning_rate,
        momentum,
    )
    .map_err(tensor_err)?;
    Ok((PyTensor { inner: p }, PyTensor { inner: v }))
}

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
    let (p, nm, nv) = rusttorch_core::ops::adam_update(
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
    .map_err(tensor_err)?;
    Ok((
        PyTensor { inner: p },
        PyTensor { inner: nm },
        PyTensor { inner: nv },
    ))
}

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
    let (p, nm, nv) = rusttorch_core::ops::adamw_update(
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
    .map_err(tensor_err)?;
    Ok((
        PyTensor { inner: p },
        PyTensor { inner: nm },
        PyTensor { inner: nv },
    ))
}

// Broadcasting operations
#[pyfunction]
fn add_broadcast(a: &PyTensor, b: &PyTensor) -> PyResult<PyTensor> {
    Ok(PyTensor {
        inner: rusttorch_core::ops::add_broadcast(&a.inner, &b.inner).map_err(tensor_err)?,
    })
}

#[pyfunction]
fn mul_broadcast(a: &PyTensor, b: &PyTensor) -> PyResult<PyTensor> {
    Ok(PyTensor {
        inner: rusttorch_core::ops::mul_broadcast(&a.inner, &b.inner).map_err(tensor_err)?,
    })
}

#[pyfunction]
fn sub_broadcast(a: &PyTensor, b: &PyTensor) -> PyResult<PyTensor> {
    Ok(PyTensor {
        inner: rusttorch_core::ops::sub_broadcast(&a.inner, &b.inner).map_err(tensor_err)?,
    })
}

#[pyfunction]
fn div_broadcast(a: &PyTensor, b: &PyTensor) -> PyResult<PyTensor> {
    Ok(PyTensor {
        inner: rusttorch_core::ops::div_broadcast(&a.inner, &b.inner).map_err(tensor_err)?,
    })
}

// SIMD-optimized operations (all now return Result)
#[pyfunction]
fn add_simd(a: &PyTensor, b: &PyTensor) -> PyResult<PyTensor> {
    Ok(PyTensor {
        inner: rusttorch_core::ops::add_simd(&a.inner, &b.inner).map_err(tensor_err)?,
    })
}

#[pyfunction]
fn mul_simd(a: &PyTensor, b: &PyTensor) -> PyResult<PyTensor> {
    Ok(PyTensor {
        inner: rusttorch_core::ops::mul_simd(&a.inner, &b.inner).map_err(tensor_err)?,
    })
}

#[pyfunction]
fn relu_simd(tensor: &PyTensor) -> PyTensor {
    PyTensor {
        inner: rusttorch_core::ops::relu_simd(&tensor.inner),
    }
}

#[pyfunction]
fn mul_scalar_simd(tensor: &PyTensor, scalar: f32) -> PyTensor {
    PyTensor {
        inner: rusttorch_core::ops::mul_scalar_simd(&tensor.inner, scalar),
    }
}

#[pyfunction]
fn fused_multiply_add(a: &PyTensor, b: &PyTensor, c: &PyTensor) -> PyResult<PyTensor> {
    Ok(PyTensor {
        inner: rusttorch_core::ops::fused_multiply_add(&a.inner, &b.inner, &c.inner)
            .map_err(tensor_err)?,
    })
}

// Data loading and preprocessing
#[pyfunction]
fn load_csv(path: String, has_header: bool, delimiter: char) -> PyResult<PyTensor> {
    Ok(PyTensor {
        inner: rusttorch_core::data::load_csv(path, has_header, delimiter).map_err(tensor_err)?,
    })
}

#[pyfunction]
fn normalize(tensor: &PyTensor) -> (PyTensor, f64, f64) {
    let (normalized, mean, std) = rusttorch_core::data::normalize(&tensor.inner);
    (PyTensor { inner: normalized }, mean, std)
}

#[pyfunction]
fn create_batches(data: &PyTensor, batch_size: usize, drop_last: bool) -> Vec<PyTensor> {
    rusttorch_core::data::create_batches(&data.inner, batch_size, drop_last)
        .into_iter()
        .map(|tensor| PyTensor { inner: tensor })
        .collect()
}

#[pyfunction]
fn shuffle_indices(num_samples: usize) -> Vec<usize> {
    rusttorch_core::data::shuffle_indices(num_samples)
}

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

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__author__", "Theodore Tennant (@teddytennant)")?;

    Ok(())
}
