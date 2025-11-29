//! RustTorch Python Bindings
//!
//! PyO3-based Python bindings for the RustTorch library.
//! Provides a PyTorch-like API for high-performance tensor operations.

use pyo3::prelude::*;
use pyo3::types::PyList;
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
fn sigmoid(tensor: &PyTensor) -> PyResult<PyTensor> {
    match rusttorch_core::ops::sigmoid(&tensor.inner) {
        Ok(result) => Ok(PyTensor { inner: result }),
        Err(e) => Err(pyo3::exceptions::PyValueError::new_err(e)),
    }
}

/// Tanh activation
#[pyfunction]
fn tanh(tensor: &PyTensor) -> PyResult<PyTensor> {
    match rusttorch_core::ops::tanh(&tensor.inner) {
        Ok(result) => Ok(PyTensor { inner: result }),
        Err(e) => Err(pyo3::exceptions::PyValueError::new_err(e)),
    }
}

/// Python module definition
#[pymodule]
fn rusttorch(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyTensor>()?;

    // Element-wise operations
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(mul, m)?)?;

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

    // Module metadata
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__author__", "Theodore Tennant (@teddytennant)")?;

    Ok(())
}
