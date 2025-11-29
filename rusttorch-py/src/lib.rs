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

/// Python module definition
#[pymodule]
fn rusttorch(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyTensor>()?;
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(mul, m)?)?;
    m.add_function(wrap_pyfunction!(relu, m)?)?;

    // Module metadata
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__author__", "RustTorch Contributors")?;

    Ok(())
}
