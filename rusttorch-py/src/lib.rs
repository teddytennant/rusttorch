//! RustTorch Python Bindings
//!
//! PyO3-based Python bindings for the RustTorch library.
//! Provides a PyTorch-like API for high-performance tensor operations.

// The `#[pymethods]` attribute macro from pyo3 0.20 emits `impl` blocks
// inside generated functions, which rustc ≥1.80 flags via the
// `non_local_definitions` lint. The fix is a pyo3 upgrade; until then,
// silence the lint at the crate level so builds stay warning-clean.
#![allow(non_local_definitions)]

use numpy::{PyReadonlyArrayDyn, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::PyList;
use rusttorch_core::autograd::Variable as RustVariable;
use rusttorch_core::nn::{
    Adam as RustAdam, Linear as RustLinear, MSELoss as RustMSELoss, Module as _,
    Optimizer as _, Parameter as RustParameter, SGD as RustSGD,
};
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

// ===========================================================================
// Autograd / nn bindings
// ===========================================================================
//
// Variable uses Rc<RefCell<..>> internally, so every pyclass that wraps one
// must be marked `unsendable`. This forbids the type from crossing Python
// threads but avoids introducing locks on the fast path. Modules built on
// Variable (Linear, MSELoss, SGD, Adam) inherit the same restriction.

/// Python wrapper for autograd::Variable.
#[pyclass(name = "Variable", unsendable)]
#[derive(Clone)]
struct PyVariable {
    inner: RustVariable,
}

#[pymethods]
impl PyVariable {
    /// Construct a Variable from a 1-D Python list of f32s with an explicit
    /// shape. `requires_grad=True` enables gradient tracking.
    #[new]
    #[pyo3(signature = (data, shape, requires_grad = false))]
    fn new(data: Vec<f32>, shape: Vec<usize>, requires_grad: bool) -> Self {
        PyVariable {
            inner: RustVariable::new(RustTensor::from_vec(data, &shape), requires_grad),
        }
    }

    /// Construct a Variable from a numpy array (f32, any rank).
    #[staticmethod]
    #[pyo3(signature = (array, requires_grad = false))]
    fn from_numpy(array: PyReadonlyArrayDyn<f32>, requires_grad: bool) -> PyResult<Self> {
        let shape: Vec<usize> = array.shape().to_vec();
        let data = array
            .as_slice()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be contiguous"))?
            .to_vec();
        Ok(PyVariable {
            inner: RustVariable::new(RustTensor::from_vec(data, &shape), requires_grad),
        })
    }

    fn shape(&self) -> Vec<usize> {
        self.inner.shape()
    }

    fn numel(&self) -> usize {
        self.inner.numel()
    }

    fn requires_grad(&self) -> bool {
        self.inner.requires_grad()
    }

    fn is_leaf(&self) -> bool {
        self.inner.is_leaf()
    }

    /// Return the tensor data as a flat Python list of f32s. For multi-dim
    /// reads, inspect `.shape()` and reshape on the Python side (or use
    /// `.to_numpy()`).
    fn to_list(&self) -> Vec<f32> {
        self.inner.tensor().to_vec_f32()
    }

    /// Return the tensor data as a numpy array reshaped to `.shape()`.
    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<&'py numpy::PyArrayDyn<f32>> {
        let tensor = self.inner.tensor();
        let shape = tensor.shape().to_vec();
        let data = tensor.to_vec_f32();
        let arr = numpy::ndarray::Array::from_shape_vec(shape, data)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(arr.to_pyarray(py))
    }

    /// Gradient tensor as a flat list, or None if backward hasn't produced one.
    fn grad(&self) -> Option<Vec<f32>> {
        self.inner.grad().map(|t| t.to_vec_f32())
    }

    /// Gradient tensor as a numpy array, or None.
    fn grad_numpy<'py>(&self, py: Python<'py>) -> PyResult<Option<&'py numpy::PyArrayDyn<f32>>> {
        match self.inner.grad() {
            Some(t) => {
                let shape = t.shape().to_vec();
                let data = t.to_vec_f32();
                let arr = numpy::ndarray::Array::from_shape_vec(shape, data)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                Ok(Some(arr.to_pyarray(py)))
            }
            None => Ok(None),
        }
    }

    fn zero_grad(&self) {
        self.inner.zero_grad();
    }

    /// Run reverse-mode autodiff. Only callable on a scalar output.
    fn backward(&self) -> PyResult<()> {
        self.inner.backward().map_err(tensor_err)
    }

    // ---- Differentiable ops (subset matching autograd::Variable) ----

    fn add(&self, other: &PyVariable) -> PyResult<PyVariable> {
        Ok(PyVariable {
            inner: self.inner.add(&other.inner).map_err(tensor_err)?,
        })
    }

    fn sub(&self, other: &PyVariable) -> PyResult<PyVariable> {
        Ok(PyVariable {
            inner: self.inner.sub(&other.inner).map_err(tensor_err)?,
        })
    }

    fn mul(&self, other: &PyVariable) -> PyResult<PyVariable> {
        Ok(PyVariable {
            inner: self.inner.mul(&other.inner).map_err(tensor_err)?,
        })
    }

    fn div(&self, other: &PyVariable) -> PyResult<PyVariable> {
        Ok(PyVariable {
            inner: self.inner.div(&other.inner).map_err(tensor_err)?,
        })
    }

    fn matmul(&self, other: &PyVariable) -> PyResult<PyVariable> {
        Ok(PyVariable {
            inner: self.inner.matmul(&other.inner).map_err(tensor_err)?,
        })
    }

    fn relu(&self) -> PyVariable {
        PyVariable {
            inner: self.inner.relu(),
        }
    }

    fn sigmoid(&self) -> PyResult<PyVariable> {
        Ok(PyVariable {
            inner: self.inner.sigmoid().map_err(tensor_err)?,
        })
    }

    fn tanh(&self) -> PyResult<PyVariable> {
        Ok(PyVariable {
            inner: self.inner.tanh_act().map_err(tensor_err)?,
        })
    }

    fn gelu(&self) -> PyVariable {
        PyVariable {
            inner: self.inner.gelu(),
        }
    }

    fn sum(&self) -> PyResult<PyVariable> {
        Ok(PyVariable {
            inner: self.inner.sum().map_err(tensor_err)?,
        })
    }

    fn mean(&self) -> PyResult<PyVariable> {
        Ok(PyVariable {
            inner: self.inner.mean().map_err(tensor_err)?,
        })
    }

    fn mul_scalar(&self, scalar: f32) -> PyVariable {
        PyVariable {
            inner: self.inner.mul_scalar(scalar),
        }
    }

    fn t(&self) -> PyResult<PyVariable> {
        Ok(PyVariable {
            inner: self.inner.t().map_err(tensor_err)?,
        })
    }

    fn reshape(&self, new_shape: Vec<usize>) -> PyResult<PyVariable> {
        Ok(PyVariable {
            inner: self.inner.reshape(&new_shape).map_err(tensor_err)?,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "Variable(shape={:?}, requires_grad={}, is_leaf={})",
            self.inner.shape(),
            self.inner.requires_grad(),
            self.inner.is_leaf()
        )
    }
}

/// Python wrapper for an nn::Parameter. Parameters are what optimizers
/// consume; they are returned by `Linear.parameters()`.
#[pyclass(name = "Parameter", unsendable)]
#[derive(Clone)]
struct PyParameter {
    inner: RustParameter,
}

#[pymethods]
impl PyParameter {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn shape(&self) -> Vec<usize> {
        self.inner.shape()
    }

    fn to_list(&self) -> Vec<f32> {
        self.inner.tensor().to_vec_f32()
    }

    fn grad(&self) -> Option<Vec<f32>> {
        self.inner.grad().map(|t| t.to_vec_f32())
    }

    fn zero_grad(&self) {
        self.inner.zero_grad();
    }

    fn __repr__(&self) -> String {
        format!("Parameter('{}', shape={:?})", self.inner.name(), self.inner.shape())
    }
}

/// Python wrapper for nn::Linear.
#[pyclass(name = "Linear", unsendable)]
struct PyLinear {
    inner: RustLinear,
}

#[pymethods]
impl PyLinear {
    #[new]
    #[pyo3(signature = (in_features, out_features, bias = true))]
    fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        let inner = if bias {
            RustLinear::new(in_features, out_features)
        } else {
            RustLinear::no_bias(in_features, out_features)
        };
        PyLinear { inner }
    }

    fn forward(&self, input: &PyVariable) -> PyResult<PyVariable> {
        Ok(PyVariable {
            inner: self.inner.forward(&input.inner).map_err(tensor_err)?,
        })
    }

    fn parameters(&self) -> Vec<PyParameter> {
        self.inner
            .parameters()
            .into_iter()
            .map(|p| PyParameter { inner: p })
            .collect()
    }

    fn in_features(&self) -> usize {
        self.inner.in_features
    }

    fn out_features(&self) -> usize {
        self.inner.out_features
    }

    fn __call__(&self, input: &PyVariable) -> PyResult<PyVariable> {
        self.forward(input)
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

/// Python wrapper for nn::MSELoss.
#[pyclass(name = "MSELoss", unsendable)]
struct PyMSELoss {
    inner: RustMSELoss,
}

#[pymethods]
impl PyMSELoss {
    #[new]
    fn new() -> Self {
        PyMSELoss {
            inner: RustMSELoss::new(),
        }
    }

    fn forward(&self, pred: &PyVariable, target: &PyVariable) -> PyResult<PyVariable> {
        Ok(PyVariable {
            inner: self
                .inner
                .forward(&pred.inner, &target.inner)
                .map_err(tensor_err)?,
        })
    }

    fn __call__(&self, pred: &PyVariable, target: &PyVariable) -> PyResult<PyVariable> {
        self.forward(pred, target)
    }
}

/// Python wrapper for nn::SGD.
#[pyclass(name = "SGD", unsendable)]
struct PySGD {
    inner: RustSGD,
}

#[pymethods]
impl PySGD {
    #[new]
    fn new(params: Vec<PyParameter>, lr: f32) -> Self {
        let rust_params: Vec<RustParameter> = params.into_iter().map(|p| p.inner).collect();
        PySGD {
            inner: RustSGD::new(rust_params, lr),
        }
    }

    fn step(&mut self) -> PyResult<()> {
        self.inner.step().map_err(tensor_err)
    }

    fn zero_grad(&self) {
        self.inner.zero_grad();
    }
}

/// Python wrapper for nn::Adam.
#[pyclass(name = "Adam", unsendable)]
struct PyAdam {
    inner: RustAdam,
}

#[pymethods]
impl PyAdam {
    #[new]
    fn new(params: Vec<PyParameter>, lr: f32) -> Self {
        let rust_params: Vec<RustParameter> = params.into_iter().map(|p| p.inner).collect();
        PyAdam {
            inner: RustAdam::new(rust_params, lr),
        }
    }

    fn step(&mut self) -> PyResult<()> {
        self.inner.step().map_err(tensor_err)
    }

    fn zero_grad(&self) {
        self.inner.zero_grad();
    }
}

#[pymodule]
fn rusttorch(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyTensor>()?;
    m.add_class::<PyVariable>()?;
    m.add_class::<PyParameter>()?;
    m.add_class::<PyLinear>()?;
    m.add_class::<PyMSELoss>()?;
    m.add_class::<PySGD>()?;
    m.add_class::<PyAdam>()?;

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
