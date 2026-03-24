//! Variable — a tensor that tracks its computation history for automatic differentiation.

use crate::autograd::ops::GradFn;
use crate::error::{Result, TensorError};
use crate::tensor::Tensor;
use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

fn next_id() -> usize {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

/// Internal state of a Variable node in the computation graph.
pub(crate) struct VariableInner {
    pub(crate) id: usize,
    pub(crate) tensor: Tensor,
    pub(crate) grad: Option<Tensor>,
    pub(crate) grad_fn: Option<Box<dyn GradFn>>,
    pub(crate) requires_grad: bool,
    /// True if this variable was created directly (not as a result of an operation)
    pub(crate) is_leaf: bool,
}

/// A tensor that participates in automatic differentiation.
///
/// `Variable` wraps a `Tensor` and optionally tracks the computation graph.
/// When `requires_grad` is true, operations on this variable are recorded
/// so that gradients can be computed via `backward()`.
///
/// Variables are cheap to clone — cloning creates a new reference to the same
/// underlying node in the computation graph.
#[derive(Clone)]
pub struct Variable {
    pub(crate) inner: Rc<RefCell<VariableInner>>,
}

impl Variable {
    /// Create a new leaf variable.
    ///
    /// # Arguments
    /// * `tensor` - The tensor data
    /// * `requires_grad` - Whether to track gradients for this variable
    pub fn new(tensor: Tensor, requires_grad: bool) -> Self {
        Variable {
            inner: Rc::new(RefCell::new(VariableInner {
                id: next_id(),
                tensor,
                grad: None,
                grad_fn: None,
                requires_grad,
                is_leaf: true,
            })),
        }
    }

    /// Create a variable that is the result of an operation (non-leaf).
    pub(crate) fn from_op(tensor: Tensor, grad_fn: Box<dyn GradFn>) -> Self {
        Variable {
            inner: Rc::new(RefCell::new(VariableInner {
                id: next_id(),
                tensor,
                grad: None,
                grad_fn: Some(grad_fn),
                requires_grad: true,
                is_leaf: false,
            })),
        }
    }

    /// Create a variable with no gradient tracking.
    pub fn detach(tensor: Tensor) -> Self {
        Variable::new(tensor, false)
    }

    /// Get the unique ID of this variable.
    pub fn id(&self) -> usize {
        self.inner.borrow().id
    }

    /// Get a clone of the underlying tensor.
    pub fn tensor(&self) -> Tensor {
        self.inner.borrow().tensor.clone()
    }

    /// Get the shape of the underlying tensor.
    pub fn shape(&self) -> Vec<usize> {
        self.inner.borrow().tensor.shape().to_vec()
    }

    /// Get the number of elements.
    pub fn numel(&self) -> usize {
        self.inner.borrow().tensor.numel()
    }

    /// Whether this variable requires gradient computation.
    pub fn requires_grad(&self) -> bool {
        self.inner.borrow().requires_grad
    }

    /// Whether this is a leaf variable (created directly, not from an operation).
    pub fn is_leaf(&self) -> bool {
        self.inner.borrow().is_leaf
    }

    /// Get the accumulated gradient, if any.
    pub fn grad(&self) -> Option<Tensor> {
        self.inner.borrow().grad.clone()
    }

    /// Zero out the gradient.
    pub fn zero_grad(&self) {
        self.inner.borrow_mut().grad = None;
    }

    /// Accumulate a gradient into this variable.
    pub(crate) fn accumulate_grad(&self, grad: &Tensor) -> Result<()> {
        let mut inner = self.inner.borrow_mut();
        match &inner.grad {
            Some(existing) => {
                let accumulated = crate::ops::add(existing, grad)?;
                inner.grad = Some(accumulated);
            }
            None => {
                inner.grad = Some(grad.clone());
            }
        }
        Ok(())
    }

    /// Compute gradients via backpropagation.
    ///
    /// This can only be called on a scalar (single-element) variable.
    /// Gradients are accumulated on all leaf variables that have `requires_grad = true`.
    pub fn backward(&self) -> Result<()> {
        let inner = self.inner.borrow();
        if inner.tensor.numel() != 1 {
            return Err(TensorError::InvalidArgument {
                parameter: "variable".to_string(),
                reason: format!(
                    "backward() can only be called on scalar outputs, got {} elements",
                    inner.tensor.numel()
                ),
            });
        }
        drop(inner);

        crate::autograd::graph::backward(self)
    }

    // ---- Differentiable operations ----
    // Each operation returns a new Variable with a GradFn that records the backward pass.

    /// Element-wise addition.
    pub fn add(&self, other: &Variable) -> Result<Variable> {
        crate::autograd::ops::add_forward(self, other)
    }

    /// Element-wise subtraction.
    pub fn sub(&self, other: &Variable) -> Result<Variable> {
        crate::autograd::ops::sub_forward(self, other)
    }

    /// Element-wise multiplication.
    pub fn mul(&self, other: &Variable) -> Result<Variable> {
        crate::autograd::ops::mul_forward(self, other)
    }

    /// Element-wise division.
    pub fn div(&self, other: &Variable) -> Result<Variable> {
        crate::autograd::ops::div_forward(self, other)
    }

    /// Matrix multiplication.
    pub fn matmul(&self, other: &Variable) -> Result<Variable> {
        crate::autograd::ops::matmul_forward(self, other)
    }

    /// ReLU activation.
    pub fn relu(&self) -> Variable {
        crate::autograd::ops::relu_forward(self)
    }

    /// Sigmoid activation.
    pub fn sigmoid(&self) -> Result<Variable> {
        crate::autograd::ops::sigmoid_forward(self)
    }

    /// Tanh activation.
    pub fn tanh_act(&self) -> Result<Variable> {
        crate::autograd::ops::tanh_forward(self)
    }

    /// Sum all elements to a scalar.
    pub fn sum(&self) -> Result<Variable> {
        crate::autograd::ops::sum_forward(self)
    }

    /// Mean of all elements to a scalar.
    pub fn mean(&self) -> Result<Variable> {
        crate::autograd::ops::mean_forward(self)
    }

    /// Scalar multiplication.
    pub fn mul_scalar(&self, scalar: f32) -> Variable {
        crate::autograd::ops::mul_scalar_forward(self, scalar)
    }

    /// Broadcasting addition (supports different but broadcastable shapes).
    pub fn broadcast_add(&self, other: &Variable) -> Result<Variable> {
        crate::autograd::ops::broadcast_add_forward(self, other)
    }

    /// Transpose a 2D variable (preserves computation graph).
    pub fn t(&self) -> Result<Variable> {
        crate::autograd::ops::transpose_forward(self)
    }

    /// Reshape the variable (preserves computation graph).
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Variable> {
        crate::autograd::ops::reshape_forward(self, new_shape)
    }

    /// Log-softmax along a dimension (numerically stable).
    pub fn log_softmax(&self, dim: usize) -> Result<Variable> {
        crate::autograd::ops::log_softmax_forward(self, dim)
    }

    /// GELU activation (tanh approximation, differentiable).
    pub fn gelu(&self) -> Variable {
        crate::autograd::ops::gelu_forward(self)
    }

    /// Layer normalization over the last `norm_size` elements.
    pub fn layer_norm(
        &self,
        norm_size: usize,
        weight: Option<&Variable>,
        bias: Option<&Variable>,
        eps: f32,
    ) -> Result<Variable> {
        crate::autograd::ops::layer_norm_forward(self, norm_size, weight, bias, eps)
    }
}

impl fmt::Debug for Variable {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let inner = self.inner.borrow();
        write!(
            f,
            "Variable(id={}, shape={:?}, requires_grad={}, is_leaf={}, has_grad={})",
            inner.id,
            inner.tensor.shape(),
            inner.requires_grad,
            inner.is_leaf,
            inner.grad.is_some()
        )
    }
}

impl fmt::Display for Variable {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let inner = self.inner.borrow();
        write!(
            f,
            "Variable(shape={:?}, requires_grad={})",
            inner.tensor.shape(),
            inner.requires_grad
        )
    }
}
