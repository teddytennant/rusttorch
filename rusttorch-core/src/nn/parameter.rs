//! Parameter — a Variable that is a learnable model weight.

use crate::autograd::Variable;
use crate::tensor::{DType, Tensor};
use rand::thread_rng;
use rand::Rng;

/// A learnable parameter. Wraps a Variable with `requires_grad = true`.
///
/// Parameters are the trainable weights and biases of a neural network.
/// They are always leaf Variables that track gradients.
#[derive(Clone)]
pub struct Parameter {
    pub(crate) var: Variable,
    pub(crate) name: String,
}

impl Parameter {
    /// Create a parameter from a tensor. Always tracks gradients.
    pub fn new(tensor: Tensor, name: impl Into<String>) -> Self {
        Parameter {
            var: Variable::new(tensor, true),
            name: name.into(),
        }
    }

    /// Get the underlying Variable.
    pub fn var(&self) -> &Variable {
        &self.var
    }

    /// Get the name of this parameter.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the shape.
    pub fn shape(&self) -> Vec<usize> {
        self.var.shape()
    }

    /// Get the current tensor data.
    pub fn tensor(&self) -> Tensor {
        self.var.tensor()
    }

    /// Get the gradient, if computed.
    pub fn grad(&self) -> Option<Tensor> {
        self.var.grad()
    }

    /// Zero the gradient.
    pub fn zero_grad(&self) {
        self.var.zero_grad();
    }

    /// Create a parameter initialized with Kaiming uniform (He) initialization.
    ///
    /// Suitable for layers followed by ReLU.
    /// Draws from U(-bound, bound) where bound = sqrt(6 / fan_in).
    pub fn kaiming_uniform(shape: &[usize], fan_in: usize, name: impl Into<String>) -> Self {
        let bound = (6.0_f32 / fan_in as f32).sqrt();
        let mut rng = thread_rng();
        let data: Vec<f32> = (0..shape.iter().product::<usize>())
            .map(|_| rng.gen_range(-bound..bound))
            .collect();
        Parameter::new(Tensor::from_vec(data, shape), name)
    }

    /// Create a parameter initialized with uniform distribution U(-bound, bound).
    ///
    /// Used for bias initialization: bound = 1 / sqrt(fan_in).
    pub fn uniform(shape: &[usize], bound: f32, name: impl Into<String>) -> Self {
        let mut rng = thread_rng();
        let data: Vec<f32> = (0..shape.iter().product::<usize>())
            .map(|_| rng.gen_range(-bound..bound))
            .collect();
        Parameter::new(Tensor::from_vec(data, shape), name)
    }

    /// Create a zero-initialized parameter.
    pub fn zeros(shape: &[usize], name: impl Into<String>) -> Self {
        Parameter::new(Tensor::zeros(shape, DType::Float32), name)
    }

    /// Update this parameter's tensor in-place (for optimizer step).
    pub fn update(&self, new_tensor: Tensor) {
        let mut inner = self.var.inner.borrow_mut();
        inner.tensor = new_tensor;
        inner.grad = None;
    }

    /// Set the gradient tensor (used by gradient clipping).
    pub fn set_grad(&self, grad: Tensor) {
        let mut inner = self.var.inner.borrow_mut();
        inner.grad = Some(grad);
    }
}

impl std::fmt::Debug for Parameter {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Parameter(\"{}\", shape={:?})", self.name, self.shape())
    }
}
