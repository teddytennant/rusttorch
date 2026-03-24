//! Layer Normalization — normalizes over the last dimension(s).
//!
//! Used in Transformers (pre-norm or post-norm) to stabilize training.
//! Unlike BatchNorm, LayerNorm normalizes per-instance (no batch statistics).

use crate::autograd::Variable;
use crate::error::Result;
use crate::nn::module::Module;
use crate::nn::parameter::Parameter;
use crate::tensor::{DType, Tensor};

/// Layer normalization module.
///
/// Normalizes over the last `normalized_shape` elements:
/// `y = (x - mean) / sqrt(var + eps) * weight + bias`
///
/// Weight (gamma) initialized to 1, bias (beta) initialized to 0.
pub struct LayerNorm {
    pub weight: Parameter,
    pub bias: Parameter,
    pub normalized_shape: usize,
    pub eps: f32,
}

impl LayerNorm {
    /// Create a new LayerNorm that normalizes over the last `normalized_shape` elements.
    pub fn new(normalized_shape: usize) -> Self {
        let weight = Parameter::new(Tensor::ones(&[normalized_shape], DType::Float32), "weight");
        let bias = Parameter::zeros(&[normalized_shape], "bias");

        LayerNorm {
            weight,
            bias,
            normalized_shape,
            eps: 1e-5,
        }
    }

    /// Create with custom epsilon.
    pub fn with_eps(normalized_shape: usize, eps: f32) -> Self {
        let mut ln = Self::new(normalized_shape);
        ln.eps = eps;
        ln
    }
}

impl Module for LayerNorm {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        input.layer_norm(
            self.normalized_shape,
            Some(self.weight.var()),
            Some(self.bias.var()),
            self.eps,
        )
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}

impl std::fmt::Debug for LayerNorm {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "LayerNorm(normalized_shape={}, eps={})",
            self.normalized_shape, self.eps
        )
    }
}
