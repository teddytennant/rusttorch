//! Flatten layer — reshapes [B, C, H, W] to [B, C*H*W].

use crate::autograd::Variable;
use crate::error::Result;
use crate::nn::module::Module;
use crate::nn::parameter::Parameter;

/// Flattens all dimensions after the batch dimension.
///
/// Input: [B, d1, d2, ...] -> Output: [B, d1*d2*...]
pub struct Flatten;

impl Flatten {
    pub fn new() -> Self {
        Flatten
    }
}

impl Default for Flatten {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Flatten {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        let shape = input.shape();
        if shape.len() <= 2 {
            // Already flat or 1D, return as-is
            return Ok(input.clone());
        }
        let batch = shape[0];
        let flat_dim: usize = shape[1..].iter().product();
        input.reshape(&[batch, flat_dim])
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![]
    }
}

impl std::fmt::Debug for Flatten {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Flatten()")
    }
}
