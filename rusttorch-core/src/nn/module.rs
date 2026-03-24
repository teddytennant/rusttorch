//! Module trait — the base abstraction for all neural network layers.

use crate::autograd::Variable;
use crate::error::Result;
use crate::nn::parameter::Parameter;

/// The base trait for all neural network modules (layers, activations, losses, containers).
///
/// Every module must implement `forward()` and `parameters()`.
/// This mirrors PyTorch's `nn.Module`.
pub trait Module {
    /// Forward pass: compute output from input.
    fn forward(&self, input: &Variable) -> Result<Variable>;

    /// Return all learnable parameters in this module.
    fn parameters(&self) -> Vec<Parameter>;

    /// Zero all parameter gradients.
    fn zero_grad(&self) {
        for param in self.parameters() {
            param.zero_grad();
        }
    }

    /// Count total number of trainable parameters.
    fn num_parameters(&self) -> usize {
        self.parameters()
            .iter()
            .map(|p| p.shape().iter().product::<usize>())
            .sum()
    }
}
