//! Loss function modules.
//!
//! These compute scalar loss values from predictions and targets,
//! returning Variables that can be backpropagated through.

use crate::autograd::Variable;
use crate::error::Result;

/// Mean Squared Error loss: L = mean((pred - target)^2)
#[derive(Debug)]
pub struct MSELoss;

impl MSELoss {
    pub fn new() -> Self {
        MSELoss
    }

    /// Compute MSE loss between predictions and targets.
    ///
    /// Both must have the same shape. Returns a scalar Variable.
    pub fn forward(&self, prediction: &Variable, target: &Variable) -> Result<Variable> {
        let diff = prediction.sub(target)?;
        let sq = diff.mul(&diff)?;
        sq.mean()
    }
}

impl Default for MSELoss {
    fn default() -> Self {
        Self::new()
    }
}
