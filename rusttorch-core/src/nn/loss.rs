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

/// Cross-entropy loss for multi-class classification.
///
/// Combines log-softmax and negative log likelihood in a numerically stable way.
///
/// Input: raw logits [B, C] (NOT softmax probabilities)
/// Target: one-hot encoded labels [B, C]
/// Output: scalar loss
///
/// Formula: L = -mean(sum_c(target_c * log_softmax(input_c)))
#[derive(Debug)]
pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    pub fn new() -> Self {
        CrossEntropyLoss
    }

    /// Compute cross-entropy loss.
    ///
    /// - `logits`: raw scores [B, C], NOT passed through softmax
    /// - `target`: one-hot encoded labels [B, C]
    pub fn forward(&self, logits: &Variable, target: &Variable) -> Result<Variable> {
        crate::autograd::ops::cross_entropy_loss_forward(logits, target)
    }
}

impl Default for CrossEntropyLoss {
    fn default() -> Self {
        Self::new()
    }
}
