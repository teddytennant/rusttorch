//! Dropout — randomly zeroes elements during training for regularization.
//!
//! During training: each element is zeroed with probability `p`, surviving elements
//! are scaled by 1/(1-p) to maintain expected values (inverted dropout).
//! During eval: pass-through (identity).

use crate::autograd::Variable;
use crate::error::Result;
use crate::nn::module::Module;
use crate::nn::parameter::Parameter;
use std::cell::Cell;

/// Dropout layer — randomly zeros elements during training.
///
/// Uses inverted dropout: surviving elements are scaled by 1/(1-p) during training,
/// so no scaling is needed at eval time.
pub struct Dropout {
    pub p: f32,
    training: Cell<bool>,
}

impl Dropout {
    /// Create a new Dropout layer with given drop probability.
    ///
    /// `p` is the probability of an element being zeroed. Default in PyTorch is 0.5.
    pub fn new(p: f32) -> Self {
        assert!(
            (0.0..1.0).contains(&p),
            "Dropout probability must be in [0, 1), got {}",
            p
        );
        Dropout {
            p,
            training: Cell::new(true),
        }
    }

    /// Set training mode.
    pub fn train(&self) {
        self.training.set(true);
    }

    /// Set evaluation mode.
    pub fn eval(&self) {
        self.training.set(false);
    }

    /// Whether in training mode.
    pub fn is_training(&self) -> bool {
        self.training.get()
    }
}

impl Module for Dropout {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        if !self.training.get() || self.p == 0.0 {
            return Ok(input.clone());
        }
        crate::autograd::ops::dropout_forward(input, self.p)
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![] // Dropout has no learnable parameters
    }
}

impl std::fmt::Debug for Dropout {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Dropout(p={}, training={})", self.p, self.training.get())
    }
}
