//! BatchNorm2d — batch normalization for 2D inputs (4D tensors: [B, C, H, W]).
//!
//! During training: normalizes using batch statistics and updates running stats.
//! During eval: normalizes using running statistics (accumulated during training).

use crate::autograd::Variable;
use crate::error::Result;
use crate::nn::module::Module;
use crate::nn::parameter::Parameter;
use crate::nn::state_dict::StateDict;
use crate::tensor::Tensor;
use std::cell::Cell;

/// Batch normalization over 4D input [B, C, H, W].
///
/// Normalizes each channel across the batch and spatial dimensions,
/// then applies a learnable affine transform: y = gamma * x_norm + beta.
///
/// Tracks running mean and variance for inference via exponential moving average.
pub struct BatchNorm2d {
    pub num_features: usize,
    pub eps: f32,
    pub momentum: f32,
    pub weight: Parameter, // gamma
    pub bias: Parameter,   // beta
    running_mean: std::cell::RefCell<Vec<f32>>,
    running_var: std::cell::RefCell<Vec<f32>>,
    training: Cell<bool>,
}

impl BatchNorm2d {
    /// Create a new BatchNorm2d layer.
    ///
    /// - `num_features`: number of channels (C dimension)
    /// - Weight (gamma) initialized to 1.0
    /// - Bias (beta) initialized to 0.0
    /// - Running mean initialized to 0.0
    /// - Running variance initialized to 1.0
    pub fn new(num_features: usize) -> Self {
        let weight_data = vec![1.0f32; num_features];
        let bias_data = vec![0.0f32; num_features];

        BatchNorm2d {
            num_features,
            eps: 1e-5,
            momentum: 0.1,
            weight: Parameter::new(Tensor::from_vec(weight_data, &[num_features]), "weight"),
            bias: Parameter::new(Tensor::from_vec(bias_data, &[num_features]), "bias"),
            running_mean: std::cell::RefCell::new(vec![0.0f32; num_features]),
            running_var: std::cell::RefCell::new(vec![1.0f32; num_features]),
            training: Cell::new(true),
        }
    }

    /// Create with custom eps and momentum.
    pub fn with_options(num_features: usize, eps: f32, momentum: f32) -> Self {
        let mut bn = Self::new(num_features);
        bn.eps = eps;
        bn.momentum = momentum;
        bn
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

    /// Get a copy of the running mean.
    pub fn running_mean(&self) -> Vec<f32> {
        self.running_mean.borrow().clone()
    }

    /// Get a copy of the running variance.
    pub fn running_var(&self) -> Vec<f32> {
        self.running_var.borrow().clone()
    }
}

impl Module for BatchNorm2d {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        let is_training = self.training.get();
        crate::autograd::ops::batchnorm2d_forward(
            input,
            self.weight.var(),
            self.bias.var(),
            &self.running_mean,
            &self.running_var,
            is_training,
            self.momentum,
            self.eps,
        )
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    fn state_dict(&self) -> StateDict {
        let mut sd = StateDict::new();
        sd.insert("weight", self.weight.tensor());
        sd.insert("bias", self.bias.tensor());
        // Include non-learnable running stats as buffers
        let rm = self.running_mean.borrow();
        sd.insert(
            "running_mean",
            Tensor::from_vec(rm.clone(), &[self.num_features]),
        );
        let rv = self.running_var.borrow();
        sd.insert(
            "running_var",
            Tensor::from_vec(rv.clone(), &[self.num_features]),
        );
        sd
    }

    fn load_state_dict(&self, state_dict: &StateDict) {
        if let Some(t) = state_dict.get("weight") {
            self.weight.update(t.clone());
        }
        if let Some(t) = state_dict.get("bias") {
            self.bias.update(t.clone());
        }
        if let Some(t) = state_dict.get("running_mean") {
            *self.running_mean.borrow_mut() = t.to_vec_f32();
        }
        if let Some(t) = state_dict.get("running_var") {
            *self.running_var.borrow_mut() = t.to_vec_f32();
        }
    }
}

impl std::fmt::Debug for BatchNorm2d {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "BatchNorm2d(num_features={}, eps={}, momentum={}, training={})",
            self.num_features,
            self.eps,
            self.momentum,
            self.training.get()
        )
    }
}
