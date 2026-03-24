//! Optimizers that work with nn::Parameter.
//!
//! These read gradients from Parameters and update their tensors in-place.

use crate::error::{Result, TensorError};
use crate::nn::parameter::Parameter;
use crate::tensor::Tensor;

/// Optimizer trait — step() reads grads and updates parameters.
pub trait Optimizer {
    /// Perform a single optimization step.
    fn step(&mut self) -> Result<()>;

    /// Zero all parameter gradients.
    fn zero_grad(&self);
}

// --- Gradient Clipping ---

/// Clip gradients by maximum norm (L2 norm).
///
/// Rescales all parameter gradients so their combined L2 norm does not exceed
/// `max_norm`. This is the standard gradient clipping method used in training
/// deep networks (RNNs, Transformers, deep ResNets).
///
/// Returns the total gradient norm before clipping.
///
/// # Arguments
/// * `params` — parameters whose gradients to clip
/// * `max_norm` — maximum allowed L2 norm
pub fn clip_grad_norm(params: &[Parameter], max_norm: f32) -> f32 {
    // Compute total L2 norm across all parameter gradients
    let mut total_norm_sq = 0.0f64;
    for param in params {
        if let Some(grad) = param.grad() {
            let g = grad.to_vec_f32();
            for &v in &g {
                total_norm_sq += (v as f64) * (v as f64);
            }
        }
    }
    let total_norm = total_norm_sq.sqrt() as f32;

    if total_norm > max_norm {
        let scale = max_norm / (total_norm + 1e-6);
        for param in params {
            if let Some(grad) = param.grad() {
                let clipped: Vec<f32> = grad.to_vec_f32().iter().map(|&v| v * scale).collect();
                param.set_grad(Tensor::from_vec(clipped, grad.shape()));
            }
        }
    }

    total_norm
}

/// Clip gradients by value.
///
/// Clamps each gradient element to `[-clip_value, clip_value]`.
///
/// # Arguments
/// * `params` — parameters whose gradients to clip
/// * `clip_value` — maximum absolute value for any gradient element
pub fn clip_grad_value(params: &[Parameter], clip_value: f32) {
    for param in params {
        if let Some(grad) = param.grad() {
            let clipped: Vec<f32> = grad
                .to_vec_f32()
                .iter()
                .map(|&v| v.clamp(-clip_value, clip_value))
                .collect();
            param.set_grad(Tensor::from_vec(clipped, grad.shape()));
        }
    }
}

/// Stochastic Gradient Descent optimizer.
///
/// Supports momentum and weight decay (L2 regularization).
/// With weight decay, the effective gradient becomes `grad + weight_decay * param`.
pub struct SGD {
    params: Vec<Parameter>,
    lr: f32,
    momentum: f32,
    weight_decay: f32,
    velocities: Vec<Option<Tensor>>,
}

impl SGD {
    /// Create a new SGD optimizer.
    ///
    /// # Arguments
    /// * `params` - Parameters to optimize
    /// * `lr` - Learning rate
    pub fn new(params: Vec<Parameter>, lr: f32) -> Self {
        let n = params.len();
        SGD {
            params,
            lr,
            momentum: 0.0,
            weight_decay: 0.0,
            velocities: vec![None; n],
        }
    }

    /// Create SGD with momentum.
    pub fn with_momentum(params: Vec<Parameter>, lr: f32, momentum: f32) -> Self {
        let n = params.len();
        SGD {
            params,
            lr,
            momentum,
            weight_decay: 0.0,
            velocities: vec![None; n],
        }
    }

    /// Create SGD with momentum and weight decay (L2 regularization).
    ///
    /// Standard for ResNet training: `momentum=0.9, weight_decay=5e-4`.
    pub fn with_momentum_and_weight_decay(
        params: Vec<Parameter>,
        lr: f32,
        momentum: f32,
        weight_decay: f32,
    ) -> Self {
        let n = params.len();
        SGD {
            params,
            lr,
            momentum,
            weight_decay,
            velocities: vec![None; n],
        }
    }

    /// Get the current learning rate.
    pub fn lr(&self) -> f32 {
        self.lr
    }

    /// Set the learning rate (used by LR schedulers).
    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

impl Optimizer for SGD {
    fn step(&mut self) -> Result<()> {
        for (i, param) in self.params.iter().enumerate() {
            let grad = match param.grad() {
                Some(g) => g,
                None => continue,
            };

            let tensor = param.tensor();
            let tensor_data = tensor.to_vec_f32();
            let grad_data = grad.to_vec_f32();

            if tensor_data.len() != grad_data.len() {
                return Err(TensorError::ShapeMismatch {
                    expected: tensor.shape().to_vec(),
                    actual: grad.shape().to_vec(),
                    context: "SGD step".to_string(),
                });
            }

            let new_data: Vec<f32> = if self.momentum > 0.0 {
                // SGD with momentum + weight decay:
                // effective_grad = grad + weight_decay * param
                // v = momentum * v + effective_grad
                // param -= lr * v
                let vel = self.velocities[i]
                    .as_ref()
                    .map(|v| v.to_vec_f32())
                    .unwrap_or_else(|| vec![0.0; tensor_data.len()]);

                let wd = self.weight_decay;
                let new_vel: Vec<f32> = vel
                    .iter()
                    .zip(grad_data.iter())
                    .zip(tensor_data.iter())
                    .map(|((&v, &g), &p)| self.momentum * v + g + wd * p)
                    .collect();

                let updated: Vec<f32> = tensor_data
                    .iter()
                    .zip(new_vel.iter())
                    .map(|(&p, &v)| p - self.lr * v)
                    .collect();

                self.velocities[i] = Some(Tensor::from_vec(new_vel, tensor.shape()));

                updated
            } else {
                // Plain SGD with optional weight decay:
                // param -= lr * (grad + weight_decay * param)
                let wd = self.weight_decay;
                tensor_data
                    .iter()
                    .zip(grad_data.iter())
                    .map(|(&p, &g)| p - self.lr * (g + wd * p))
                    .collect()
            };

            param.update(Tensor::from_vec(new_data, tensor.shape()));
        }
        Ok(())
    }

    fn zero_grad(&self) {
        for param in &self.params {
            param.zero_grad();
        }
    }
}

/// Adam optimizer (Adaptive Moment Estimation).
pub struct Adam {
    params: Vec<Parameter>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    t: usize,
    m: Vec<Option<Tensor>>,
    v: Vec<Option<Tensor>>,
}

impl Adam {
    /// Create a new Adam optimizer with default betas (0.9, 0.999) and eps (1e-8).
    pub fn new(params: Vec<Parameter>, lr: f32) -> Self {
        let n = params.len();
        Adam {
            params,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            t: 0,
            m: vec![None; n],
            v: vec![None; n],
        }
    }

    /// Create Adam with custom hyperparameters.
    pub fn with_params(params: Vec<Parameter>, lr: f32, beta1: f32, beta2: f32, eps: f32) -> Self {
        let n = params.len();
        Adam {
            params,
            lr,
            beta1,
            beta2,
            eps,
            t: 0,
            m: vec![None; n],
            v: vec![None; n],
        }
    }
}

impl Optimizer for Adam {
    fn step(&mut self) -> Result<()> {
        self.t += 1;
        let t = self.t as f32;

        let bias_correction1 = 1.0 - self.beta1.powf(t);
        let bias_correction2 = 1.0 - self.beta2.powf(t);

        for (i, param) in self.params.iter().enumerate() {
            let grad = match param.grad() {
                Some(g) => g,
                None => continue,
            };

            let tensor = param.tensor();
            let p = tensor.to_vec_f32();
            let g = grad.to_vec_f32();
            let n = p.len();

            // Get or initialize moment estimates
            let m_prev = self.m[i]
                .as_ref()
                .map(|t| t.to_vec_f32())
                .unwrap_or_else(|| vec![0.0; n]);
            let v_prev = self.v[i]
                .as_ref()
                .map(|t| t.to_vec_f32())
                .unwrap_or_else(|| vec![0.0; n]);

            let mut new_m = vec![0.0f32; n];
            let mut new_v = vec![0.0f32; n];
            let mut new_p = vec![0.0f32; n];

            for j in 0..n {
                // Update biased first moment estimate
                new_m[j] = self.beta1 * m_prev[j] + (1.0 - self.beta1) * g[j];
                // Update biased second raw moment estimate
                new_v[j] = self.beta2 * v_prev[j] + (1.0 - self.beta2) * g[j] * g[j];
                // Bias-corrected estimates
                let m_hat = new_m[j] / bias_correction1;
                let v_hat = new_v[j] / bias_correction2;
                // Update parameters
                new_p[j] = p[j] - self.lr * m_hat / (v_hat.sqrt() + self.eps);
            }

            let shape = tensor.shape();
            self.m[i] = Some(Tensor::from_vec(new_m, shape));
            self.v[i] = Some(Tensor::from_vec(new_v, shape));
            param.update(Tensor::from_vec(new_p, shape));
        }
        Ok(())
    }

    fn zero_grad(&self) {
        for param in &self.params {
            param.zero_grad();
        }
    }
}

// --- Learning Rate Schedulers ---

/// Step learning rate decay.
///
/// Multiplies the learning rate by `gamma` every `step_size` epochs.
pub struct StepLR {
    base_lr: f32,
    step_size: usize,
    gamma: f32,
}

impl StepLR {
    /// Create a step LR scheduler.
    pub fn new(base_lr: f32, step_size: usize, gamma: f32) -> Self {
        StepLR {
            base_lr,
            step_size,
            gamma,
        }
    }

    /// Compute learning rate for a given epoch.
    pub fn lr_at(&self, epoch: usize) -> f32 {
        let num_decays = epoch / self.step_size;
        self.base_lr * self.gamma.powi(num_decays as i32)
    }
}

/// Multi-step learning rate decay at specified milestones.
///
/// Multiplies LR by `gamma` each time a milestone epoch is reached.
/// Standard for CIFAR-10 ResNet: `milestones=[80, 120], gamma=0.1` with 160 epochs.
pub struct MultiStepLR {
    base_lr: f32,
    milestones: Vec<usize>,
    gamma: f32,
}

impl MultiStepLR {
    /// Create a multi-step LR scheduler.
    pub fn new(base_lr: f32, milestones: Vec<usize>, gamma: f32) -> Self {
        MultiStepLR {
            base_lr,
            milestones,
            gamma,
        }
    }

    /// Compute learning rate for a given epoch.
    pub fn lr_at(&self, epoch: usize) -> f32 {
        let num_decays = self.milestones.iter().filter(|&&m| epoch >= m).count();
        self.base_lr * self.gamma.powi(num_decays as i32)
    }
}

/// Cosine annealing learning rate schedule.
///
/// Decays LR from `base_lr` to `min_lr` following a cosine curve over `total_epochs`.
pub struct CosineAnnealingLR {
    base_lr: f32,
    min_lr: f32,
    total_epochs: usize,
}

impl CosineAnnealingLR {
    /// Create a cosine annealing scheduler.
    pub fn new(base_lr: f32, min_lr: f32, total_epochs: usize) -> Self {
        CosineAnnealingLR {
            base_lr,
            min_lr,
            total_epochs,
        }
    }

    /// Compute learning rate for a given epoch.
    pub fn lr_at(&self, epoch: usize) -> f32 {
        let t = (epoch as f32) / (self.total_epochs as f32);
        let t = t.min(1.0);
        self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + (t * std::f32::consts::PI).cos())
    }
}
