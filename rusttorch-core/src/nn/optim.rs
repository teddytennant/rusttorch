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

/// Stochastic Gradient Descent optimizer.
pub struct SGD {
    params: Vec<Parameter>,
    lr: f32,
    momentum: f32,
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
            velocities: vec![None; n],
        }
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
                // SGD with momentum: v = momentum * v + grad; param -= lr * v
                let vel = self.velocities[i]
                    .as_ref()
                    .map(|v| v.to_vec_f32())
                    .unwrap_or_else(|| vec![0.0; tensor_data.len()]);

                let new_vel: Vec<f32> = vel
                    .iter()
                    .zip(grad_data.iter())
                    .map(|(&v, &g)| self.momentum * v + g)
                    .collect();

                let updated: Vec<f32> = tensor_data
                    .iter()
                    .zip(new_vel.iter())
                    .map(|(&p, &v)| p - self.lr * v)
                    .collect();

                self.velocities[i] = Some(Tensor::from_vec(new_vel, tensor.shape()));

                updated
            } else {
                // Plain SGD: param -= lr * grad
                tensor_data
                    .iter()
                    .zip(grad_data.iter())
                    .map(|(&p, &g)| p - self.lr * g)
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
