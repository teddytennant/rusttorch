//! Optimizer update rules
//!
//! This module implements parameter update rules for common optimizers:
//! - SGD (Stochastic Gradient Descent)
//! - SGD with Momentum
//! - Adam (Adaptive Moment Estimation)
//! - AdamW (Adam with decoupled weight decay)
//!
//! All functions return `Result` types for proper error handling.

use crate::error::{Result, TensorError};
use crate::tensor::{Tensor, TensorData, DType};

/// SGD (Stochastic Gradient Descent) parameter update
///
/// Updates parameters using: params = params - learning_rate * gradients
///
/// # Arguments
/// * `params` - Current parameter values
/// * `gradients` - Computed gradients
/// * `learning_rate` - Step size for the update (must be positive)
///
/// # Returns
/// Updated parameters, or error if inputs are invalid
///
/// # Errors
/// Returns error if:
/// - Shapes don't match
/// - Data types don't match
/// - Tensors are not floating-point
/// - Learning rate is not positive
///
/// # Example
/// ```
/// use rusttorch_core::{Tensor, DType};
/// use rusttorch_core::ops::sgd_update;
///
/// let params = Tensor::ones(&[10], DType::Float32);
/// let grads = Tensor::from_vec(vec![0.1; 10], &[10]);
/// let updated = sgd_update(&params, &grads, 0.01).unwrap();
/// ```
pub fn sgd_update(params: &Tensor, gradients: &Tensor, learning_rate: f64) -> Result<Tensor> {
    // Validate shapes match
    if params.shape() != gradients.shape() {
        return Err(TensorError::ShapeMismatch {
            expected: params.shape().to_vec(),
            actual: gradients.shape().to_vec(),
            context: "sgd_update".to_string(),
        });
    }

    // Validate data types match
    if params.dtype() != gradients.dtype() {
        return Err(TensorError::DTypeMismatch {
            expected: format!("{}", params.dtype()),
            actual: format!("{}", gradients.dtype()),
            context: "sgd_update".to_string(),
        });
    }

    // Validate learning rate
    if learning_rate <= 0.0 {
        return Err(TensorError::InvalidArgument {
            parameter: "learning_rate".to_string(),
            reason: format!("must be positive, got {}", learning_rate),
        });
    }

    let dtype = params.dtype();
    let lr = learning_rate;

    // Compute update
    let data = match (params.data(), gradients.data()) {
        (TensorData::Float32(p), TensorData::Float32(g)) => {
            TensorData::Float32(p - &(g * lr as f32))
        }
        (TensorData::Float64(p), TensorData::Float64(g)) => TensorData::Float64(p - &(g * lr)),
        _ => {
            return Err(TensorError::DTypeMismatch {
                expected: "float32 or float64".to_string(),
                actual: format!("{}", params.dtype()),
                context: "sgd_update (only floating-point supported)".to_string(),
            })
        }
    };

    Ok(Tensor::from_data(data, dtype))
}

/// SGD with Momentum parameter update
///
/// Updates using exponentially weighted moving average of gradients:
/// velocity = momentum * velocity + gradients
/// params = params - learning_rate * velocity
///
/// # Arguments
/// * `params` - Current parameter values
/// * `gradients` - Computed gradients
/// * `velocity` - Momentum buffer (exponentially weighted average of past gradients)
/// * `learning_rate` - Step size (must be positive)
/// * `momentum` - Momentum coefficient (must be in [0, 1), typically 0.9)
///
/// # Returns
/// (Updated parameters, Updated velocity), or error if inputs are invalid
///
/// # Errors
/// Returns error if shapes don't match, invalid parameters, or non-floating-point tensors
pub fn sgd_momentum_update(
    params: &Tensor,
    gradients: &Tensor,
    velocity: &Tensor,
    learning_rate: f64,
    momentum: f64,
) -> Result<(Tensor, Tensor)> {
    // Validate shapes match
    if params.shape() != gradients.shape() {
        return Err(TensorError::ShapeMismatch {
            expected: params.shape().to_vec(),
            actual: gradients.shape().to_vec(),
            context: "sgd_momentum_update (params vs gradients)".to_string(),
        });
    }
    if params.shape() != velocity.shape() {
        return Err(TensorError::ShapeMismatch {
            expected: params.shape().to_vec(),
            actual: velocity.shape().to_vec(),
            context: "sgd_momentum_update (params vs velocity)".to_string(),
        });
    }

    // Validate learning rate
    if learning_rate <= 0.0 {
        return Err(TensorError::InvalidArgument {
            parameter: "learning_rate".to_string(),
            reason: format!("must be positive, got {}", learning_rate),
        });
    }

    // Validate momentum
    if !(0.0..1.0).contains(&momentum) {
        return Err(TensorError::InvalidArgument {
            parameter: "momentum".to_string(),
            reason: format!("must be in [0, 1), got {}", momentum),
        });
    }

    let dtype = params.dtype();
    let lr = learning_rate;
    let m = momentum;

    // Compute update
    let (new_params_data, new_velocity_data) =
        match (params.data(), gradients.data(), velocity.data()) {
            (TensorData::Float32(p), TensorData::Float32(g), TensorData::Float32(v)) => {
                let new_v = v * m as f32 + g;
                let new_p = p - &(&new_v * lr as f32);
                (TensorData::Float32(new_p), TensorData::Float32(new_v))
            }
            (TensorData::Float64(p), TensorData::Float64(g), TensorData::Float64(v)) => {
                let new_v = v * m + g;
                let new_p = p - &(&new_v * lr);
                (TensorData::Float64(new_p), TensorData::Float64(new_v))
            }
            _ => {
                return Err(TensorError::DTypeMismatch {
                    expected: "float32 or float64".to_string(),
                    actual: format!("{}", params.dtype()),
                    context: "sgd_momentum_update (only floating-point supported)".to_string(),
                })
            }
        };

    Ok((
        Tensor::from_data(new_params_data, dtype),
        Tensor::from_data(new_velocity_data, dtype),
    ))
}

/// Adam optimizer parameter update
///
/// Adaptive Moment Estimation optimizer that computes adaptive learning rates.
///
/// Updates:
/// m = beta1 * m + (1 - beta1) * gradients
/// v = beta2 * v + (1 - beta2) * gradients²
/// m_hat = m / (1 - beta1^t)
/// v_hat = v / (1 - beta2^t)
/// params = params - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
///
/// # Arguments
/// * `params` - Current parameter values
/// * `gradients` - Computed gradients
/// * `m` - First moment estimate (mean of gradients)
/// * `v` - Second moment estimate (uncentered variance of gradients)
/// * `learning_rate` - Step size (must be positive, typically 0.001)
/// * `beta1` - Exponential decay rate for first moment (must be in (0, 1), typically 0.9)
/// * `beta2` - Exponential decay rate for second moment (must be in (0, 1), typically 0.999)
/// * `epsilon` - Small constant for numerical stability (must be positive, typically 1e-8)
/// * `timestep` - Current optimization step (must be >= 1)
///
/// # Returns
/// (Updated parameters, Updated m, Updated v), or error if inputs are invalid
///
/// # Errors
/// Returns error if shapes don't match, invalid parameters, or non-floating-point tensors
#[allow(clippy::too_many_arguments)]
pub fn adam_update(
    params: &Tensor,
    gradients: &Tensor,
    m: &Tensor,
    v: &Tensor,
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    timestep: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    // Validate shapes match
    if params.shape() != gradients.shape() {
        return Err(TensorError::ShapeMismatch {
            expected: params.shape().to_vec(),
            actual: gradients.shape().to_vec(),
            context: "adam_update (params vs gradients)".to_string(),
        });
    }
    if params.shape() != m.shape() {
        return Err(TensorError::ShapeMismatch {
            expected: params.shape().to_vec(),
            actual: m.shape().to_vec(),
            context: "adam_update (params vs m)".to_string(),
        });
    }
    if params.shape() != v.shape() {
        return Err(TensorError::ShapeMismatch {
            expected: params.shape().to_vec(),
            actual: v.shape().to_vec(),
            context: "adam_update (params vs v)".to_string(),
        });
    }

    // Validate parameters
    if learning_rate <= 0.0 {
        return Err(TensorError::InvalidArgument {
            parameter: "learning_rate".to_string(),
            reason: format!("must be positive, got {}", learning_rate),
        });
    }
    if beta1 <= 0.0 || beta1 >= 1.0 {
        return Err(TensorError::InvalidArgument {
            parameter: "beta1".to_string(),
            reason: format!("must be in (0, 1), got {}", beta1),
        });
    }
    if beta2 <= 0.0 || beta2 >= 1.0 {
        return Err(TensorError::InvalidArgument {
            parameter: "beta2".to_string(),
            reason: format!("must be in (0, 1), got {}", beta2),
        });
    }
    if epsilon <= 0.0 {
        return Err(TensorError::InvalidArgument {
            parameter: "epsilon".to_string(),
            reason: format!("must be positive, got {}", epsilon),
        });
    }
    if timestep == 0 {
        return Err(TensorError::InvalidArgument {
            parameter: "timestep".to_string(),
            reason: "must be >= 1".to_string(),
        });
    }

    let dtype = params.dtype();
    let lr = learning_rate;
    let b1 = beta1;
    let b2 = beta2;
    let eps = epsilon;
    let t = timestep as f64;

    // Bias correction terms
    let bias_correction1 = 1.0 - b1.powf(t);
    let bias_correction2 = 1.0 - b2.powf(t);

    // Compute update
    let (new_params_data, new_m_data, new_v_data) =
        match (params.data(), gradients.data(), m.data(), v.data()) {
            (
                TensorData::Float32(p),
                TensorData::Float32(g),
                TensorData::Float32(m_old),
                TensorData::Float32(v_old),
            ) => {
                // Update biased first moment estimate
                let new_m = m_old * b1 as f32 + g * (1.0 - b1 as f32);

                // Update biased second raw moment estimate
                let g_squared = g * g;
                let new_v = v_old * b2 as f32 + &g_squared * (1.0 - b2 as f32);

                // Compute bias-corrected first moment estimate
                let m_hat = &new_m / bias_correction1 as f32;

                // Compute bias-corrected second raw moment estimate
                let v_hat = &new_v / bias_correction2 as f32;

                // Update parameters
                let denominator = v_hat.mapv(|x| x.sqrt() + eps as f32);
                let update = &m_hat / &denominator;
                let new_p = p - &(&update * lr as f32);

                (
                    TensorData::Float32(new_p),
                    TensorData::Float32(new_m),
                    TensorData::Float32(new_v),
                )
            }
            (
                TensorData::Float64(p),
                TensorData::Float64(g),
                TensorData::Float64(m_old),
                TensorData::Float64(v_old),
            ) => {
                let new_m = m_old * b1 + g * (1.0 - b1);
                let g_squared = g * g;
                let new_v = v_old * b2 + &g_squared * (1.0 - b2);

                let m_hat = &new_m / bias_correction1;
                let v_hat = &new_v / bias_correction2;

                let denominator = v_hat.mapv(|x| x.sqrt() + eps);
                let update = &m_hat / &denominator;
                let new_p = p - &(&update * lr);

                (
                    TensorData::Float64(new_p),
                    TensorData::Float64(new_m),
                    TensorData::Float64(new_v),
                )
            }
            _ => {
                return Err(TensorError::DTypeMismatch {
                    expected: "float32 or float64".to_string(),
                    actual: format!("{}", params.dtype()),
                    context: "adam_update (only floating-point supported)".to_string(),
                })
            }
        };

    Ok((
        Tensor::from_data(new_params_data, dtype),
        Tensor::from_data(new_m_data, dtype),
        Tensor::from_data(new_v_data, dtype),
    ))
}

/// AdamW optimizer parameter update
///
/// Adam with decoupled weight decay regularization.
/// This separates weight decay from the gradient-based update.
///
/// # Arguments
/// * `params` - Current parameter values
/// * `gradients` - Computed gradients
/// * `m` - First moment estimate
/// * `v` - Second moment estimate
/// * `learning_rate` - Step size
/// * `beta1` - Exponential decay rate for first moment
/// * `beta2` - Exponential decay rate for second moment
/// * `epsilon` - Small constant for numerical stability
/// * `weight_decay` - Weight decay coefficient (must be non-negative, L2 penalty)
/// * `timestep` - Current optimization step
///
/// # Returns
/// (Updated parameters, Updated m, Updated v), or error if inputs are invalid
///
/// # Errors
/// Returns error if weight_decay is negative or if adam_update returns an error
#[allow(clippy::too_many_arguments)]
pub fn adamw_update(
    params: &Tensor,
    gradients: &Tensor,
    m: &Tensor,
    v: &Tensor,
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    weight_decay: f64,
    timestep: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    // Validate weight decay
    if weight_decay < 0.0 {
        return Err(TensorError::InvalidArgument {
            parameter: "weight_decay".to_string(),
            reason: format!("must be non-negative, got {}", weight_decay),
        });
    }

    // First do regular Adam update
    let (params_after_adam, new_m, new_v) = adam_update(
        params,
        gradients,
        m,
        v,
        learning_rate,
        beta1,
        beta2,
        epsilon,
        timestep,
    )?;

    // Then apply weight decay directly to parameters
    if weight_decay > 0.0 {
        let dtype = params.dtype();
        let wd = weight_decay;
        let lr = learning_rate;

        let data = match params_after_adam.data() {
            TensorData::Float32(p) => {
                let original = as_float32(params.data())?;
                TensorData::Float32(p - &(original * (lr * wd) as f32))
            }
            TensorData::Float64(p) => {
                let original = as_float64(params.data())?;
                TensorData::Float64(p - &(original * (lr * wd)))
            }
            _ => {
                return Err(TensorError::DTypeMismatch {
                    expected: "float32 or float64".to_string(),
                    actual: format!("{}", params.dtype()),
                    context: "adamw_update (only floating-point supported)".to_string(),
                })
            }
        };

        Ok((Tensor::from_data(data, dtype), new_m, new_v))
    } else {
        Ok((params_after_adam, new_m, new_v))
    }
}

// Helper functions for TensorData
fn as_float32(data: &TensorData) -> Result<&ndarray::Array<f32, ndarray::IxDyn>> {
    match data {
        TensorData::Float32(arr) => Ok(arr),
        _ => Err(TensorError::DTypeMismatch {
            expected: "float32".to_string(),
            actual: "other type".to_string(),
            context: "as_float32".to_string(),
        }),
    }
}

fn as_float64(data: &TensorData) -> Result<&ndarray::Array<f64, ndarray::IxDyn>> {
    match data {
        TensorData::Float64(arr) => Ok(arr),
        _ => Err(TensorError::DTypeMismatch {
            expected: "float64".to_string(),
            actual: "other type".to_string(),
            context: "as_float64".to_string(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgd_update() {
        let params = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
        let grads = Tensor::from_vec(vec![0.1, 0.2, 0.3], &[3]);
        let updated = sgd_update(&params, &grads, 0.1).unwrap();

        assert_eq!(updated.shape(), &[3]);
        assert_eq!(updated.dtype(), DType::Float32);
    }

    #[test]
    fn test_sgd_momentum_update() {
        let params = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
        let grads = Tensor::from_vec(vec![0.1, 0.2, 0.3], &[3]);
        let velocity = Tensor::zeros(&[3], DType::Float32);

        let (updated_params, updated_velocity) =
            sgd_momentum_update(&params, &grads, &velocity, 0.1, 0.9).unwrap();

        assert_eq!(updated_params.shape(), &[3]);
        assert_eq!(updated_velocity.shape(), &[3]);
    }

    #[test]
    fn test_adam_update() {
        let params = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
        let grads = Tensor::from_vec(vec![0.1, 0.2, 0.3], &[3]);
        let m = Tensor::zeros(&[3], DType::Float32);
        let v = Tensor::zeros(&[3], DType::Float32);

        let (updated_params, updated_m, updated_v) =
            adam_update(&params, &grads, &m, &v, 0.001, 0.9, 0.999, 1e-8, 1).unwrap();

        assert_eq!(updated_params.shape(), &[3]);
        assert_eq!(updated_m.shape(), &[3]);
        assert_eq!(updated_v.shape(), &[3]);
    }

    #[test]
    fn test_adam_multiple_steps() {
        let mut params = Tensor::from_vec(vec![1.0], &[1]);
        let grads = Tensor::from_vec(vec![0.1], &[1]);
        let mut m = Tensor::zeros(&[1], DType::Float32);
        let mut v = Tensor::zeros(&[1], DType::Float32);

        // Run multiple steps to verify timestep handling
        for t in 1..=5 {
            let (new_params, new_m, new_v) =
                adam_update(&params, &grads, &m, &v, 0.001, 0.9, 0.999, 1e-8, t).unwrap();
            params = new_params;
            m = new_m;
            v = new_v;
        }

        assert_eq!(params.shape(), &[1]);
    }

    #[test]
    fn test_sgd_shape_mismatch() {
        let params = Tensor::from_vec(vec![1.0, 2.0], &[2]);
        let grads = Tensor::from_vec(vec![0.1, 0.2, 0.3], &[3]);
        let result = sgd_update(&params, &grads, 0.1);
        assert!(result.is_err());
        match result {
            Err(TensorError::ShapeMismatch { .. }) => {} // Expected
            _ => panic!("Expected ShapeMismatch error"),
        }
    }

    #[test]
    fn test_sgd_negative_lr() {
        let params = Tensor::from_vec(vec![1.0], &[1]);
        let grads = Tensor::from_vec(vec![0.1], &[1]);
        let result = sgd_update(&params, &grads, -0.1);
        assert!(result.is_err());
        match result {
            Err(TensorError::InvalidArgument { .. }) => {} // Expected
            _ => panic!("Expected InvalidArgument error"),
        }
    }

    #[test]
    fn test_adam_zero_timestep() {
        let params = Tensor::from_vec(vec![1.0], &[1]);
        let grads = Tensor::from_vec(vec![0.1], &[1]);
        let m = Tensor::zeros(&[1], DType::Float32);
        let v = Tensor::zeros(&[1], DType::Float32);
        let result = adam_update(&params, &grads, &m, &v, 0.001, 0.9, 0.999, 1e-8, 0);
        assert!(result.is_err());
        match result {
            Err(TensorError::InvalidArgument { .. }) => {} // Expected
            _ => panic!("Expected InvalidArgument error"),
        }
    }
}
