//! Loss functions for neural networks
//!
//! This module implements common loss functions used in deep learning:
//! - Mean Squared Error (MSE)
//! - Mean Absolute Error (L1 Loss)
//! - Cross-Entropy Loss
//! - Binary Cross-Entropy Loss
//!
//! All functions return `Result<f64, TensorError>` for proper error handling.

use crate::error::{TensorError, Result};
use crate::tensor::{Tensor, TensorData};
use crate::DType;

/// Mean Squared Error loss
///
/// Computes: 1/n * Σ(predictions - targets)²
///
/// # Arguments
/// * `predictions` - Predicted values
/// * `targets` - Target/ground truth values
///
/// # Returns
/// Scalar loss value as f64, or error if inputs are invalid
///
/// # Errors
/// Returns error if:
/// - Shapes don't match
/// - Data types don't match
/// - Tensors are not floating-point
/// - Tensors are empty
///
/// # Example
/// ```
/// use rusttorch_core::{Tensor, DType};
/// use rusttorch_core::ops::mse_loss;
///
/// let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
/// let target = Tensor::from_vec(vec![1.5, 2.5, 2.5], &[3]);
/// let loss = mse_loss(&pred, &target).unwrap();
/// ```
pub fn mse_loss(predictions: &Tensor, targets: &Tensor) -> Result<f64> {
    // Validate shapes match
    if predictions.shape() != targets.shape() {
        return Err(TensorError::ShapeMismatch {
            expected: targets.shape().to_vec(),
            actual: predictions.shape().to_vec(),
            context: "mse_loss".to_string(),
        });
    }

    // Validate data types match
    if predictions.dtype() != targets.dtype() {
        return Err(TensorError::DTypeMismatch {
            expected: format!("{}", targets.dtype()),
            actual: format!("{}", predictions.dtype()),
            context: "mse_loss".to_string(),
        });
    }

    // Check for empty tensors
    let n = predictions.numel();
    if n == 0 {
        return Err(TensorError::EmptyTensor {
            operation: "mse_loss".to_string(),
        });
    }
    let n = n as f64;

    // Compute loss
    match (predictions.data(), targets.data()) {
        (TensorData::Float32(pred), TensorData::Float32(targ)) => {
            let diff = pred - targ;
            let squared = &diff * &diff;
            Ok(squared.sum() as f64 / n)
        }
        (TensorData::Float64(pred), TensorData::Float64(targ)) => {
            let diff = pred - targ;
            let squared = &diff * &diff;
            Ok(squared.sum() / n)
        }
        _ => Err(TensorError::DTypeMismatch {
            expected: "float32 or float64".to_string(),
            actual: format!("{}", predictions.dtype()),
            context: "mse_loss (only floating-point supported)".to_string(),
        }),
    }
}

/// Mean Absolute Error (L1 Loss)
///
/// Computes: 1/n * Σ|predictions - targets|
///
/// # Arguments
/// * `predictions` - Predicted values
/// * `targets` - Target/ground truth values
///
/// # Returns
/// Scalar loss value as f64, or error if inputs are invalid
///
/// # Errors
/// Returns error if shapes/dtypes don't match or tensors are empty
pub fn l1_loss(predictions: &Tensor, targets: &Tensor) -> Result<f64> {
    // Validate shapes match
    if predictions.shape() != targets.shape() {
        return Err(TensorError::ShapeMismatch {
            expected: targets.shape().to_vec(),
            actual: predictions.shape().to_vec(),
            context: "l1_loss".to_string(),
        });
    }

    // Validate data types match
    if predictions.dtype() != targets.dtype() {
        return Err(TensorError::DTypeMismatch {
            expected: format!("{}", targets.dtype()),
            actual: format!("{}", predictions.dtype()),
            context: "l1_loss".to_string(),
        });
    }

    // Check for empty tensors
    let n = predictions.numel();
    if n == 0 {
        return Err(TensorError::EmptyTensor {
            operation: "l1_loss".to_string(),
        });
    }
    let n = n as f64;

    // Compute loss
    match (predictions.data(), targets.data()) {
        (TensorData::Float32(pred), TensorData::Float32(targ)) => {
            let diff = pred - targ;
            Ok(diff.mapv(|x| x.abs()).sum() as f64 / n)
        }
        (TensorData::Float64(pred), TensorData::Float64(targ)) => {
            let diff = pred - targ;
            Ok(diff.mapv(|x| x.abs()).sum() / n)
        }
        _ => Err(TensorError::DTypeMismatch {
            expected: "float32 or float64".to_string(),
            actual: format!("{}", predictions.dtype()),
            context: "l1_loss (only floating-point supported)".to_string(),
        }),
    }
}

/// Smooth L1 Loss (Huber Loss)
///
/// Combines L1 and L2 loss for robust regression:
/// - Uses L2 for small errors (|x| < beta)
/// - Uses L1 for large errors (|x| >= beta)
///
/// Formula:
/// - 0.5 * x² / beta if |x| < beta
/// - |x| - 0.5 * beta otherwise
///
/// # Arguments
/// * `predictions` - Predicted values
/// * `targets` - Target/ground truth values
/// * `beta` - Threshold for switching between L1 and L2 (must be positive)
///
/// # Errors
/// Returns error if shapes/dtypes don't match, tensors are empty, or beta <= 0
pub fn smooth_l1_loss(predictions: &Tensor, targets: &Tensor, beta: f64) -> Result<f64> {
    // Validate shapes match
    if predictions.shape() != targets.shape() {
        return Err(TensorError::ShapeMismatch {
            expected: targets.shape().to_vec(),
            actual: predictions.shape().to_vec(),
            context: "smooth_l1_loss".to_string(),
        });
    }

    // Validate beta parameter
    if beta <= 0.0 {
        return Err(TensorError::InvalidArgument {
            parameter: "beta".to_string(),
            reason: format!("must be positive, got {}", beta),
        });
    }

    // Check for empty tensors
    let n = predictions.numel();
    if n == 0 {
        return Err(TensorError::EmptyTensor {
            operation: "smooth_l1_loss".to_string(),
        });
    }
    let n = n as f64;

    // Compute loss
    match (predictions.data(), targets.data()) {
        (TensorData::Float32(pred), TensorData::Float32(targ)) => {
            let diff = pred - targ;
            let loss = diff.mapv(|x| {
                let abs_x = x.abs();
                if abs_x < beta as f32 {
                    0.5 * x * x / beta as f32
                } else {
                    abs_x - 0.5 * beta as f32
                }
            });
            Ok(loss.sum() as f64 / n)
        }
        (TensorData::Float64(pred), TensorData::Float64(targ)) => {
            let diff = pred - targ;
            let loss = diff.mapv(|x| {
                let abs_x = x.abs();
                if abs_x < beta {
                    0.5 * x * x / beta
                } else {
                    abs_x - 0.5 * beta
                }
            });
            Ok(loss.sum() / n)
        }
        _ => Err(TensorError::DTypeMismatch {
            expected: "float32 or float64".to_string(),
            actual: format!("{}", predictions.dtype()),
            context: "smooth_l1_loss (only floating-point supported)".to_string(),
        }),
    }
}

/// Binary Cross-Entropy Loss
///
/// Computes: -1/n * Σ(targets * log(predictions) + (1 - targets) * log(1 - predictions))
///
/// # Arguments
/// * `predictions` - Predicted probabilities (should be in range [0, 1])
/// * `targets` - Target values (should be 0 or 1)
/// * `epsilon` - Small value to avoid log(0), must be in (0, 1)
///
/// # Returns
/// Scalar loss value as f64
///
/// # Errors
/// Returns error if shapes don't match, tensors are empty, or epsilon is invalid
pub fn binary_cross_entropy_loss(predictions: &Tensor, targets: &Tensor, epsilon: f64) -> Result<f64> {
    // Validate shapes match
    if predictions.shape() != targets.shape() {
        return Err(TensorError::ShapeMismatch {
            expected: targets.shape().to_vec(),
            actual: predictions.shape().to_vec(),
            context: "binary_cross_entropy_loss".to_string(),
        });
    }

    // Validate epsilon parameter
    if epsilon <= 0.0 || epsilon >= 1.0 {
        return Err(TensorError::InvalidArgument {
            parameter: "epsilon".to_string(),
            reason: format!("must be in (0, 1), got {}", epsilon),
        });
    }

    // Check for empty tensors
    let n = predictions.numel();
    if n == 0 {
        return Err(TensorError::EmptyTensor {
            operation: "binary_cross_entropy_loss".to_string(),
        });
    }
    let n = n as f64;
    let eps = epsilon;

    // Compute loss
    match (predictions.data(), targets.data()) {
        (TensorData::Float32(pred), TensorData::Float32(targ)) => {
            let loss = pred
                .iter()
                .zip(targ.iter())
                .map(|(p, t)| {
                    // Clamp predictions to [epsilon, 1-epsilon] to avoid log(0)
                    let p_clamped = p.max(eps as f32).min(1.0 - eps as f32);
                    -(t * p_clamped.ln() + (1.0 - t) * (1.0 - p_clamped).ln())
                })
                .sum::<f32>();
            Ok(loss as f64 / n)
        }
        (TensorData::Float64(pred), TensorData::Float64(targ)) => {
            let loss = pred
                .iter()
                .zip(targ.iter())
                .map(|(p, t)| {
                    let p_clamped = p.max(eps).min(1.0 - eps);
                    -(t * p_clamped.ln() + (1.0 - t) * (1.0 - p_clamped).ln())
                })
                .sum::<f64>();
            Ok(loss / n)
        }
        _ => Err(TensorError::DTypeMismatch {
            expected: "float32 or float64".to_string(),
            actual: format!("{}", predictions.dtype()),
            context: "binary_cross_entropy_loss (only floating-point supported)".to_string(),
        }),
    }
}

/// Cross-Entropy Loss (for multi-class classification)
///
/// Computes: -1/n * Σ Σ(targets * log(predictions))
///
/// # Arguments
/// * `predictions` - Predicted probabilities (should sum to 1 along last dimension)
/// * `targets` - Target probabilities or one-hot encoded labels
/// * `epsilon` - Small value to avoid log(0), must be in (0, 1)
///
/// # Returns
/// Scalar loss value as f64
///
/// # Errors
/// Returns error if shapes don't match, tensors are empty, or epsilon is invalid
///
/// # Note
/// This expects predictions to already be passed through softmax.
/// For numerical stability, use cross_entropy_with_logits instead.
pub fn cross_entropy_loss(predictions: &Tensor, targets: &Tensor, epsilon: f64) -> Result<f64> {
    // Validate shapes match
    if predictions.shape() != targets.shape() {
        return Err(TensorError::ShapeMismatch {
            expected: targets.shape().to_vec(),
            actual: predictions.shape().to_vec(),
            context: "cross_entropy_loss".to_string(),
        });
    }

    // Validate epsilon parameter
    if epsilon <= 0.0 || epsilon >= 1.0 {
        return Err(TensorError::InvalidArgument {
            parameter: "epsilon".to_string(),
            reason: format!("must be in (0, 1), got {}", epsilon),
        });
    }

    // Check for empty tensors
    let n = predictions.numel();
    if n == 0 {
        return Err(TensorError::EmptyTensor {
            operation: "cross_entropy_loss".to_string(),
        });
    }
    let n = n as f64;
    let eps = epsilon;

    // Compute loss
    match (predictions.data(), targets.data()) {
        (TensorData::Float32(pred), TensorData::Float32(targ)) => {
            let loss = pred
                .iter()
                .zip(targ.iter())
                .map(|(p, t)| {
                    let p_clamped = p.max(eps as f32);
                    -t * p_clamped.ln()
                })
                .sum::<f32>();
            Ok(loss as f64 / n)
        }
        (TensorData::Float64(pred), TensorData::Float64(targ)) => {
            let loss = pred
                .iter()
                .zip(targ.iter())
                .map(|(p, t)| {
                    let p_clamped = p.max(eps);
                    -t * p_clamped.ln()
                })
                .sum::<f64>();
            Ok(loss / n)
        }
        _ => Err(TensorError::DTypeMismatch {
            expected: "float32 or float64".to_string(),
            actual: format!("{}", predictions.dtype()),
            context: "cross_entropy_loss (only floating-point supported)".to_string(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse_loss() {
        let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]);
        let target = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]);
        let loss = mse_loss(&pred, &target).unwrap();
        assert!((loss - 0.0).abs() < 1e-6, "Perfect predictions should have zero loss");
    }

    #[test]
    fn test_mse_loss_nonzero() {
        let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
        let target = Tensor::from_vec(vec![2.0, 2.0, 2.0], &[3]);
        let loss = mse_loss(&pred, &target).unwrap();
        // (1² + 0² + 1²) / 3 = 2/3 ≈ 0.6667
        assert!((loss - 0.6667).abs() < 0.001);
    }

    #[test]
    fn test_l1_loss() {
        let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
        let target = Tensor::from_vec(vec![2.0, 2.0, 2.0], &[3]);
        let loss = l1_loss(&pred, &target).unwrap();
        // (|1-2| + |2-2| + |3-2|) / 3 = (1 + 0 + 1) / 3 = 0.6667
        assert!((loss - 0.6667).abs() < 0.001);
    }

    #[test]
    fn test_smooth_l1_loss() {
        let pred = Tensor::from_vec(vec![0.0, 2.0, 5.0], &[3]);
        let target = Tensor::from_vec(vec![0.0, 0.0, 0.0], &[3]);
        let loss = smooth_l1_loss(&pred, &target, 1.0).unwrap();
        // x=0: 0
        // x=2: 2 - 0.5 = 1.5 (large error, L1)
        // x=5: 5 - 0.5 = 4.5 (large error, L1)
        // Sum = 6.0, Mean = 2.0
        assert!((loss - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_binary_cross_entropy() {
        let pred = Tensor::from_vec(vec![0.9, 0.1, 0.8, 0.2], &[4]);
        let target = Tensor::from_vec(vec![1.0, 0.0, 1.0, 0.0], &[4]);
        let loss = binary_cross_entropy_loss(&pred, &target, 1e-7).unwrap();
        // Should be low because predictions match targets well
        assert!(loss > 0.0 && loss < 0.5);
    }

    #[test]
    fn test_cross_entropy() {
        // Simple 2-class example
        let pred = Tensor::from_vec(vec![0.7, 0.3, 0.4, 0.6], &[2, 2]);
        let target = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let loss = cross_entropy_loss(&pred, &target, 1e-7).unwrap();
        assert!(loss > 0.0);
    }

    #[test]
    fn test_mse_shape_mismatch() {
        let pred = Tensor::from_vec(vec![1.0, 2.0], &[2]);
        let target = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
        let result = mse_loss(&pred, &target);
        assert!(result.is_err());
        match result {
            Err(TensorError::ShapeMismatch { .. }) => {}, // Expected
            _ => panic!("Expected ShapeMismatch error"),
        }
    }

    #[test]
    fn test_mse_dtype_mismatch() {
        let pred = Tensor::ones(&[2, 2], DType::Float32);
        let target = Tensor::ones(&[2, 2], DType::Float64);
        let result = mse_loss(&pred, &target);
        assert!(result.is_err());
        match result {
            Err(TensorError::DTypeMismatch { .. }) => {}, // Expected
            _ => panic!("Expected DTypeMismatch error"),
        }
    }

    #[test]
    fn test_mse_empty_tensor() {
        let pred = Tensor::from_vec(vec![], &[0]);
        let target = Tensor::from_vec(vec![], &[0]);
        let result = mse_loss(&pred, &target);
        assert!(result.is_err());
        match result {
            Err(TensorError::EmptyTensor { .. }) => {}, // Expected
            _ => panic!("Expected EmptyTensor error"),
        }
    }
}
