//! Activation functions for neural networks

use crate::tensor::{Tensor, TensorData, DType};
use crate::error::{Result, TensorError};
use std::f32::consts::PI;

/// ReLU activation: max(0, x)
pub fn relu(tensor: &Tensor) -> Tensor {
    let dtype = tensor.dtype();
    let data = match tensor.data() {
        TensorData::Float32(arr) => TensorData::Float32(arr.mapv(|x| x.max(0.0))),
        TensorData::Float64(arr) => TensorData::Float64(arr.mapv(|x| x.max(0.0))),
        TensorData::Int32(arr) => TensorData::Int32(arr.mapv(|x| x.max(0))),
        TensorData::Int64(arr) => TensorData::Int64(arr.mapv(|x| x.max(0))),
    };

    Tensor::from_data(data, dtype)
}

/// Sigmoid activation: 1 / (1 + exp(-x))
pub fn sigmoid(tensor: &Tensor) -> Result<Tensor> {
    if !tensor.dtype().is_float() {
        return Err(TensorError::InvalidArgument {
            parameter: "dtype".to_string(),
            reason: "Sigmoid requires floating-point tensors".to_string(),
        });
    }

    let dtype = tensor.dtype();
    let data = match tensor.data() {
        TensorData::Float32(arr) => TensorData::Float32(arr.mapv(|x| 1.0 / (1.0 + (-x).exp()))),
        TensorData::Float64(arr) => TensorData::Float64(arr.mapv(|x| 1.0 / (1.0 + (-x).exp()))),
        _ => panic!("Sigmoid only supports floating-point types"),
    };

    Ok(Tensor::from_data(data, dtype))
}

/// Tanh activation
pub fn tanh(tensor: &Tensor) -> Result<Tensor> {
    if !tensor.dtype().is_float() {
        return Err(TensorError::InvalidArgument {
            parameter: "dtype".to_string(),
            reason: "Tanh requires floating-point tensors".to_string(),
        });
    }

    let dtype = tensor.dtype();
    let data = match tensor.data() {
        TensorData::Float32(arr) => TensorData::Float32(arr.mapv(|x| x.tanh())),
        TensorData::Float64(arr) => TensorData::Float64(arr.mapv(|x| x.tanh())),
        _ => panic!("Tanh only supports floating-point types"),
    };

    Ok(Tensor::from_data(data, dtype))
}

/// GELU activation (Gaussian Error Linear Unit)
/// GELU(x) = x * Φ(x) where Φ(x) is the cumulative distribution function of the standard normal distribution
/// Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
pub fn gelu(tensor: &Tensor) -> Result<Tensor> {
    if !tensor.dtype().is_float() {
        return Err(TensorError::InvalidArgument {
            parameter: "dtype".to_string(),
            reason: "GELU requires floating-point tensors".to_string(),
        });
    }

    let dtype = tensor.dtype();
    let data =
        match tensor.data() {
            TensorData::Float32(arr) => {
                let sqrt_2_over_pi = (2.0_f32 / PI).sqrt();
                TensorData::Float32(arr.mapv(|x| {
                    0.5 * x * (1.0 + (sqrt_2_over_pi * (x + 0.044715 * x.powi(3))).tanh())
                }))
            }
            TensorData::Float64(arr) => {
                let sqrt_2_over_pi = (2.0_f64 / std::f64::consts::PI).sqrt();
                TensorData::Float64(arr.mapv(|x| {
                    0.5 * x * (1.0 + (sqrt_2_over_pi * (x + 0.044715 * x.powi(3))).tanh())
                }))
            }
            _ => panic!("GELU only supports floating-point types"),
        };

    Ok(Tensor::from_data(data, dtype))
}

/// Softmax activation along a dimension
/// softmax(x_i) = exp(x_i) / sum(exp(x_j))
pub fn softmax(tensor: &Tensor, dim: usize) -> Result<Tensor> {
    if !tensor.dtype().is_float() {
        return Err(TensorError::InvalidArgument {
            parameter: "dtype".to_string(),
            reason: "Softmax requires floating-point tensors".to_string(),
        });
    }
    if dim >= tensor.ndim() {
        return Err(TensorError::InvalidDimension {
            dimension: dim,
            max_dimension: tensor.ndim() - 1,
            context: "softmax".to_string(),
        });
    }

    let dtype = tensor.dtype();
    let data = match tensor.data() {
        TensorData::Float32(arr) => {
            // Subtract max for numerical stability
            let max_vals = arr.map_axis(ndarray::Axis(dim), |view| {
                view.iter()
                    .copied()
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or(0.0)
            });

            // Compute exp(x - max)
            let mut exp_arr = arr.clone();
            ndarray::Zip::from(&mut exp_arr)
                .and_broadcast(&max_vals.insert_axis(ndarray::Axis(dim)))
                .for_each(|x, &max_val| {
                    *x = (*x - max_val).exp();
                });

            // Compute sum of exps along dimension
            let sum_exp = exp_arr.sum_axis(ndarray::Axis(dim));

            // Divide by sum
            ndarray::Zip::from(&mut exp_arr)
                .and_broadcast(&sum_exp.insert_axis(ndarray::Axis(dim)))
                .for_each(|x, &sum_val| {
                    *x /= sum_val;
                });

            TensorData::Float32(exp_arr)
        }
        TensorData::Float64(arr) => {
            let max_vals = arr.map_axis(ndarray::Axis(dim), |view| {
                view.iter()
                    .copied()
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or(0.0)
            });

            let mut exp_arr = arr.clone();
            ndarray::Zip::from(&mut exp_arr)
                .and_broadcast(&max_vals.insert_axis(ndarray::Axis(dim)))
                .for_each(|x, &max_val| {
                    *x = (*x - max_val).exp();
                });

            let sum_exp = exp_arr.sum_axis(ndarray::Axis(dim));

            ndarray::Zip::from(&mut exp_arr)
                .and_broadcast(&sum_exp.insert_axis(ndarray::Axis(dim)))
                .for_each(|x, &sum_val| {
                    *x /= sum_val;
                });

            TensorData::Float64(exp_arr)
        }
        _ => panic!("Softmax only supports floating-point types"),
    };

    Ok(Tensor::from_data(data, dtype))
}

/// Leaky ReLU activation: max(alpha * x, x)
pub fn leaky_relu(tensor: &Tensor, alpha: f32) -> Tensor {
    let dtype = tensor.dtype();
    let data = match tensor.data() {
        TensorData::Float32(arr) => {
            TensorData::Float32(arr.mapv(|x| if x > 0.0 { x } else { alpha * x }))
        }
        TensorData::Float64(arr) => {
            let alpha = alpha as f64;
            TensorData::Float64(arr.mapv(|x| if x > 0.0 { x } else { alpha * x }))
        }
        TensorData::Int32(arr) => {
            let alpha = alpha as i32;
            TensorData::Int32(arr.mapv(|x| if x > 0 { x } else { alpha * x }))
        }
        TensorData::Int64(arr) => {
            let alpha = alpha as i64;
            TensorData::Int64(arr.mapv(|x| if x > 0 { x } else { alpha * x }))
        }
    };

    Tensor::from_data(data, dtype)
}

/// ELU (Exponential Linear Unit) activation
/// ELU(x) = x if x > 0, alpha * (exp(x) - 1) if x <= 0
pub fn elu(tensor: &Tensor, alpha: f32) -> Result<Tensor> {
    if !tensor.dtype().is_float() {
        return Err(TensorError::InvalidArgument {
            parameter: "dtype".to_string(),
            reason: "ELU requires floating-point tensors".to_string(),
        });
    }
    if alpha <= 0.0 {
        return Err(TensorError::InvalidArgument {
            parameter: "alpha".to_string(),
            reason: "Alpha must be positive".to_string(),
        });
    }

    let dtype = tensor.dtype();
    let data = match tensor.data() {
        TensorData::Float32(arr) => {
            TensorData::Float32(arr.mapv(|x| if x > 0.0 { x } else { alpha * (x.exp() - 1.0) }))
        }
        TensorData::Float64(arr) => {
            let alpha = alpha as f64;
            TensorData::Float64(arr.mapv(|x| if x > 0.0 { x } else { alpha * (x.exp() - 1.0) }))
        }
        _ => panic!("ELU only supports floating-point types"),
    };

    Ok(Tensor::from_data(data, dtype))
}

/// SELU (Scaled Exponential Linear Unit) activation
/// Self-normalizing activation function
/// SELU(x) = scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))
/// where alpha ≈ 1.6733 and scale ≈ 1.0507
pub fn selu(tensor: &Tensor) -> Result<Tensor> {
    if !tensor.dtype().is_float() {
        return Err(TensorError::InvalidArgument {
            parameter: "dtype".to_string(),
            reason: "SELU requires floating-point tensors".to_string(),
        });
    }

    const ALPHA: f64 = 1.673_263_242_354_377_2;
    const SCALE: f64 = 1.050_700_987_355_480_5;

    let alpha = ALPHA;
    let scale = SCALE;

    let dtype = tensor.dtype();
    let data = match tensor.data() {
        TensorData::Float32(arr) => TensorData::Float32(arr.mapv(|x| {
            scale as f32
                * if x > 0.0 {
                    x
                } else {
                    alpha as f32 * (x.exp() - 1.0)
                }
        })),
        TensorData::Float64(arr) => TensorData::Float64(
            arr.mapv(|x| scale * if x > 0.0 { x } else { alpha * (x.exp() - 1.0) }),
        ),
        _ => panic!("SELU only supports floating-point types"),
    };

    Ok(Tensor::from_data(data, dtype))
}

/// Swish activation (also known as SiLU - Sigmoid Linear Unit)
/// Swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
pub fn swish(tensor: &Tensor) -> Result<Tensor> {
    if !tensor.dtype().is_float() {
        return Err(TensorError::InvalidArgument {
            parameter: "dtype".to_string(),
            reason: "Swish requires floating-point tensors".to_string(),
        });
    }

    let dtype = tensor.dtype();
    let data = match tensor.data() {
        TensorData::Float32(arr) => TensorData::Float32(arr.mapv(|x| x / (1.0 + (-x).exp()))),
        TensorData::Float64(arr) => TensorData::Float64(arr.mapv(|x| x / (1.0 + (-x).exp()))),
        _ => panic!("Swish only supports floating-point types"),
    };

    Ok(Tensor::from_data(data, dtype))
}

/// Mish activation
/// Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
/// A smooth, non-monotonic activation function
pub fn mish(tensor: &Tensor) -> Result<Tensor> {
    if !tensor.dtype().is_float() {
        return Err(TensorError::InvalidArgument {
            parameter: "dtype".to_string(),
            reason: "Mish requires floating-point tensors".to_string(),
        });
    }

    let dtype = tensor.dtype();
    let data = match tensor.data() {
        TensorData::Float32(arr) => {
            TensorData::Float32(arr.mapv(|x| {
                // softplus(x) = ln(1 + exp(x))
                // For numerical stability, use different formulas for different ranges
                let softplus = if x > 20.0 {
                    x // For large x, softplus(x) ≈ x
                } else {
                    (1.0 + x.exp()).ln()
                };
                x * softplus.tanh()
            }))
        }
        TensorData::Float64(arr) => TensorData::Float64(arr.mapv(|x| {
            let softplus = if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
            x * softplus.tanh()
        })),
        _ => panic!("Mish only supports floating-point types"),
    };

    Ok(Tensor::from_data(data, dtype))
}

/// Softplus activation
/// Softplus(x) = ln(1 + exp(x))
/// A smooth approximation of ReLU
pub fn softplus(tensor: &Tensor) -> Result<Tensor> {
    if !tensor.dtype().is_float() {
        return Err(TensorError::InvalidArgument {
            parameter: "dtype".to_string(),
            reason: "Softplus requires floating-point tensors".to_string(),
        });
    }

    let dtype = tensor.dtype();
    let data = match tensor.data() {
        TensorData::Float32(arr) => {
            TensorData::Float32(arr.mapv(|x| {
                // For numerical stability
                if x > 20.0 {
                    x
                } else {
                    (1.0 + x.exp()).ln()
                }
            }))
        }
        TensorData::Float64(arr) => {
            TensorData::Float64(arr.mapv(|x| if x > 20.0 { x } else { (1.0 + x.exp()).ln() }))
        }
        _ => panic!("Softplus only supports floating-point types"),
    };

    Ok(Tensor::from_data(data, dtype))
}

/// Softsign activation
/// Softsign(x) = x / (1 + |x|)
/// Similar to tanh but with polynomial tail decay
pub fn softsign(tensor: &Tensor) -> Result<Tensor> {
    if !tensor.dtype().is_float() {
        return Err(TensorError::InvalidArgument {
            parameter: "dtype".to_string(),
            reason: "Softsign requires floating-point tensors".to_string(),
        });
    }

    let dtype = tensor.dtype();
    let data = match tensor.data() {
        TensorData::Float32(arr) => TensorData::Float32(arr.mapv(|x| x / (1.0 + x.abs()))),
        TensorData::Float64(arr) => TensorData::Float64(arr.mapv(|x| x / (1.0 + x.abs()))),
        _ => panic!("Softsign only supports floating-point types"),
    };

    Ok(Tensor::from_data(data, dtype))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu() {
        let t = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]);
        let r = relu(&t);
        assert_eq!(r.shape(), &[5]);
    }

    #[test]
    fn test_sigmoid() {
        let t = Tensor::from_vec(vec![-1.0, 0.0, 1.0], &[3]);
        let s = sigmoid(&t).unwrap();
        assert_eq!(s.shape(), &[3]);
        // Sigmoid(0) should be 0.5
    }

    #[test]
    fn test_tanh() {
        let t = Tensor::from_vec(vec![-1.0, 0.0, 1.0], &[3]);
        let th = tanh(&t).unwrap();
        assert_eq!(th.shape(), &[3]);
    }

    #[test]
    fn test_gelu() {
        let t = Tensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0], &[4]);
        let g = gelu(&t).unwrap();
        assert_eq!(g.shape(), &[4]);
    }

    #[test]
    fn test_softmax() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let s = softmax(&t, 1).unwrap();
        assert_eq!(s.shape(), &[2, 3]);
    }

    #[test]
    fn test_leaky_relu() {
        let t = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]);
        let lr = leaky_relu(&t, 0.01);
        assert_eq!(lr.shape(), &[5]);
    }

    #[test]
    fn test_sigmoid_int_error() {
        let t = Tensor::ones(&[2, 2], DType::Int32);
        let result = sigmoid(&t);
        assert!(result.is_err());
        match result.unwrap_err() {
            TensorError::InvalidArgument { .. } => {}
            _ => panic!("Expected InvalidArgument error"),
        }
    }

    #[test]
    fn test_softmax_dim_out_of_range() {
        let t = Tensor::ones(&[2, 3], DType::Float32);
        let result = softmax(&t, 5);
        assert!(result.is_err());
        match result.unwrap_err() {
            TensorError::InvalidDimension { .. } => {}
            _ => panic!("Expected InvalidDimension error"),
        }
    }

    #[test]
    fn test_elu() {
        let t = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]);
        let e = elu(&t, 1.0).unwrap();
        assert_eq!(e.shape(), &[5]);
    }

    #[test]
    fn test_selu() {
        let t = Tensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0], &[4]);
        let s = selu(&t).unwrap();
        assert_eq!(s.shape(), &[4]);
    }

    #[test]
    fn test_swish() {
        let t = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]);
        let sw = swish(&t).unwrap();
        assert_eq!(sw.shape(), &[5]);
    }

    #[test]
    fn test_mish() {
        let t = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]);
        let m = mish(&t).unwrap();
        assert_eq!(m.shape(), &[5]);
    }

    #[test]
    fn test_softplus() {
        let t = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]);
        let sp = softplus(&t).unwrap();
        assert_eq!(sp.shape(), &[5]);
    }

    #[test]
    fn test_softsign() {
        let t = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]);
        let ss = softsign(&t).unwrap();
        assert_eq!(ss.shape(), &[5]);
    }

    #[test]
    fn test_elu_negative_alpha() {
        let t = Tensor::from_vec(vec![1.0], &[1]);
        let result = elu(&t, -0.5);
        assert!(result.is_err());
        match result.unwrap_err() {
            TensorError::InvalidArgument { .. } => {}
            _ => panic!("Expected InvalidArgument error"),
        }
    }
}
