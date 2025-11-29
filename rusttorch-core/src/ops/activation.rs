//! Activation functions for neural networks

use crate::tensor::{Tensor, TensorData};
use crate::DType;
use ndarray::Array;
use std::f32::consts::PI;

/// ReLU activation: max(0, x)
pub fn relu(tensor: &Tensor) -> Tensor {
    let dtype = tensor.dtype();
    let data = match tensor.data() {
        TensorData::Float32(arr) => {
            TensorData::Float32(arr.mapv(|x| x.max(0.0)))
        }
        TensorData::Float64(arr) => {
            TensorData::Float64(arr.mapv(|x| x.max(0.0)))
        }
        TensorData::Int32(arr) => {
            TensorData::Int32(arr.mapv(|x| x.max(0)))
        }
        TensorData::Int64(arr) => {
            TensorData::Int64(arr.mapv(|x| x.max(0)))
        }
    };

    Tensor::from_data(data, dtype)
}

/// Sigmoid activation: 1 / (1 + exp(-x))
pub fn sigmoid(tensor: &Tensor) -> Tensor {
    assert!(tensor.dtype().is_float(), "Sigmoid requires floating-point tensors");

    let dtype = tensor.dtype();
    let data = match tensor.data() {
        TensorData::Float32(arr) => {
            TensorData::Float32(arr.mapv(|x| 1.0 / (1.0 + (-x).exp())))
        }
        TensorData::Float64(arr) => {
            TensorData::Float64(arr.mapv(|x| 1.0 / (1.0 + (-x).exp())))
        }
        _ => panic!("Sigmoid only supports floating-point types"),
    };

    Tensor::from_data(data, dtype)
}

/// Tanh activation
pub fn tanh(tensor: &Tensor) -> Tensor {
    assert!(tensor.dtype().is_float(), "Tanh requires floating-point tensors");

    let dtype = tensor.dtype();
    let data = match tensor.data() {
        TensorData::Float32(arr) => {
            TensorData::Float32(arr.mapv(|x| x.tanh()))
        }
        TensorData::Float64(arr) => {
            TensorData::Float64(arr.mapv(|x| x.tanh()))
        }
        _ => panic!("Tanh only supports floating-point types"),
    };

    Tensor::from_data(data, dtype)
}

/// GELU activation (Gaussian Error Linear Unit)
/// GELU(x) = x * Φ(x) where Φ(x) is the cumulative distribution function of the standard normal distribution
/// Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
pub fn gelu(tensor: &Tensor) -> Tensor {
    assert!(tensor.dtype().is_float(), "GELU requires floating-point tensors");

    let dtype = tensor.dtype();
    let data = match tensor.data() {
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

    Tensor::from_data(data, dtype)
}

/// Softmax activation along a dimension
/// softmax(x_i) = exp(x_i) / sum(exp(x_j))
pub fn softmax(tensor: &Tensor, dim: usize) -> Tensor {
    assert!(tensor.dtype().is_float(), "Softmax requires floating-point tensors");
    assert!(dim < tensor.ndim(), "Dimension out of range");

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

    Tensor::from_data(data, dtype)
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
        let s = sigmoid(&t);
        assert_eq!(s.shape(), &[3]);
        // Sigmoid(0) should be 0.5
    }

    #[test]
    fn test_tanh() {
        let t = Tensor::from_vec(vec![-1.0, 0.0, 1.0], &[3]);
        let th = tanh(&t);
        assert_eq!(th.shape(), &[3]);
    }

    #[test]
    fn test_gelu() {
        let t = Tensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0], &[4]);
        let g = gelu(&t);
        assert_eq!(g.shape(), &[4]);
    }

    #[test]
    fn test_softmax() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let s = softmax(&t, 1);
        assert_eq!(s.shape(), &[2, 3]);
    }

    #[test]
    fn test_leaky_relu() {
        let t = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]);
        let lr = leaky_relu(&t, 0.01);
        assert_eq!(lr.shape(), &[5]);
    }

    #[test]
    #[should_panic(expected = "floating-point")]
    fn test_sigmoid_int_panic() {
        let t = Tensor::ones(&[2, 2], DType::Int32);
        let _ = sigmoid(&t);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn test_softmax_dim_out_of_range() {
        let t = Tensor::ones(&[2, 3], DType::Float32);
        let _ = softmax(&t, 5);
    }
}
