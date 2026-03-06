//! SIMD optimizations for tensor operations
//!
//! This module provides SIMD-accelerated implementations of common operations.
//! Currently uses auto-vectorization with iterator patterns optimized for LLVM.
//! Future work: Explicit SIMD using std::simd (unstable) or packed_simd.

use crate::error::{Result, TensorError};
use crate::tensor::{Tensor, TensorData};
use rayon::prelude::*;

/// SIMD-optimized element-wise addition
///
/// Uses auto-vectorization for small-medium tensors and parallel SIMD for large tensors.
pub fn add_simd(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.dtype() != b.dtype() {
        return Err(TensorError::DTypeMismatch {
            expected: format!("{:?}", a.dtype()),
            actual: format!("{:?}", b.dtype()),
            context: "add_simd".to_string(),
        });
    }
    if a.shape() != b.shape() {
        return Err(TensorError::ShapeMismatch {
            expected: a.shape().to_vec(),
            actual: b.shape().to_vec(),
            context: "add_simd".to_string(),
        });
    }

    let dtype = a.dtype();
    let numel = a.numel();

    let data = match (a.data(), b.data()) {
        (TensorData::Float32(arr_a), TensorData::Float32(arr_b)) => {
            if let (Some(slice_a), Some(slice_b)) = (arr_a.as_slice(), arr_b.as_slice()) {
                if numel >= 10_000 {
                    // Parallel SIMD for large tensors
                    let result: Vec<f32> = slice_a
                        .par_iter()
                        .zip(slice_b.par_iter())
                        .map(|(a, b)| a + b)
                        .collect();
                    TensorData::Float32(
                        ndarray::Array::from_shape_vec(arr_a.raw_dim(), result).unwrap(),
                    )
                } else {
                    // Sequential SIMD (auto-vectorized by LLVM)
                    let result: Vec<f32> = slice_a
                        .iter()
                        .zip(slice_b.iter())
                        .map(|(a, b)| a + b)
                        .collect();
                    TensorData::Float32(
                        ndarray::Array::from_shape_vec(arr_a.raw_dim(), result).unwrap(),
                    )
                }
            } else {
                TensorData::Float32(arr_a + arr_b)
            }
        }
        (TensorData::Float64(arr_a), TensorData::Float64(arr_b)) => {
            if let (Some(slice_a), Some(slice_b)) = (arr_a.as_slice(), arr_b.as_slice()) {
                if numel >= 10_000 {
                    let result: Vec<f64> = slice_a
                        .par_iter()
                        .zip(slice_b.par_iter())
                        .map(|(a, b)| a + b)
                        .collect();
                    TensorData::Float64(
                        ndarray::Array::from_shape_vec(arr_a.raw_dim(), result).unwrap(),
                    )
                } else {
                    let result: Vec<f64> = slice_a
                        .iter()
                        .zip(slice_b.iter())
                        .map(|(a, b)| a + b)
                        .collect();
                    TensorData::Float64(
                        ndarray::Array::from_shape_vec(arr_a.raw_dim(), result).unwrap(),
                    )
                }
            } else {
                TensorData::Float64(arr_a + arr_b)
            }
        }
        (TensorData::Int32(arr_a), TensorData::Int32(arr_b)) => TensorData::Int32(arr_a + arr_b),
        (TensorData::Int64(arr_a), TensorData::Int64(arr_b)) => TensorData::Int64(arr_a + arr_b),
        _ => unreachable!("dtype mismatch already checked"),
    };

    Ok(Tensor::from_data(data, dtype))
}

/// SIMD-optimized element-wise multiplication
pub fn mul_simd(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.dtype() != b.dtype() {
        return Err(TensorError::DTypeMismatch {
            expected: format!("{:?}", a.dtype()),
            actual: format!("{:?}", b.dtype()),
            context: "mul_simd".to_string(),
        });
    }
    if a.shape() != b.shape() {
        return Err(TensorError::ShapeMismatch {
            expected: a.shape().to_vec(),
            actual: b.shape().to_vec(),
            context: "mul_simd".to_string(),
        });
    }

    let dtype = a.dtype();
    let numel = a.numel();

    let data = match (a.data(), b.data()) {
        (TensorData::Float32(arr_a), TensorData::Float32(arr_b)) => {
            if let (Some(slice_a), Some(slice_b)) = (arr_a.as_slice(), arr_b.as_slice()) {
                if numel >= 10_000 {
                    let result: Vec<f32> = slice_a
                        .par_iter()
                        .zip(slice_b.par_iter())
                        .map(|(a, b)| a * b)
                        .collect();
                    TensorData::Float32(
                        ndarray::Array::from_shape_vec(arr_a.raw_dim(), result).unwrap(),
                    )
                } else {
                    let result: Vec<f32> = slice_a
                        .iter()
                        .zip(slice_b.iter())
                        .map(|(a, b)| a * b)
                        .collect();
                    TensorData::Float32(
                        ndarray::Array::from_shape_vec(arr_a.raw_dim(), result).unwrap(),
                    )
                }
            } else {
                TensorData::Float32(arr_a * arr_b)
            }
        }
        (TensorData::Float64(arr_a), TensorData::Float64(arr_b)) => {
            if let (Some(slice_a), Some(slice_b)) = (arr_a.as_slice(), arr_b.as_slice()) {
                if numel >= 10_000 {
                    let result: Vec<f64> = slice_a
                        .par_iter()
                        .zip(slice_b.par_iter())
                        .map(|(a, b)| a * b)
                        .collect();
                    TensorData::Float64(
                        ndarray::Array::from_shape_vec(arr_a.raw_dim(), result).unwrap(),
                    )
                } else {
                    let result: Vec<f64> = slice_a
                        .iter()
                        .zip(slice_b.iter())
                        .map(|(a, b)| a * b)
                        .collect();
                    TensorData::Float64(
                        ndarray::Array::from_shape_vec(arr_a.raw_dim(), result).unwrap(),
                    )
                }
            } else {
                TensorData::Float64(arr_a * arr_b)
            }
        }
        (TensorData::Int32(arr_a), TensorData::Int32(arr_b)) => TensorData::Int32(arr_a * arr_b),
        (TensorData::Int64(arr_a), TensorData::Int64(arr_b)) => TensorData::Int64(arr_a * arr_b),
        _ => unreachable!("dtype mismatch already checked"),
    };

    Ok(Tensor::from_data(data, dtype))
}

/// SIMD-optimized ReLU activation
///
/// Uses vectorized max operation for better performance
pub fn relu_simd(tensor: &Tensor) -> Tensor {
    let dtype = tensor.dtype();
    let numel = tensor.numel();

    let data = match tensor.data() {
        TensorData::Float32(arr) => {
            if let Some(slice) = arr.as_slice() {
                if numel >= 10_000 {
                    // Parallel vectorized ReLU
                    let result: Vec<f32> = slice.par_iter().map(|&x| x.max(0.0)).collect();
                    TensorData::Float32(
                        ndarray::Array::from_shape_vec(arr.raw_dim(), result).unwrap(),
                    )
                } else {
                    // Sequential vectorized ReLU (auto-vectorized)
                    let result: Vec<f32> = slice.iter().map(|&x| x.max(0.0)).collect();
                    TensorData::Float32(
                        ndarray::Array::from_shape_vec(arr.raw_dim(), result).unwrap(),
                    )
                }
            } else {
                TensorData::Float32(arr.mapv(|x| x.max(0.0)))
            }
        }
        TensorData::Float64(arr) => {
            if let Some(slice) = arr.as_slice() {
                if numel >= 10_000 {
                    let result: Vec<f64> = slice.par_iter().map(|&x| x.max(0.0)).collect();
                    TensorData::Float64(
                        ndarray::Array::from_shape_vec(arr.raw_dim(), result).unwrap(),
                    )
                } else {
                    let result: Vec<f64> = slice.iter().map(|&x| x.max(0.0)).collect();
                    TensorData::Float64(
                        ndarray::Array::from_shape_vec(arr.raw_dim(), result).unwrap(),
                    )
                }
            } else {
                TensorData::Float64(arr.mapv(|x| x.max(0.0)))
            }
        }
        TensorData::Int32(arr) => TensorData::Int32(arr.mapv(|x| x.max(0))),
        TensorData::Int64(arr) => TensorData::Int64(arr.mapv(|x| x.max(0))),
    };

    Tensor::from_data(data, dtype)
}

/// SIMD-optimized scalar multiplication
pub fn mul_scalar_simd(tensor: &Tensor, scalar: f32) -> Tensor {
    let dtype = tensor.dtype();
    let numel = tensor.numel();

    let data = match tensor.data() {
        TensorData::Float32(arr) => {
            if let Some(slice) = arr.as_slice() {
                if numel >= 10_000 {
                    let result: Vec<f32> = slice.par_iter().map(|&x| x * scalar).collect();
                    TensorData::Float32(
                        ndarray::Array::from_shape_vec(arr.raw_dim(), result).unwrap(),
                    )
                } else {
                    let result: Vec<f32> = slice.iter().map(|&x| x * scalar).collect();
                    TensorData::Float32(
                        ndarray::Array::from_shape_vec(arr.raw_dim(), result).unwrap(),
                    )
                }
            } else {
                TensorData::Float32(arr * scalar)
            }
        }
        TensorData::Float64(arr) => {
            let scalar_f64 = scalar as f64;
            if let Some(slice) = arr.as_slice() {
                if numel >= 10_000 {
                    let result: Vec<f64> = slice.par_iter().map(|&x| x * scalar_f64).collect();
                    TensorData::Float64(
                        ndarray::Array::from_shape_vec(arr.raw_dim(), result).unwrap(),
                    )
                } else {
                    let result: Vec<f64> = slice.iter().map(|&x| x * scalar_f64).collect();
                    TensorData::Float64(
                        ndarray::Array::from_shape_vec(arr.raw_dim(), result).unwrap(),
                    )
                }
            } else {
                TensorData::Float64(arr * scalar_f64)
            }
        }
        TensorData::Int32(arr) => TensorData::Int32(arr * scalar as i32),
        TensorData::Int64(arr) => TensorData::Int64(arr * scalar as i64),
    };

    Tensor::from_data(data, dtype)
}

/// Fused multiply-add: a * b + c (SIMD optimized)
///
/// More efficient than separate mul and add operations
pub fn fused_multiply_add(a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> {
    if a.shape() != b.shape() || a.shape() != c.shape() {
        return Err(TensorError::ShapeMismatch {
            expected: a.shape().to_vec(),
            actual: if a.shape() != b.shape() { b.shape().to_vec() } else { c.shape().to_vec() },
            context: "fused_multiply_add".to_string(),
        });
    }
    if a.dtype() != b.dtype() || a.dtype() != c.dtype() {
        return Err(TensorError::DTypeMismatch {
            expected: format!("{:?}", a.dtype()),
            actual: if a.dtype() != b.dtype() { format!("{:?}", b.dtype()) } else { format!("{:?}", c.dtype()) },
            context: "fused_multiply_add".to_string(),
        });
    }

    let dtype = a.dtype();
    let numel = a.numel();

    let data = match (a.data(), b.data(), c.data()) {
        (TensorData::Float32(arr_a), TensorData::Float32(arr_b), TensorData::Float32(arr_c)) => {
            if let (Some(slice_a), Some(slice_b), Some(slice_c)) =
                (arr_a.as_slice(), arr_b.as_slice(), arr_c.as_slice())
            {
                if numel >= 10_000 {
                    let result: Vec<f32> = slice_a
                        .par_iter()
                        .zip(slice_b.par_iter())
                        .zip(slice_c.par_iter())
                        .map(|((&a, &b), &c)| a * b + c)
                        .collect();
                    TensorData::Float32(
                        ndarray::Array::from_shape_vec(arr_a.raw_dim(), result).unwrap(),
                    )
                } else {
                    let result: Vec<f32> = slice_a
                        .iter()
                        .zip(slice_b.iter())
                        .zip(slice_c.iter())
                        .map(|((&a, &b), &c)| a * b + c)
                        .collect();
                    TensorData::Float32(
                        ndarray::Array::from_shape_vec(arr_a.raw_dim(), result).unwrap(),
                    )
                }
            } else {
                TensorData::Float32(arr_a * arr_b + arr_c)
            }
        }
        (TensorData::Float64(arr_a), TensorData::Float64(arr_b), TensorData::Float64(arr_c)) => {
            if let (Some(slice_a), Some(slice_b), Some(slice_c)) =
                (arr_a.as_slice(), arr_b.as_slice(), arr_c.as_slice())
            {
                if numel >= 10_000 {
                    let result: Vec<f64> = slice_a
                        .par_iter()
                        .zip(slice_b.par_iter())
                        .zip(slice_c.par_iter())
                        .map(|((&a, &b), &c)| a * b + c)
                        .collect();
                    TensorData::Float64(
                        ndarray::Array::from_shape_vec(arr_a.raw_dim(), result).unwrap(),
                    )
                } else {
                    let result: Vec<f64> = slice_a
                        .iter()
                        .zip(slice_b.iter())
                        .zip(slice_c.iter())
                        .map(|((&a, &b), &c)| a * b + c)
                        .collect();
                    TensorData::Float64(
                        ndarray::Array::from_shape_vec(arr_a.raw_dim(), result).unwrap(),
                    )
                }
            } else {
                TensorData::Float64(arr_a * arr_b + arr_c)
            }
        }
        _ => {
            return Err(TensorError::InvalidArgument {
                parameter: "dtype".to_string(),
                reason: "fused_multiply_add only supports floating-point tensors".to_string(),
            })
        }
    };

    Ok(Tensor::from_data(data, dtype))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DType;

    #[test]
    fn test_add_simd() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]);
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[4]);
        let c = add_simd(&a, &b).unwrap();
        assert_eq!(c.shape(), &[4]);
    }

    #[test]
    fn test_mul_simd() {
        let a = Tensor::from_vec(vec![2.0, 3.0, 4.0, 5.0], &[4]);
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]);
        let c = mul_simd(&a, &b).unwrap();
        assert_eq!(c.shape(), &[4]);
    }

    #[test]
    fn test_relu_simd() {
        let t = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]);
        let r = relu_simd(&t);
        assert_eq!(r.shape(), &[5]);
    }

    #[test]
    fn test_mul_scalar_simd() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]);
        let r = mul_scalar_simd(&t, 2.0);
        assert_eq!(r.shape(), &[4]);
    }

    #[test]
    fn test_fused_multiply_add() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
        let b = Tensor::from_vec(vec![2.0, 3.0, 4.0], &[3]);
        let c = Tensor::from_vec(vec![1.0, 1.0, 1.0], &[3]);
        let result = fused_multiply_add(&a, &b, &c).unwrap();
        // Expected: [1*2+1, 2*3+1, 3*4+1] = [3, 7, 13]
        assert_eq!(result.shape(), &[3]);
    }

    #[test]
    fn test_simd_large_tensor() {
        // Test parallel execution path
        let size = 100_000;
        let a = Tensor::ones(&[size], DType::Float32);
        let b = Tensor::ones(&[size], DType::Float32);
        let c = add_simd(&a, &b).unwrap();
        assert_eq!(c.numel(), size);
    }

    #[test]
    fn test_add_simd_shape_mismatch() {
        let a = Tensor::from_vec(vec![1.0, 2.0], &[2]);
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
        assert!(add_simd(&a, &b).is_err());
    }

    #[test]
    fn test_add_simd_dtype_mismatch() {
        let a = Tensor::ones(&[4], DType::Float32);
        let b = Tensor::ones(&[4], DType::Float64);
        assert!(add_simd(&a, &b).is_err());
    }

    #[test]
    fn test_mul_simd_shape_mismatch() {
        let a = Tensor::from_vec(vec![1.0, 2.0], &[2]);
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
        assert!(mul_simd(&a, &b).is_err());
    }

    #[test]
    fn test_fma_shape_mismatch() {
        let a = Tensor::from_vec(vec![1.0, 2.0], &[2]);
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
        let c = Tensor::from_vec(vec![1.0, 2.0], &[2]);
        assert!(fused_multiply_add(&a, &b, &c).is_err());
    }
}
