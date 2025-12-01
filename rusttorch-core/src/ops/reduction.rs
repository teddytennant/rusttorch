//! Reduction operations (sum, mean, max, min)

use crate::tensor::{Tensor, TensorData, DType};
use crate::error::{Result, TensorError};

/// Sum all elements in a tensor
pub fn sum(tensor: &Tensor) -> f64 {
    match tensor.data() {
        TensorData::Float32(arr) => arr.iter().map(|&x| x as f64).sum(),
        TensorData::Float64(arr) => arr.iter().copied().sum(),
        TensorData::Int32(arr) => arr.iter().map(|&x| x as f64).sum(),
        TensorData::Int64(arr) => arr.iter().map(|&x| x as f64).sum(),
    }
}

/// Compute the mean of all elements
pub fn mean(tensor: &Tensor) -> f64 {
    let total = sum(tensor);
    let count = tensor.numel() as f64;
    total / count
}

/// Find the maximum element
pub fn max(tensor: &Tensor) -> f64 {
    match tensor.data() {
        TensorData::Float32(arr) => arr
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(f32::NEG_INFINITY) as f64,
        TensorData::Float64(arr) => arr
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(f64::NEG_INFINITY),
        TensorData::Int32(arr) => arr.iter().copied().max().unwrap_or(i32::MIN) as f64,
        TensorData::Int64(arr) => arr.iter().copied().max().unwrap_or(i64::MIN) as f64,
    }
}

/// Find the minimum element
pub fn min(tensor: &Tensor) -> f64 {
    match tensor.data() {
        TensorData::Float32(arr) => arr
            .iter()
            .copied()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(f32::INFINITY) as f64,
        TensorData::Float64(arr) => arr
            .iter()
            .copied()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(f64::INFINITY),
        TensorData::Int32(arr) => arr.iter().copied().min().unwrap_or(i32::MAX) as f64,
        TensorData::Int64(arr) => arr.iter().copied().min().unwrap_or(i64::MAX) as f64,
    }
}

/// Sum along a specific dimension
pub fn sum_dim(tensor: &Tensor, dim: usize) -> Result<Tensor> {
    if dim >= tensor.ndim() {
        return Err(TensorError::InvalidDimension {
            dimension: dim,
            max_dimension: tensor.ndim() - 1,
            context: "sum_dim".to_string(),
        });
    }

    match tensor.data() {
        TensorData::Float32(arr) => {
            let result = arr.sum_axis(ndarray::Axis(dim));
            Ok(Tensor::from_data(TensorData::Float32(result), DType::Float32))
        }
        TensorData::Float64(arr) => {
            let result = arr.sum_axis(ndarray::Axis(dim));
            Ok(Tensor::from_data(TensorData::Float64(result), DType::Float64))
        }
        TensorData::Int32(arr) => {
            let result = arr.sum_axis(ndarray::Axis(dim));
            Ok(Tensor::from_data(TensorData::Int32(result), DType::Int32))
        }
        TensorData::Int64(arr) => {
            let result = arr.sum_axis(ndarray::Axis(dim));
            Ok(Tensor::from_data(TensorData::Int64(result), DType::Int64))
        }
    }
}

/// Mean along a specific dimension
pub fn mean_dim(tensor: &Tensor, dim: usize) -> Result<Tensor> {
    if dim >= tensor.ndim() {
        return Err(TensorError::InvalidDimension {
            dimension: dim,
            max_dimension: tensor.ndim() - 1,
            context: "mean_dim".to_string(),
        });
    }

    match tensor.data() {
        TensorData::Float32(arr) => {
            let result = arr.mean_axis(ndarray::Axis(dim)).unwrap();
            Ok(Tensor::from_data(TensorData::Float32(result), DType::Float32))
        }
        TensorData::Float64(arr) => {
            let result = arr.mean_axis(ndarray::Axis(dim)).unwrap();
            Ok(Tensor::from_data(TensorData::Float64(result), DType::Float64))
        }
        TensorData::Int32(arr) => {
            // For integers, compute sum then divide
            let sum_result = arr.sum_axis(ndarray::Axis(dim));
            let size = arr.len_of(ndarray::Axis(dim)) as i32;
            let result = sum_result / size;
            Ok(Tensor::from_data(TensorData::Int32(result), DType::Int32))
        }
        TensorData::Int64(arr) => {
            // For integers, compute sum then divide
            let sum_result = arr.sum_axis(ndarray::Axis(dim));
            let size = arr.len_of(ndarray::Axis(dim)) as i64;
            let result = sum_result / size;
            Ok(Tensor::from_data(TensorData::Int64(result), DType::Int64))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let s = sum(&t);
        assert_eq!(s, 10.0);
    }

    #[test]
    fn test_mean() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let m = mean(&t);
        assert_eq!(m, 2.5);
    }

    #[test]
    fn test_max() {
        let t = Tensor::from_vec(vec![1.0, 5.0, 3.0, 2.0], &[2, 2]);
        let m = max(&t);
        assert_eq!(m, 5.0);
    }

    #[test]
    fn test_min() {
        let t = Tensor::from_vec(vec![3.0, 1.0, 5.0, 2.0], &[2, 2]);
        let m = min(&t);
        assert_eq!(m, 1.0);
    }

    #[test]
    fn test_sum_dim() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let s = sum_dim(&t, 0).unwrap();
        assert_eq!(s.shape(), &[3]);
    }

    #[test]
    fn test_mean_dim() {
        let t = Tensor::from_vec(vec![2.0, 4.0, 6.0, 8.0], &[2, 2]);
        let m = mean_dim(&t, 1).unwrap();
        assert_eq!(m.shape(), &[2]);
    }

    #[test]
    fn test_sum_dim_out_of_range() {
        let t = Tensor::ones(&[2, 2], DType::Float32);
        let result = sum_dim(&t, 5);
        assert!(result.is_err());
        match result.unwrap_err() {
            TensorError::InvalidDimension { .. } => {}
            _ => panic!("Expected InvalidDimension error"),
        }
    }
}
