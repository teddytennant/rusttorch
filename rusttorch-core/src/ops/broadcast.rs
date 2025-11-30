//! Broadcasting utilities for tensor operations
//!
//! Broadcasting allows operations between tensors of different shapes
//! by automatically expanding them to compatible shapes.
//!
//! Broadcasting rules (NumPy/PyTorch compatible):
//! 1. If tensors have different number of dimensions, prepend 1s to the smaller shape
//! 2. Dimensions are compatible if they are equal or one of them is 1
//! 3. The result shape is the maximum along each dimension

use crate::error::{TensorError, Result};
use crate::tensor::{Tensor, TensorData};
use crate::DType;
use ndarray::{Array, IxDyn};

/// Check if two shapes are broadcastable
pub fn shapes_broadcastable(shape_a: &[usize], shape_b: &[usize]) -> bool {
    let max_ndim = shape_a.len().max(shape_b.len());

    for i in 0..max_ndim {
        let dim_a = shape_a.iter().rev().nth(i).copied().unwrap_or(1);
        let dim_b = shape_b.iter().rev().nth(i).copied().unwrap_or(1);

        if dim_a != dim_b && dim_a != 1 && dim_b != 1 {
            return false;
        }
    }

    true
}

/// Compute the broadcast shape of two shapes
pub fn broadcast_shape(shape_a: &[usize], shape_b: &[usize]) -> Result<Vec<usize>> {
    if !shapes_broadcastable(shape_a, shape_b) {
        return Err(TensorError::BroadcastError {
            shape_a: shape_a.to_vec(),
            shape_b: shape_b.to_vec(),
            reason: "Incompatible shapes for broadcasting".to_string(),
        });
    }

    let max_ndim = shape_a.len().max(shape_b.len());
    let mut result_shape = Vec::with_capacity(max_ndim);

    for i in 0..max_ndim {
        let dim_a = shape_a.iter().rev().nth(i).copied().unwrap_or(1);
        let dim_b = shape_b.iter().rev().nth(i).copied().unwrap_or(1);
        result_shape.push(dim_a.max(dim_b));
    }

    result_shape.reverse();
    Ok(result_shape)
}

/// Broadcast two tensors to compatible shapes
pub fn broadcast_tensors(a: &Tensor, b: &Tensor) -> Result<(Tensor, Tensor)> {
    // Validate dtypes match
    if a.dtype() != b.dtype() {
        return Err(TensorError::DTypeMismatch {
            expected: format!("{}", a.dtype()),
            actual: format!("{}", b.dtype()),
            context: "broadcast_tensors".to_string(),
        });
    }

    let shape_a = a.shape();
    let shape_b = b.shape();

    // If shapes are already equal, return clones
    if shape_a == shape_b {
        return Ok((a.clone(), b.clone()));
    }

    let target_shape = broadcast_shape(shape_a, shape_b)?;

    let dtype = a.dtype();

    // Broadcast arrays
    let (broadcasted_a, broadcasted_b) = match (a.data(), b.data()) {
        (TensorData::Float32(arr_a), TensorData::Float32(arr_b)) => {
            let bc_a = broadcast_array_f32(arr_a, &target_shape)?;
            let bc_b = broadcast_array_f32(arr_b, &target_shape)?;
            (
                Tensor::from_data(TensorData::Float32(bc_a), dtype),
                Tensor::from_data(TensorData::Float32(bc_b), dtype),
            )
        }
        (TensorData::Float64(arr_a), TensorData::Float64(arr_b)) => {
            let bc_a = broadcast_array_f64(arr_a, &target_shape)?;
            let bc_b = broadcast_array_f64(arr_b, &target_shape)?;
            (
                Tensor::from_data(TensorData::Float64(bc_a), dtype),
                Tensor::from_data(TensorData::Float64(bc_b), dtype),
            )
        }
        (TensorData::Int32(arr_a), TensorData::Int32(arr_b)) => {
            let bc_a = broadcast_array_i32(arr_a, &target_shape)?;
            let bc_b = broadcast_array_i32(arr_b, &target_shape)?;
            (
                Tensor::from_data(TensorData::Int32(bc_a), dtype),
                Tensor::from_data(TensorData::Int32(bc_b), dtype),
            )
        }
        (TensorData::Int64(arr_a), TensorData::Int64(arr_b)) => {
            let bc_a = broadcast_array_i64(arr_a, &target_shape)?;
            let bc_b = broadcast_array_i64(arr_b, &target_shape)?;
            (
                Tensor::from_data(TensorData::Int64(bc_a), dtype),
                Tensor::from_data(TensorData::Int64(bc_b), dtype),
            )
        }
        _ => {
            return Err(TensorError::DTypeMismatch {
                expected: format!("{}", a.dtype()),
                actual: format!("{}", b.dtype()),
                context: "broadcast_tensors (data type mismatch)".to_string(),
            })
        }
    };

    Ok((broadcasted_a, broadcasted_b))
}

// Helper functions to broadcast arrays
fn broadcast_array_f32(arr: &Array<f32, IxDyn>, target_shape: &[usize]) -> Result<Array<f32, IxDyn>> {
    arr.broadcast(IxDyn(target_shape))
        .map(|broadcasted| broadcasted.to_owned())
        .map_err(|e| TensorError::BroadcastError {
            shape_a: arr.shape().to_vec(),
            shape_b: target_shape.to_vec(),
            reason: format!("ndarray broadcast failed: {}", e),
        })
}

fn broadcast_array_f64(arr: &Array<f64, IxDyn>, target_shape: &[usize]) -> Result<Array<f64, IxDyn>> {
    arr.broadcast(IxDyn(target_shape))
        .map(|broadcasted| broadcasted.to_owned())
        .map_err(|e| TensorError::BroadcastError {
            shape_a: arr.shape().to_vec(),
            shape_b: target_shape.to_vec(),
            reason: format!("ndarray broadcast failed: {}", e),
        })
}

fn broadcast_array_i32(arr: &Array<i32, IxDyn>, target_shape: &[usize]) -> Result<Array<i32, IxDyn>> {
    arr.broadcast(IxDyn(target_shape))
        .map(|broadcasted| broadcasted.to_owned())
        .map_err(|e| TensorError::BroadcastError {
            shape_a: arr.shape().to_vec(),
            shape_b: target_shape.to_vec(),
            reason: format!("ndarray broadcast failed: {}", e),
        })
}

fn broadcast_array_i64(arr: &Array<i64, IxDyn>, target_shape: &[usize]) -> Result<Array<i64, IxDyn>> {
    arr.broadcast(IxDyn(target_shape))
        .map(|broadcasted| broadcasted.to_owned())
        .map_err(|e| TensorError::BroadcastError {
            shape_a: arr.shape().to_vec(),
            shape_b: target_shape.to_vec(),
            reason: format!("ndarray broadcast failed: {}", e),
        })
}

/// Element-wise addition with broadcasting
pub fn add_broadcast(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let (bc_a, bc_b) = broadcast_tensors(a, b)?;
    Ok(crate::ops::add(&bc_a, &bc_b))
}

/// Element-wise multiplication with broadcasting
pub fn mul_broadcast(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let (bc_a, bc_b) = broadcast_tensors(a, b)?;
    Ok(crate::ops::mul(&bc_a, &bc_b))
}

/// Element-wise subtraction with broadcasting
pub fn sub_broadcast(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let (bc_a, bc_b) = broadcast_tensors(a, b)?;
    Ok(crate::ops::sub(&bc_a, &bc_b))
}

/// Element-wise division with broadcasting
pub fn div_broadcast(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let (bc_a, bc_b) = broadcast_tensors(a, b)?;
    Ok(crate::ops::div(&bc_a, &bc_b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shapes_broadcastable() {
        // Compatible shapes
        assert!(shapes_broadcastable(&[3, 4], &[3, 4])); // Same shape
        assert!(shapes_broadcastable(&[3, 4], &[1, 4])); // Leading 1
        assert!(shapes_broadcastable(&[3, 4], &[3, 1])); // Trailing 1
        assert!(shapes_broadcastable(&[3, 4], &[1, 1])); // All 1s
        assert!(shapes_broadcastable(&[3, 4, 5], &[4, 5])); // Different ndim

        // Incompatible shapes
        assert!(!shapes_broadcastable(&[3, 4], &[3, 5])); // Different sizes
        assert!(!shapes_broadcastable(&[2, 3], &[3, 2])); // Swapped dimensions
    }

    #[test]
    fn test_broadcast_shape() {
        assert_eq!(
            broadcast_shape(&[3, 4], &[3, 4]).unwrap(),
            vec![3, 4]
        );
        assert_eq!(
            broadcast_shape(&[3, 4], &[1, 4]).unwrap(),
            vec![3, 4]
        );
        assert_eq!(
            broadcast_shape(&[3, 4], &[4]).unwrap(),
            vec![3, 4]
        );
        assert_eq!(
            broadcast_shape(&[3, 4, 5], &[4, 5]).unwrap(),
            vec![3, 4, 5]
        );
        assert_eq!(
            broadcast_shape(&[1, 3, 1], &[2, 1, 4]).unwrap(),
            vec![2, 3, 4]
        );
    }

    #[test]
    fn test_broadcast_shape_error() {
        assert!(broadcast_shape(&[3, 4], &[3, 5]).is_err());
        assert!(broadcast_shape(&[2, 3], &[3, 2]).is_err());
    }

    #[test]
    fn test_add_broadcast() {
        // [3, 4] + [1, 4] -> [3, 4]
        let a = Tensor::ones(&[3, 4], DType::Float32);
        let b = Tensor::ones(&[1, 4], DType::Float32);
        let c = add_broadcast(&a, &b).unwrap();
        assert_eq!(c.shape(), &[3, 4]);
    }

    #[test]
    fn test_mul_broadcast() {
        // [3, 4] * [4] -> [3, 4]
        let a = Tensor::from_vec(vec![1.0; 12], &[3, 4]);
        let b = Tensor::from_vec(vec![2.0; 4], &[4]);
        let c = mul_broadcast(&a, &b).unwrap();
        assert_eq!(c.shape(), &[3, 4]);
    }

    #[test]
    fn test_broadcast_scalar_like() {
        // [3, 4] + [1, 1] -> [3, 4]
        let a = Tensor::ones(&[3, 4], DType::Float32);
        let b = Tensor::from_vec(vec![5.0], &[1, 1]);
        let c = add_broadcast(&a, &b).unwrap();
        assert_eq!(c.shape(), &[3, 4]);
    }

    #[test]
    fn test_broadcast_different_ndim() {
        // [2, 3, 4] + [3, 4] -> [2, 3, 4]
        let a = Tensor::ones(&[2, 3, 4], DType::Float32);
        let b = Tensor::ones(&[3, 4], DType::Float32);
        let c = add_broadcast(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_broadcast_error_incompatible() {
        let a = Tensor::ones(&[3, 4], DType::Float32);
        let b = Tensor::ones(&[3, 5], DType::Float32);
        assert!(add_broadcast(&a, &b).is_err());
    }

    #[test]
    fn test_broadcast_dtype_mismatch() {
        let a = Tensor::ones(&[3, 4], DType::Float32);
        let b = Tensor::ones(&[3, 4], DType::Float64);
        assert!(add_broadcast(&a, &b).is_err());
    }
}
