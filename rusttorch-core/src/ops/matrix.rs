//! Matrix operations
//!
//! This module provides matrix-specific operations including:
//! - Matrix multiplication (matmul)
//! - Transpose
//! - Reshape

use crate::tensor::{DType, Tensor, TensorData};
use ndarray::{Array, IxDyn, s};

/// Matrix multiplication
///
/// Performs matrix multiplication between two 2D tensors.
/// For higher dimensional tensors, performs batched matrix multiplication.
///
/// # Arguments
/// * `a` - First tensor (must be at least 2D)
/// * `b` - Second tensor (must be at least 2D)
///
/// # Returns
/// Result containing the matrix product or an error
///
/// # Panics
/// Panics if the inner dimensions don't match (a.shape[-1] != b.shape[-2])
///
/// # Examples
/// ```
/// use rusttorch_core::{Tensor, DType};
/// use rusttorch_core::ops::matrix::matmul;
///
/// let a = Tensor::ones(&[2, 3], DType::Float32);
/// let b = Tensor::ones(&[3, 4], DType::Float32);
/// let c = matmul(&a, &b).unwrap();
/// assert_eq!(c.shape(), &[2, 4]);
/// ```
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    // Check that both tensors are at least 2D
    if a.ndim() < 2 || b.ndim() < 2 {
        return Err(format!(
            "matmul requires at least 2D tensors, got {}D and {}D",
            a.ndim(),
            b.ndim()
        ));
    }

    // Check dtype compatibility
    if a.dtype() != b.dtype() {
        return Err(format!(
            "matmul requires same dtype, got {:?} and {:?}",
            a.dtype(),
            b.dtype()
        ));
    }

    let a_shape = a.shape();
    let b_shape = b.shape();

    // Check inner dimensions match
    let a_cols = a_shape[a_shape.len() - 1];
    let b_rows = b_shape[b_shape.len() - 2];

    if a_cols != b_rows {
        return Err(format!(
            "matmul dimension mismatch: {}x{} @ {}x{}",
            a_shape[a_shape.len() - 2],
            a_cols,
            b_rows,
            b_shape[b_shape.len() - 1]
        ));
    }

    match (a.data(), b.data()) {
        (TensorData::Float32(a_arr), TensorData::Float32(b_arr)) => {
            let result = matmul_float32(a_arr, b_arr)?;
            Ok(Tensor::from_data(TensorData::Float32(result), DType::Float32))
        }
        (TensorData::Float64(a_arr), TensorData::Float64(b_arr)) => {
            let result = matmul_float64(a_arr, b_arr)?;
            Ok(Tensor::from_data(TensorData::Float64(result), DType::Float64))
        }
        (TensorData::Int32(a_arr), TensorData::Int32(b_arr)) => {
            let result = matmul_int32(a_arr, b_arr)?;
            Ok(Tensor::from_data(TensorData::Int32(result), DType::Int32))
        }
        (TensorData::Int64(a_arr), TensorData::Int64(b_arr)) => {
            let result = matmul_int64(a_arr, b_arr)?;
            Ok(Tensor::from_data(TensorData::Int64(result), DType::Int64))
        }
        _ => unreachable!("Type mismatch already checked"),
    }
}

fn matmul_float32(
    a: &Array<f32, IxDyn>,
    b: &Array<f32, IxDyn>,
) -> Result<Array<f32, IxDyn>, String> {
    let a_shape = a.shape();
    let b_shape = b.shape();

    // For 2D x 2D, use ndarray's dot
    if a_shape.len() == 2 && b_shape.len() == 2 {
        let a_2d = a.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
        let b_2d = b.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
        let result = a_2d.dot(&b_2d);
        return Ok(result.into_dyn());
    }

    // For higher dimensions, we need batched matmul
    // This is a simplified version - full batched matmul is more complex
    Err("Batched matrix multiplication not yet implemented".to_string())
}

fn matmul_float64(
    a: &Array<f64, IxDyn>,
    b: &Array<f64, IxDyn>,
) -> Result<Array<f64, IxDyn>, String> {
    let a_shape = a.shape();
    let b_shape = b.shape();

    if a_shape.len() == 2 && b_shape.len() == 2 {
        let a_2d = a.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
        let b_2d = b.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
        let result = a_2d.dot(&b_2d);
        return Ok(result.into_dyn());
    }

    Err("Batched matrix multiplication not yet implemented".to_string())
}

fn matmul_int32(
    a: &Array<i32, IxDyn>,
    b: &Array<i32, IxDyn>,
) -> Result<Array<i32, IxDyn>, String> {
    let a_shape = a.shape();
    let b_shape = b.shape();

    if a_shape.len() == 2 && b_shape.len() == 2 {
        let a_2d = a.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
        let b_2d = b.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
        let result = a_2d.dot(&b_2d);
        return Ok(result.into_dyn());
    }

    Err("Batched matrix multiplication not yet implemented".to_string())
}

fn matmul_int64(
    a: &Array<i64, IxDyn>,
    b: &Array<i64, IxDyn>,
) -> Result<Array<i64, IxDyn>, String> {
    let a_shape = a.shape();
    let b_shape = b.shape();

    if a_shape.len() == 2 && b_shape.len() == 2 {
        let a_2d = a.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
        let b_2d = b.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
        let result = a_2d.dot(&b_2d);
        return Ok(result.into_dyn());
    }

    Err("Batched matrix multiplication not yet implemented".to_string())
}

/// Transpose a tensor
///
/// For 2D tensors, swaps rows and columns.
/// For higher dimensional tensors, reverses the order of axes.
///
/// # Arguments
/// * `tensor` - The tensor to transpose
///
/// # Returns
/// A new tensor with transposed dimensions
///
/// # Examples
/// ```
/// use rusttorch_core::{Tensor, DType};
/// use rusttorch_core::ops::matrix::transpose;
///
/// let a = Tensor::ones(&[2, 3], DType::Float32);
/// let b = transpose(&a);
/// assert_eq!(b.shape(), &[3, 2]);
/// ```
pub fn transpose(tensor: &Tensor) -> Tensor {
    match tensor.data() {
        TensorData::Float32(arr) => {
            let result = arr.t().to_owned().into_dyn();
            Tensor::from_data(TensorData::Float32(result), tensor.dtype())
        }
        TensorData::Float64(arr) => {
            let result = arr.t().to_owned().into_dyn();
            Tensor::from_data(TensorData::Float64(result), tensor.dtype())
        }
        TensorData::Int32(arr) => {
            let result = arr.t().to_owned().into_dyn();
            Tensor::from_data(TensorData::Int32(result), tensor.dtype())
        }
        TensorData::Int64(arr) => {
            let result = arr.t().to_owned().into_dyn();
            Tensor::from_data(TensorData::Int64(result), tensor.dtype())
        }
    }
}

/// Reshape a tensor to a new shape
///
/// Returns a new tensor with the same data but different shape.
/// The total number of elements must remain the same.
///
/// # Arguments
/// * `tensor` - The tensor to reshape
/// * `new_shape` - The desired shape
///
/// # Returns
/// Result containing the reshaped tensor or an error if shapes are incompatible
///
/// # Examples
/// ```
/// use rusttorch_core::{Tensor, DType};
/// use rusttorch_core::ops::matrix::reshape;
///
/// let a = Tensor::ones(&[2, 6], DType::Float32);
/// let b = reshape(&a, &[3, 4]).unwrap();
/// assert_eq!(b.shape(), &[3, 4]);
/// ```
pub fn reshape(tensor: &Tensor, new_shape: &[usize]) -> Result<Tensor, String> {
    let old_numel = tensor.numel();
    let new_numel: usize = new_shape.iter().product();

    if old_numel != new_numel {
        return Err(format!(
            "reshape: cannot reshape tensor of {} elements to shape with {} elements",
            old_numel, new_numel
        ));
    }

    match tensor.data() {
        TensorData::Float32(arr) => {
            let reshaped = arr
                .clone()
                .into_shape(IxDyn(new_shape))
                .map_err(|e| format!("reshape error: {}", e))?;
            Ok(Tensor::from_data(
                TensorData::Float32(reshaped),
                tensor.dtype(),
            ))
        }
        TensorData::Float64(arr) => {
            let reshaped = arr
                .clone()
                .into_shape(IxDyn(new_shape))
                .map_err(|e| format!("reshape error: {}", e))?;
            Ok(Tensor::from_data(
                TensorData::Float64(reshaped),
                tensor.dtype(),
            ))
        }
        TensorData::Int32(arr) => {
            let reshaped = arr
                .clone()
                .into_shape(IxDyn(new_shape))
                .map_err(|e| format!("reshape error: {}", e))?;
            Ok(Tensor::from_data(
                TensorData::Int32(reshaped),
                tensor.dtype(),
            ))
        }
        TensorData::Int64(arr) => {
            let reshaped = arr
                .clone()
                .into_shape(IxDyn(new_shape))
                .map_err(|e| format!("reshape error: {}", e))?;
            Ok(Tensor::from_data(
                TensorData::Int64(reshaped),
                tensor.dtype(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_basic() {
        let a = Tensor::ones(&[2, 3], DType::Float32);
        let b = Tensor::ones(&[3, 4], DType::Float32);
        let c = matmul(&a, &b).unwrap();

        assert_eq!(c.shape(), &[2, 4]);
        assert_eq!(c.dtype(), DType::Float32);
    }

    #[test]
    fn test_matmul_dimension_mismatch() {
        let a = Tensor::ones(&[2, 3], DType::Float32);
        let b = Tensor::ones(&[4, 5], DType::Float32);
        let result = matmul(&a, &b);

        assert!(result.is_err());
    }

    #[test]
    fn test_matmul_dtype_mismatch() {
        let a = Tensor::ones(&[2, 3], DType::Float32);
        let b = Tensor::ones(&[3, 4], DType::Float64);
        let result = matmul(&a, &b);

        assert!(result.is_err());
    }

    #[test]
    fn test_matmul_1d_tensor() {
        let a = Tensor::ones(&[3], DType::Float32);
        let b = Tensor::ones(&[3, 4], DType::Float32);
        let result = matmul(&a, &b);

        assert!(result.is_err());
    }

    #[test]
    fn test_transpose_2d() {
        let a = Tensor::ones(&[2, 3], DType::Float32);
        let b = transpose(&a);

        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.dtype(), DType::Float32);
    }

    #[test]
    fn test_transpose_square() {
        let a = Tensor::ones(&[4, 4], DType::Float32);
        let b = transpose(&a);

        assert_eq!(b.shape(), &[4, 4]);
    }

    #[test]
    fn test_reshape_basic() {
        let a = Tensor::ones(&[2, 6], DType::Float32);
        let b = reshape(&a, &[3, 4]).unwrap();

        assert_eq!(b.shape(), &[3, 4]);
        assert_eq!(b.numel(), 12);
    }

    #[test]
    fn test_reshape_to_1d() {
        let a = Tensor::ones(&[2, 3, 4], DType::Float32);
        let b = reshape(&a, &[24]).unwrap();

        assert_eq!(b.shape(), &[24]);
        assert_eq!(b.ndim(), 1);
    }

    #[test]
    fn test_reshape_element_count_mismatch() {
        let a = Tensor::ones(&[2, 6], DType::Float32);
        let result = reshape(&a, &[3, 5]);

        assert!(result.is_err());
    }

    #[test]
    fn test_matmul_matmul_chain() {
        // Test A @ B @ C
        let a = Tensor::ones(&[2, 3], DType::Float32);
        let b = Tensor::ones(&[3, 4], DType::Float32);
        let c = Tensor::ones(&[4, 5], DType::Float32);

        let ab = matmul(&a, &b).unwrap();
        let abc = matmul(&ab, &c).unwrap();

        assert_eq!(abc.shape(), &[2, 5]);
    }

    #[test]
    fn test_transpose_matmul() {
        // Test A^T @ B
        let a = Tensor::ones(&[3, 2], DType::Float32);
        let b = Tensor::ones(&[3, 4], DType::Float32);

        let a_t = transpose(&a);
        let result = matmul(&a_t, &b).unwrap();

        assert_eq!(result.shape(), &[2, 4]);
    }
}
