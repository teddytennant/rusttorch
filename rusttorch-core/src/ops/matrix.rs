//! Matrix operations
//!
//! This module provides matrix-specific operations including:
//! - Matrix multiplication (matmul)
//! - Transpose
//! - Reshape

use crate::tensor::{DType, Tensor, TensorData};
use ndarray::{Array, IxDyn};

/// Matrix multiplication
///
/// Performs matrix multiplication between two tensors.
/// Supports 2D matmul and batched matmul for N-D (N>=3) where leading batch
/// dimensions match exactly. Batch broadcasting is not supported (use explicit
/// expand or separate calls).
///
/// # Arguments
/// * `a` - First tensor (at least 2D, shape [..., M, K])
/// * `b` - Second tensor (at least 2D, shape [..., K, N] with matching batch prefix)
///
/// # Returns
/// Result [..., M, N] or error on dim mismatch / unsupported batch shape
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
            Ok(Tensor::from_data(
                TensorData::Float32(result),
                DType::Float32,
            ))
        }
        (TensorData::Float64(a_arr), TensorData::Float64(b_arr)) => {
            let result = matmul_float64(a_arr, b_arr)?;
            Ok(Tensor::from_data(
                TensorData::Float64(result),
                DType::Float64,
            ))
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

    if a_shape.len() == 2 && b_shape.len() == 2 {
        let a_2d = a.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
        let b_2d = b.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
        let result = a_2d.dot(&b_2d);
        return Ok(result.into_dyn());
    }

    // Batched: require same ndim >=3 and exact match on leading batch dims
    if a_shape.len() == b_shape.len() && a_shape.len() >= 3 {
        let ndim = a_shape.len();
        let prefix = &a_shape[..ndim - 2];
        if prefix != &b_shape[..ndim - 2] {
            return Err("Batched matmul requires identical leading batch dimensions".to_string());
        }
        let batch: usize = prefix.iter().product();
        let m = a_shape[ndim - 2];
        let k = a_shape[ndim - 1];
        let n = b_shape[ndim - 1];

        let a_flat = a.clone().into_shape((batch, m, k)).map_err(|e| e.to_string())?;
        let b_flat = b.clone().into_shape((batch, k, n)).map_err(|e| e.to_string())?;

        let mut out = Array::<f32, _>::zeros((batch, m, n));
        for i in 0..batch {
            let ai = a_flat
                .index_axis(ndarray::Axis(0), i)
                .into_dimensionality::<ndarray::Ix2>()
                .unwrap();
            let bi = b_flat
                .index_axis(ndarray::Axis(0), i)
                .into_dimensionality::<ndarray::Ix2>()
                .unwrap();
            let oi = ai.dot(&bi);
            out.index_axis_mut(ndarray::Axis(0), i).assign(&oi);
        }

        // Reshape back to original batch prefix + [m, n]
        let mut new_shape = prefix.to_vec();
        new_shape.push(m);
        new_shape.push(n);
        return out.into_shape(new_shape).map_err(|e| e.to_string());
    }

    Err(format!(
        "matmul for {}D @ {}D not supported (only 2D or N-D with matching batch prefix)",
        a_shape.len(),
        b_shape.len()
    ))
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

    if a_shape.len() == b_shape.len() && a_shape.len() >= 3 {
        let ndim = a_shape.len();
        let prefix = &a_shape[..ndim - 2];
        if prefix != &b_shape[..ndim - 2] {
            return Err("Batched matmul requires identical leading batch dimensions".to_string());
        }
        let batch: usize = prefix.iter().product();
        let m = a_shape[ndim - 2];
        let k = a_shape[ndim - 1];
        let n = b_shape[ndim - 1];

        let a_flat = a.clone().into_shape((batch, m, k)).map_err(|e| e.to_string())?;
        let b_flat = b.clone().into_shape((batch, k, n)).map_err(|e| e.to_string())?;

        let mut out = Array::<f64, _>::zeros((batch, m, n));
        for i in 0..batch {
            let ai = a_flat
                .index_axis(ndarray::Axis(0), i)
                .into_dimensionality::<ndarray::Ix2>()
                .unwrap();
            let bi = b_flat
                .index_axis(ndarray::Axis(0), i)
                .into_dimensionality::<ndarray::Ix2>()
                .unwrap();
            let oi = ai.dot(&bi);
            out.index_axis_mut(ndarray::Axis(0), i).assign(&oi);
        }

        let mut new_shape = prefix.to_vec();
        new_shape.push(m);
        new_shape.push(n);
        return out.into_shape(new_shape).map_err(|e| e.to_string());
    }

    Err(format!(
        "matmul for {}D @ {}D not supported",
        a_shape.len(),
        b_shape.len()
    ))
}

fn matmul_int32(a: &Array<i32, IxDyn>, b: &Array<i32, IxDyn>) -> Result<Array<i32, IxDyn>, String> {
    let a_shape = a.shape();
    let b_shape = b.shape();

    if a_shape.len() == 2 && b_shape.len() == 2 {
        let a_2d = a.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
        let b_2d = b.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
        let result = a_2d.dot(&b_2d);
        return Ok(result.into_dyn());
    }

    if a_shape.len() == b_shape.len() && a_shape.len() >= 3 {
        let ndim = a_shape.len();
        let prefix = &a_shape[..ndim - 2];
        if prefix != &b_shape[..ndim - 2] {
            return Err("Batched matmul requires identical leading batch dimensions".to_string());
        }
        let batch: usize = prefix.iter().product();
        let m = a_shape[ndim - 2];
        let k = a_shape[ndim - 1];
        let n = b_shape[ndim - 1];

        let a_flat = a.clone().into_shape((batch, m, k)).map_err(|e| e.to_string())?;
        let b_flat = b.clone().into_shape((batch, k, n)).map_err(|e| e.to_string())?;

        let mut out = Array::<i32, _>::zeros((batch, m, n));
        for i in 0..batch {
            let ai = a_flat
                .index_axis(ndarray::Axis(0), i)
                .into_dimensionality::<ndarray::Ix2>()
                .unwrap();
            let bi = b_flat
                .index_axis(ndarray::Axis(0), i)
                .into_dimensionality::<ndarray::Ix2>()
                .unwrap();
            let oi = ai.dot(&bi);
            out.index_axis_mut(ndarray::Axis(0), i).assign(&oi);
        }

        let mut new_shape = prefix.to_vec();
        new_shape.push(m);
        new_shape.push(n);
        return out.into_shape(new_shape).map_err(|e| e.to_string());
    }

    Err(format!(
        "matmul for {}D @ {}D not supported",
        a_shape.len(),
        b_shape.len()
    ))
}

fn matmul_int64(a: &Array<i64, IxDyn>, b: &Array<i64, IxDyn>) -> Result<Array<i64, IxDyn>, String> {
    let a_shape = a.shape();
    let b_shape = b.shape();

    if a_shape.len() == 2 && b_shape.len() == 2 {
        let a_2d = a.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
        let b_2d = b.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
        let result = a_2d.dot(&b_2d);
        return Ok(result.into_dyn());
    }

    if a_shape.len() == b_shape.len() && a_shape.len() >= 3 {
        let ndim = a_shape.len();
        let prefix = &a_shape[..ndim - 2];
        if prefix != &b_shape[..ndim - 2] {
            return Err("Batched matmul requires identical leading batch dimensions".to_string());
        }
        let batch: usize = prefix.iter().product();
        let m = a_shape[ndim - 2];
        let k = a_shape[ndim - 1];
        let n = b_shape[ndim - 1];

        let a_flat = a.clone().into_shape((batch, m, k)).map_err(|e| e.to_string())?;
        let b_flat = b.clone().into_shape((batch, k, n)).map_err(|e| e.to_string())?;

        let mut out = Array::<i64, _>::zeros((batch, m, n));
        for i in 0..batch {
            let ai = a_flat
                .index_axis(ndarray::Axis(0), i)
                .into_dimensionality::<ndarray::Ix2>()
                .unwrap();
            let bi = b_flat
                .index_axis(ndarray::Axis(0), i)
                .into_dimensionality::<ndarray::Ix2>()
                .unwrap();
            let oi = ai.dot(&bi);
            out.index_axis_mut(ndarray::Axis(0), i).assign(&oi);
        }

        let mut new_shape = prefix.to_vec();
        new_shape.push(m);
        new_shape.push(n);
        return out.into_shape(new_shape).map_err(|e| e.to_string());
    }

    Err(format!(
        "matmul for {}D @ {}D not supported",
        a_shape.len(),
        b_shape.len()
    ))
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

    #[test]
    fn test_matmul_batched_3d() {
        // [2, 3, 4] @ [2, 4, 5] -> [2, 3, 5]
        let a = Tensor::from_vec((0..24).map(|x| x as f32).collect(), &[2, 3, 4]);
        let b = Tensor::from_vec((0..40).map(|x| x as f32).collect(), &[2, 4, 5]);
        let c = matmul(&a, &b).unwrap();

        assert_eq!(c.shape(), &[2, 3, 5]);
        assert_eq!(c.numel(), 30);
    }

    #[test]
    fn test_matmul_batched_4d() {
        // [1, 2, 2, 2] @ [1, 2, 2, 3] -> [1, 2, 2, 3]
        let a = Tensor::ones(&[1, 2, 2, 2], DType::Float32);
        let b = Tensor::ones(&[1, 2, 2, 3], DType::Float32);
        let c = matmul(&a, &b).unwrap();

        assert_eq!(c.shape(), &[1, 2, 2, 3]);
    }

    #[test]
    fn test_matmul_batched_mismatched_batch() {
        let a = Tensor::ones(&[2, 3, 4], DType::Float32);
        let b = Tensor::ones(&[3, 4, 5], DType::Float32); // batch 3 != 2
        let result = matmul(&a, &b);

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("batch"));
    }
}
