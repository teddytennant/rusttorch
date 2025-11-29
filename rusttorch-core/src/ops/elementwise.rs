//! Element-wise tensor operations

use crate::tensor::{Tensor, TensorData};
use crate::DType;
use ndarray::{Array, IxDyn, Zip};

/// Element-wise addition of two tensors
pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.dtype(), b.dtype(), "Tensors must have the same dtype");
    assert_eq!(a.shape(), b.shape(), "Tensors must have the same shape for element-wise addition");

    let dtype = a.dtype();
    let data = match (a.data(), b.data()) {
        (TensorData::Float32(arr_a), TensorData::Float32(arr_b)) => {
            TensorData::Float32(arr_a + arr_b)
        }
        (TensorData::Float64(arr_a), TensorData::Float64(arr_b)) => {
            TensorData::Float64(arr_a + arr_b)
        }
        (TensorData::Int32(arr_a), TensorData::Int32(arr_b)) => {
            TensorData::Int32(arr_a + arr_b)
        }
        (TensorData::Int64(arr_a), TensorData::Int64(arr_b)) => {
            TensorData::Int64(arr_a + arr_b)
        }
        _ => panic!("Mismatched tensor data types"),
    };

    Tensor::from_data(data, dtype)
}

/// Element-wise multiplication of two tensors
pub fn mul(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.dtype(), b.dtype(), "Tensors must have the same dtype");
    assert_eq!(a.shape(), b.shape(), "Tensors must have the same shape for element-wise multiplication");

    let dtype = a.dtype();
    let data = match (a.data(), b.data()) {
        (TensorData::Float32(arr_a), TensorData::Float32(arr_b)) => {
            TensorData::Float32(arr_a * arr_b)
        }
        (TensorData::Float64(arr_a), TensorData::Float64(arr_b)) => {
            TensorData::Float64(arr_a * arr_b)
        }
        (TensorData::Int32(arr_a), TensorData::Int32(arr_b)) => {
            TensorData::Int32(arr_a * arr_b)
        }
        (TensorData::Int64(arr_a), TensorData::Int64(arr_b)) => {
            TensorData::Int64(arr_a * arr_b)
        }
        _ => panic!("Mismatched tensor data types"),
    };

    Tensor::from_data(data, dtype)
}

/// Element-wise subtraction
pub fn sub(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.dtype(), b.dtype(), "Tensors must have the same dtype");
    assert_eq!(a.shape(), b.shape(), "Tensors must have the same shape for element-wise subtraction");

    let dtype = a.dtype();
    let data = match (a.data(), b.data()) {
        (TensorData::Float32(arr_a), TensorData::Float32(arr_b)) => {
            TensorData::Float32(arr_a - arr_b)
        }
        (TensorData::Float64(arr_a), TensorData::Float64(arr_b)) => {
            TensorData::Float64(arr_a - arr_b)
        }
        (TensorData::Int32(arr_a), TensorData::Int32(arr_b)) => {
            TensorData::Int32(arr_a - arr_b)
        }
        (TensorData::Int64(arr_a), TensorData::Int64(arr_b)) => {
            TensorData::Int64(arr_a - arr_b)
        }
        _ => panic!("Mismatched tensor data types"),
    };

    Tensor::from_data(data, dtype)
}

/// Element-wise division
pub fn div(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.dtype(), b.dtype(), "Tensors must have the same dtype");
    assert_eq!(a.shape(), b.shape(), "Tensors must have the same shape for element-wise division");
    assert!(a.dtype().is_float(), "Division requires floating-point tensors");

    let dtype = a.dtype();
    let data = match (a.data(), b.data()) {
        (TensorData::Float32(arr_a), TensorData::Float32(arr_b)) => {
            TensorData::Float32(arr_a / arr_b)
        }
        (TensorData::Float64(arr_a), TensorData::Float64(arr_b)) => {
            TensorData::Float64(arr_a / arr_b)
        }
        _ => panic!("Division only supported for floating-point types"),
    };

    Tensor::from_data(data, dtype)
}

/// Scalar addition
pub fn add_scalar(tensor: &Tensor, scalar: f32) -> Tensor {
    let dtype = tensor.dtype();
    let data = match tensor.data() {
        TensorData::Float32(arr) => {
            TensorData::Float32(arr + scalar)
        }
        TensorData::Float64(arr) => {
            TensorData::Float64(arr + scalar as f64)
        }
        TensorData::Int32(arr) => {
            TensorData::Int32(arr + scalar as i32)
        }
        TensorData::Int64(arr) => {
            TensorData::Int64(arr + scalar as i64)
        }
    };

    Tensor::from_data(data, dtype)
}

/// Scalar multiplication
pub fn mul_scalar(tensor: &Tensor, scalar: f32) -> Tensor {
    let dtype = tensor.dtype();
    let data = match tensor.data() {
        TensorData::Float32(arr) => {
            TensorData::Float32(arr * scalar)
        }
        TensorData::Float64(arr) => {
            TensorData::Float64(arr * scalar as f64)
        }
        TensorData::Int32(arr) => {
            TensorData::Int32(arr * scalar as i32)
        }
        TensorData::Int64(arr) => {
            TensorData::Int64(arr * scalar as i64)
        }
    };

    Tensor::from_data(data, dtype)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let c = add(&a, &b);

        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.dtype(), DType::Float32);
    }

    #[test]
    fn test_mul() {
        let a = Tensor::from_vec(vec![2.0, 3.0, 4.0, 5.0], &[2, 2]);
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let c = mul(&a, &b);

        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.dtype(), DType::Float32);
    }

    #[test]
    fn test_sub() {
        let a = Tensor::from_vec(vec![10.0, 9.0, 8.0, 7.0], &[2, 2]);
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let c = sub(&a, &b);

        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.dtype(), DType::Float32);
    }

    #[test]
    fn test_div() {
        let a = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]);
        let b = Tensor::from_vec(vec![2.0, 4.0, 5.0, 8.0], &[2, 2]);
        let c = div(&a, &b);

        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.dtype(), DType::Float32);
    }

    #[test]
    fn test_add_scalar() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = add_scalar(&a, 10.0);

        assert_eq!(b.shape(), &[2, 2]);
        assert_eq!(b.dtype(), DType::Float32);
    }

    #[test]
    fn test_mul_scalar() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = mul_scalar(&a, 2.0);

        assert_eq!(b.shape(), &[2, 2]);
        assert_eq!(b.dtype(), DType::Float32);
    }

    #[test]
    #[should_panic(expected = "same shape")]
    fn test_add_shape_mismatch() {
        let a = Tensor::ones(&[2, 2], DType::Float32);
        let b = Tensor::ones(&[3, 3], DType::Float32);
        let _ = add(&a, &b);
    }

    #[test]
    #[should_panic(expected = "same dtype")]
    fn test_add_dtype_mismatch() {
        let a = Tensor::ones(&[2, 2], DType::Float32);
        let b = Tensor::ones(&[2, 2], DType::Float64);
        let _ = add(&a, &b);
    }
}
