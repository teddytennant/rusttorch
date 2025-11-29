//! Tensor module - Core tensor types and operations
//!
//! This module provides the fundamental tensor abstractions for RustTorch,
//! designed to be similar to PyTorch's tensor API while leveraging Rust's
//! safety guarantees.

pub mod dtype;
pub mod storage;
pub mod shape;

pub use dtype::DType;
use ndarray::{Array, IxDyn};
use std::sync::Arc;

/// A multi-dimensional tensor with dynamic shape
#[derive(Debug, Clone)]
pub struct Tensor {
    /// The underlying data storage
    data: Arc<TensorData>,
    /// Data type of tensor elements
    dtype: DType,
}

/// Internal tensor data representation
#[derive(Debug)]
enum TensorData {
    Float32(Array<f32, IxDyn>),
    Float64(Array<f64, IxDyn>),
    Int32(Array<i32, IxDyn>),
    Int64(Array<i64, IxDyn>),
}

impl Tensor {
    /// Create a new tensor filled with zeros
    pub fn zeros(shape: &[usize], dtype: DType) -> Self {
        let data = match dtype {
            DType::Float32 => TensorData::Float32(Array::zeros(IxDyn(shape))),
            DType::Float64 => TensorData::Float64(Array::zeros(IxDyn(shape))),
            DType::Int32 => TensorData::Int32(Array::zeros(IxDyn(shape))),
            DType::Int64 => TensorData::Int64(Array::zeros(IxDyn(shape))),
        };
        Tensor {
            data: Arc::new(data),
            dtype,
        }
    }

    /// Create a new tensor filled with ones
    pub fn ones(shape: &[usize], dtype: DType) -> Self {
        let data = match dtype {
            DType::Float32 => TensorData::Float32(Array::ones(IxDyn(shape))),
            DType::Float64 => TensorData::Float64(Array::ones(IxDyn(shape))),
            DType::Int32 => TensorData::Int32(Array::ones(IxDyn(shape))),
            DType::Int64 => TensorData::Int64(Array::ones(IxDyn(shape))),
        };
        Tensor {
            data: Arc::new(data),
            dtype,
        }
    }

    /// Create a tensor from a Vec of data
    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Self {
        let array = Array::from_shape_vec(IxDyn(shape), data)
            .expect("Shape and data length must match");
        Tensor {
            data: Arc::new(TensorData::Float32(array)),
            dtype: DType::Float32,
        }
    }

    /// Get the shape of the tensor
    pub fn shape(&self) -> &[usize] {
        match &*self.data {
            TensorData::Float32(arr) => arr.shape(),
            TensorData::Float64(arr) => arr.shape(),
            TensorData::Int32(arr) => arr.shape(),
            TensorData::Int64(arr) => arr.shape(),
        }
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape().len()
    }

    /// Get the total number of elements
    pub fn numel(&self) -> usize {
        self.shape().iter().product()
    }

    /// Get the data type
    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

/// A view into a tensor (non-owning)
pub struct TensorView<'a> {
    _phantom: std::marker::PhantomData<&'a Tensor>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_zeros() {
        let t = Tensor::zeros(&[2, 3], DType::Float32);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.ndim(), 2);
        assert_eq!(t.numel(), 6);
    }

    #[test]
    fn test_tensor_ones() {
        let t = Tensor::ones(&[4, 5], DType::Float32);
        assert_eq!(t.shape(), &[4, 5]);
        assert_eq!(t.numel(), 20);
    }

    #[test]
    fn test_tensor_from_vec() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let t = Tensor::from_vec(data, &[2, 2]);
        assert_eq!(t.shape(), &[2, 2]);
        assert_eq!(t.dtype(), DType::Float32);
    }
}
