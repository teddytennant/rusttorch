//! Tensor module - Core tensor types and operations
//!
//! This module provides the fundamental tensor abstractions for RustTorch,
//! designed to be similar to PyTorch's tensor API while leveraging Rust's
//! safety guarantees.

pub mod dtype;
pub mod storage;
pub mod shape;

pub use dtype::DType;
use crate::error::{TensorError, Result};
use ndarray::{Array, IxDyn};
use std::fmt;
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
#[derive(Debug, Clone)]
pub(crate) enum TensorData {
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

    /// Create a tensor from a Vec of data (f32)
    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Self {
        let array = Array::from_shape_vec(IxDyn(shape), data)
            .expect("Shape and data length must match");
        Tensor {
            data: Arc::new(TensorData::Float32(array)),
            dtype: DType::Float32,
        }
    }

    /// Create a Float32 tensor from a Vec with error handling
    pub fn from_vec_f32(data: Vec<f32>, shape: &[usize]) -> Result<Self> {
        let array = Array::from_shape_vec(IxDyn(shape), data)
            .map_err(|e| TensorError::InvalidArgument {
                parameter: "shape".to_string(),
                reason: format!("Shape and data length mismatch: {}", e),
            })?;
        Ok(Tensor {
            data: Arc::new(TensorData::Float32(array)),
            dtype: DType::Float32,
        })
    }

    /// Create a Float64 tensor from a Vec
    pub fn from_vec_f64(data: Vec<f64>, shape: &[usize]) -> Result<Self> {
        let array = Array::from_shape_vec(IxDyn(shape), data)
            .map_err(|e| TensorError::InvalidArgument {
                parameter: "shape".to_string(),
                reason: format!("Shape and data length mismatch: {}", e),
            })?;
        Ok(Tensor {
            data: Arc::new(TensorData::Float64(array)),
            dtype: DType::Float64,
        })
    }

    /// Create an Int32 tensor from a Vec
    pub fn from_vec_i32(data: Vec<i32>, shape: &[usize]) -> Result<Self> {
        let array = Array::from_shape_vec(IxDyn(shape), data)
            .map_err(|e| TensorError::InvalidArgument {
                parameter: "shape".to_string(),
                reason: format!("Shape and data length mismatch: {}", e),
            })?;
        Ok(Tensor {
            data: Arc::new(TensorData::Int32(array)),
            dtype: DType::Int32,
        })
    }

    /// Create an Int64 tensor from a Vec
    pub fn from_vec_i64(data: Vec<i64>, shape: &[usize]) -> Result<Self> {
        let array = Array::from_shape_vec(IxDyn(shape), data)
            .map_err(|e| TensorError::InvalidArgument {
                parameter: "shape".to_string(),
                reason: format!("Shape and data length mismatch: {}", e),
            })?;
        Ok(Tensor {
            data: Arc::new(TensorData::Int64(array)),
            dtype: DType::Int64,
        })
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

    /// Get the total number of elements with overflow checking
    pub fn checked_numel(&self) -> Result<usize> {
        self.shape()
            .iter()
            .try_fold(1usize, |acc, &dim| {
                acc.checked_mul(dim).ok_or_else(|| TensorError::SizeOverflow {
                    dimensions: self.shape().to_vec(),
                })
            })
    }

    /// Alternative name for numel (more Rust-idiomatic)
    pub fn num_elements(&self) -> usize {
        self.numel()
    }

    /// Get the data type
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get a reference to the internal data (for operations)
    pub(crate) fn data(&self) -> &TensorData {
        &self.data
    }

    /// Create a tensor from TensorData (for operations)
    pub(crate) fn from_data(data: TensorData, dtype: DType) -> Self {
        Tensor {
            data: Arc::new(data),
            dtype,
        }
    }
}

impl TensorData {
    /// Get the shape of the underlying array
    pub fn shape(&self) -> &[usize] {
        match self {
            TensorData::Float32(arr) => arr.shape(),
            TensorData::Float64(arr) => arr.shape(),
            TensorData::Int32(arr) => arr.shape(),
            TensorData::Int64(arr) => arr.shape(),
        }
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Tensor(shape={:?}, dtype={:?})", self.shape(), self.dtype())
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        if self.shape() != other.shape() || self.dtype() != other.dtype() {
            return false;
        }

        // Compare data element-wise
        match (self.data(), other.data()) {
            (TensorData::Float32(a), TensorData::Float32(b)) => a == b,
            (TensorData::Float64(a), TensorData::Float64(b)) => a == b,
            (TensorData::Int32(a), TensorData::Int32(b)) => a == b,
            (TensorData::Int64(a), TensorData::Int64(b)) => a == b,
            _ => false,
        }
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
