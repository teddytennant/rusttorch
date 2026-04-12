//! Tensor module - Core tensor types and operations
//!
//! This module provides the fundamental tensor abstractions for RustTorch,
//! designed to be similar to PyTorch's tensor API while leveraging Rust's
//! safety guarantees.

pub mod device;
pub mod dtype;
pub mod shape;
pub mod storage;
pub mod view;

use crate::error::{Result, TensorError};
pub use device::Device;
pub use dtype::DType;
use ndarray::{Array, IxDyn};
use std::fmt;
use std::sync::Arc;
pub use view::{TensorView as ZeroCopyView, TensorViewMut as ZeroCopyViewMut};

/// A multi-dimensional tensor with dynamic shape.
///
/// Tensors carry three pieces of metadata:
/// - `data`: the actual storage (an ndarray `Array` today; will become a
///   trait-object backend as GPU support lands).
/// - `dtype`: element type (`f32`, `f64`, `i32`, `i64`).
/// - `device`: physical location of the storage (`Cpu` or `Cuda(id)`).
///
/// All existing constructors default `device` to `Device::Cpu` so this
/// refactor does not break any caller. Ops that produce new tensors
/// inherit the input device via `from_data_on_device`.
#[derive(Debug, Clone)]
pub struct Tensor {
    /// The underlying data storage
    data: Arc<TensorData>,
    /// Data type of tensor elements
    dtype: DType,
    /// Physical device where the data lives
    device: Device,
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
            device: Device::Cpu,
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
            device: Device::Cpu,
        }
    }

    /// Create a tensor from a Vec of data (f32)
    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Self {
        let array =
            Array::from_shape_vec(IxDyn(shape), data).expect("Shape and data length must match");
        Tensor {
            data: Arc::new(TensorData::Float32(array)),
            dtype: DType::Float32,
            device: Device::Cpu,
        }
    }

    /// Create a Float32 tensor from a Vec with error handling
    pub fn from_vec_f32(data: Vec<f32>, shape: &[usize]) -> Result<Self> {
        let array = Array::from_shape_vec(IxDyn(shape), data).map_err(|e| {
            TensorError::InvalidArgument {
                parameter: "shape".to_string(),
                reason: format!("Shape and data length mismatch: {}", e),
            }
        })?;
        Ok(Tensor {
            data: Arc::new(TensorData::Float32(array)),
            dtype: DType::Float32,
            device: Device::Cpu,
        })
    }

    /// Create a Float64 tensor from a Vec
    pub fn from_vec_f64(data: Vec<f64>, shape: &[usize]) -> Result<Self> {
        let array = Array::from_shape_vec(IxDyn(shape), data).map_err(|e| {
            TensorError::InvalidArgument {
                parameter: "shape".to_string(),
                reason: format!("Shape and data length mismatch: {}", e),
            }
        })?;
        Ok(Tensor {
            data: Arc::new(TensorData::Float64(array)),
            dtype: DType::Float64,
            device: Device::Cpu,
        })
    }

    /// Create an Int32 tensor from a Vec
    pub fn from_vec_i32(data: Vec<i32>, shape: &[usize]) -> Result<Self> {
        let array = Array::from_shape_vec(IxDyn(shape), data).map_err(|e| {
            TensorError::InvalidArgument {
                parameter: "shape".to_string(),
                reason: format!("Shape and data length mismatch: {}", e),
            }
        })?;
        Ok(Tensor {
            data: Arc::new(TensorData::Int32(array)),
            dtype: DType::Int32,
            device: Device::Cpu,
        })
    }

    /// Create an Int64 tensor from a Vec
    pub fn from_vec_i64(data: Vec<i64>, shape: &[usize]) -> Result<Self> {
        let array = Array::from_shape_vec(IxDyn(shape), data).map_err(|e| {
            TensorError::InvalidArgument {
                parameter: "shape".to_string(),
                reason: format!("Shape and data length mismatch: {}", e),
            }
        })?;
        Ok(Tensor {
            data: Arc::new(TensorData::Int64(array)),
            dtype: DType::Int64,
            device: Device::Cpu,
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
        self.shape().iter().try_fold(1usize, |acc, &dim| {
            acc.checked_mul(dim)
                .ok_or_else(|| TensorError::SizeOverflow {
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

    /// Get the device this tensor's storage lives on.
    #[inline]
    pub fn device(&self) -> Device {
        self.device
    }

    /// Move this tensor to a target device.
    ///
    /// For CPU → CPU this is a cheap clone of the Arc. CUDA support is
    /// gated behind the `cuda` feature; without it, requesting a CUDA
    /// device returns an error. When the feature is on but no CUDA
    /// backend has been compiled in yet (the current state), CUDA
    /// transfers likewise return a clear error so callers fail loudly
    /// rather than silently running on CPU.
    pub fn to(&self, device: Device) -> Result<Self> {
        if self.device == device {
            return Ok(self.clone());
        }
        match device {
            Device::Cpu => {
                // From CUDA back to CPU — today there is no CUDA storage,
                // so this arm is unreachable in practice. Kept for API
                // completeness.
                Ok(Tensor {
                    data: self.data.clone(),
                    dtype: self.dtype,
                    device: Device::Cpu,
                })
            }
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => Err(TensorError::InvalidArgument {
                parameter: "device".to_string(),
                reason: "CUDA backend is not yet implemented; see GPU_KERNELS.md"
                    .to_string(),
            }),
        }
    }

    /// Shorthand: ensure this tensor lives on the CPU.
    pub fn cpu(&self) -> Self {
        if self.device.is_cpu() {
            self.clone()
        } else {
            // Unreachable today since the only storage is CPU, but kept
            // so callers can write device-agnostic code.
            Tensor {
                data: self.data.clone(),
                dtype: self.dtype,
                device: Device::Cpu,
            }
        }
    }

    /// Shorthand: move this tensor to the given CUDA device. Requires the
    /// `cuda` feature.
    #[cfg(feature = "cuda")]
    pub fn cuda(&self, id: usize) -> Result<Self> {
        self.to(Device::Cuda(id))
    }

    /// Get a reference to the internal data (for operations)
    pub(crate) fn data(&self) -> &TensorData {
        &self.data
    }

    /// Create a tensor from TensorData (for operations). The resulting
    /// tensor lives on `Device::Cpu` because TensorData is an ndarray
    /// enum — GPU storage will eventually route through the Backend
    /// trait and not this path.
    pub(crate) fn from_data(data: TensorData, dtype: DType) -> Self {
        Tensor {
            data: Arc::new(data),
            dtype,
            device: Device::Cpu,
        }
    }

    /// Create a tensor of ones with the same shape and dtype
    pub fn ones_like(&self) -> Self {
        Tensor::ones(self.shape(), self.dtype())
    }

    /// Create a tensor of zeros with the same shape and dtype
    pub fn zeros_like(&self) -> Self {
        Tensor::zeros(self.shape(), self.dtype())
    }

    /// Element-wise negation
    pub fn neg(&self) -> Self {
        let dtype = self.dtype();
        let data = match self.data() {
            TensorData::Float32(arr) => TensorData::Float32(arr.mapv(|x| -x)),
            TensorData::Float64(arr) => TensorData::Float64(arr.mapv(|x| -x)),
            TensorData::Int32(arr) => TensorData::Int32(arr.mapv(|x| -x)),
            TensorData::Int64(arr) => TensorData::Int64(arr.mapv(|x| -x)),
        };
        Tensor::from_data(data, dtype)
    }

    /// Get a copy of the data as a flat Vec<f32> (converts if needed)
    pub fn to_vec_f32(&self) -> Vec<f32> {
        match self.data() {
            TensorData::Float32(arr) => arr.iter().copied().collect(),
            TensorData::Float64(arr) => arr.iter().map(|&x| x as f32).collect(),
            TensorData::Int32(arr) => arr.iter().map(|&x| x as f32).collect(),
            TensorData::Int64(arr) => arr.iter().map(|&x| x as f32).collect(),
        }
    }

    /// Create a scalar (0-dimensional) tensor from a single f32 value
    pub fn scalar(value: f32) -> Self {
        Tensor::from_vec(vec![value], &[1])
    }

    /// Get the scalar value from a 1-element tensor
    pub fn item(&self) -> Result<f64> {
        if self.numel() != 1 {
            return Err(TensorError::InvalidArgument {
                parameter: "tensor".to_string(),
                reason: format!(
                    "item() requires a single-element tensor, got {} elements",
                    self.numel()
                ),
            });
        }
        Ok(match self.data() {
            TensorData::Float32(arr) => arr.iter().next().copied().unwrap() as f64,
            TensorData::Float64(arr) => arr.iter().next().copied().unwrap(),
            TensorData::Int32(arr) => arr.iter().next().copied().unwrap() as f64,
            TensorData::Int64(arr) => arr.iter().next().copied().unwrap() as f64,
        })
    }

    /// Create a filled tensor with a specific f32 value
    pub fn full(shape: &[usize], value: f32) -> Self {
        let data = Array::from_elem(IxDyn(shape), value);
        Tensor::from_data(TensorData::Float32(data), DType::Float32)
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Tensor(shape={:?}, dtype={:?})",
            self.shape(),
            self.dtype()
        )
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        if self.shape() != other.shape()
            || self.dtype() != other.dtype()
            || self.device() != other.device()
        {
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

    #[test]
    fn tensors_default_to_cpu_device() {
        assert_eq!(Tensor::zeros(&[2, 2], DType::Float32).device(), Device::Cpu);
        assert_eq!(Tensor::ones(&[2, 2], DType::Float32).device(), Device::Cpu);
        assert_eq!(Tensor::from_vec(vec![1.0, 2.0], &[2]).device(), Device::Cpu);
        assert_eq!(Tensor::scalar(1.0).device(), Device::Cpu);
        assert_eq!(Tensor::full(&[3], 0.5).device(), Device::Cpu);
    }

    #[test]
    fn to_cpu_is_identity_clone() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
        let moved = t.to(Device::Cpu).unwrap();
        assert_eq!(moved.device(), Device::Cpu);
        assert_eq!(moved.to_vec_f32(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn cpu_shorthand() {
        let t = Tensor::from_vec(vec![1.0, 2.0], &[2]).cpu();
        assert!(t.device().is_cpu());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn to_cuda_is_error_without_kernels() {
        let t = Tensor::from_vec(vec![1.0, 2.0], &[2]);
        let r = t.to(Device::Cuda(0));
        assert!(
            r.is_err(),
            "CUDA transfer should return a clear error until kernels ship"
        );
    }

    #[test]
    fn ones_like_and_zeros_like_preserve_device() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
        assert_eq!(t.ones_like().device(), Device::Cpu);
        assert_eq!(t.zeros_like().device(), Device::Cpu);
    }

    #[test]
    fn partial_eq_respects_device() {
        let a = Tensor::from_vec(vec![1.0, 2.0], &[2]);
        let b = Tensor::from_vec(vec![1.0, 2.0], &[2]);
        assert_eq!(a, b);
    }
}
