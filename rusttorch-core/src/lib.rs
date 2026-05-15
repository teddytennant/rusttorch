//! RustTorch Core Library
//!
//! High-performance tensor, autograd, and neural network primitives with a PyTorch-like API.
//! Standalone experimental implementation (CPU, 668+ tests, clippy-clean). Not a PyTorch drop-in backend.

pub mod autograd;
pub mod data;
pub mod error;
pub mod memory;
pub mod nn;
pub mod ops;
pub mod tensor;
pub mod utils;

// FFI module for C/C++ integration (feature-gated)
#[cfg(feature = "ffi")]
pub mod ffi;

pub use data::*;
pub use error::{Result, TensorError};
pub use ops::*;
pub use tensor::{DType, Tensor};
pub use tensor::view::{TensorView as ZeroCopyView, TensorViewMut as ZeroCopyViewMut};

#[cfg(test)]
mod tests {
    #[test]
    fn test_library_loads() {
        // Basic smoke test to ensure library compiles
        let _ = crate::tensor::Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[3]);
    }
}
