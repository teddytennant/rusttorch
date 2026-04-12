//! RustTorch Core Library
//!
//! High-performance tensor operations and neural network primitives written in Rust.
//! Designed to be a drop-in replacement for performance-critical PyTorch CPU operations.

pub mod autograd;
pub mod backend;
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

pub use backend::{Backend, NdArrayBackend};
pub use data::*;
pub use error::{Result, TensorError};
pub use ops::*;
pub use tensor::{DType, Device, Tensor, TensorView};

#[cfg(test)]
mod tests {
    #[test]
    fn test_library_loads() {
        // Basic smoke test to ensure library compiles
        let _ = crate::tensor::Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[3]);
    }
}
