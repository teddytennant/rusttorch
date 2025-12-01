//! RustTorch Core Library
//!
//! High-performance tensor operations and neural network primitives written in Rust.
//! Designed to be a drop-in replacement for performance-critical PyTorch CPU operations.

pub mod data;
pub mod error;
pub mod memory;
pub mod ops;
pub mod tensor;
pub mod utils;

// FFI module for C/C++ integration (feature-gated)
#[cfg(feature = "ffi")]
pub mod ffi;

pub use data::*;
pub use error::{Result, TensorError};
pub use ops::*;
pub use tensor::{DType, Tensor, TensorView};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_library_loads() {
        // Basic smoke test to ensure library compiles
        assert!(true);
    }
}
