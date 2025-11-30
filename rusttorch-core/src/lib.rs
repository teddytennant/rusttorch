//! RustTorch Core Library
//!
//! High-performance tensor operations and neural network primitives written in Rust.
//! Designed to be a drop-in replacement for performance-critical PyTorch CPU operations.

pub mod error;
pub mod tensor;
pub mod ops;
pub mod memory;
pub mod utils;
pub mod data;

// FFI module for C/C++ integration (feature-gated)
#[cfg(feature = "ffi")]
pub mod ffi;

pub use error::{TensorError, Result};
pub use tensor::{Tensor, TensorView, DType};
pub use ops::*;
pub use data::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_library_loads() {
        // Basic smoke test to ensure library compiles
        assert!(true);
    }
}
