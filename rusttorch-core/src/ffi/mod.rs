//! Foreign Function Interface (FFI) module
//!
//! This module provides C-compatible exports of RustTorch operations
//! for integration with PyTorch's C++ backend.
//!
//! # Safety
//!
//! All FFI functions are marked `unsafe` because they accept raw pointers
//! from C/C++. Callers must ensure:
//! - Pointers are valid and properly aligned
//! - Memory is not accessed after being freed
//! - No data races occur (single-threaded or properly synchronized)

pub mod tensor;
pub mod ops;

pub use tensor::{CTensor, CTensorDescriptor};
pub use ops::*;

/// Initialize RustTorch FFI layer
///
/// Must be called once before using any FFI functions.
/// This sets up logging, thread pools, and other global state.
#[no_mangle]
pub extern "C" fn rusttorch_init() {
    // Initialize Rayon thread pool if not already initialized
    // This is safe to call multiple times
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_cpus::get())
        .build_global()
        .ok(); // Ignore error if already initialized
}

/// Cleanup RustTorch FFI layer
///
/// Should be called on shutdown to clean up resources.
#[no_mangle]
pub extern "C" fn rusttorch_cleanup() {
    // Currently no cleanup needed
    // Reserved for future use (memory pools, etc.)
}

/// Get RustTorch version string
#[no_mangle]
pub extern "C" fn rusttorch_version() -> *const std::os::raw::c_char {
    // Return static string, no need to free
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const std::os::raw::c_char
}
