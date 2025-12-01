//! C-compatible tensor types for FFI

use crate::error::TensorError;
use crate::tensor::{DType, Tensor};
use std::os::raw::{c_int, c_void};

/// C-compatible tensor descriptor
///
/// Represents tensor metadata passed from C++ to Rust.
/// Uses raw pointers for C compatibility.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CTensorDescriptor {
    /// Pointer to tensor data
    pub data: *mut c_void,
    /// Pointer to shape array
    pub shape: *const usize,
    /// Pointer to stride array
    pub strides: *const usize,
    /// Number of dimensions
    pub ndim: usize,
    /// Total number of elements
    pub numel: usize,
    /// Data type (0=Float32, 1=Float64, 2=Int32, 3=Int64)
    pub dtype: c_int,
}

/// Opaque handle to a Rust tensor
///
/// Used to pass Rust tensors back to C++ without exposing internals.
#[repr(C)]
pub struct CTensor {
    _private: [u8; 0],
}

/// C-compatible result type for operations that can fail
#[repr(C)]
pub struct CResult {
    /// Success flag (true if operation succeeded)
    pub success: bool,
    /// Error message (null if success=true)
    pub error_msg: *mut std::os::raw::c_char,
}

impl CResult {
    /// Create success result
    pub fn ok() -> Self {
        CResult {
            success: true,
            error_msg: std::ptr::null_mut(),
        }
    }

    /// Create error result
    pub fn err(error: TensorError) -> Self {
        use std::ffi::CString;
        let msg = CString::new(error.to_string())
            .unwrap_or_else(|_| CString::new("Unknown error").unwrap());
        CResult {
            success: false,
            error_msg: msg.into_raw(),
        }
    }
}

/// Free error message allocated by Rust
#[no_mangle]
pub unsafe extern "C" fn rusttorch_free_error_message(msg: *mut std::os::raw::c_char) {
    if !msg.is_null() {
        let _ = std::ffi::CString::from_raw(msg);
    }
}

/// Check if tensor is contiguous
fn is_contiguous(desc: &CTensorDescriptor) -> bool {
    if desc.ndim == 0 {
        return true;
    }

    unsafe {
        let shape = std::slice::from_raw_parts(desc.shape, desc.ndim);
        let strides = std::slice::from_raw_parts(desc.strides, desc.ndim);

        let mut expected_stride = 1;
        for i in (0..desc.ndim).rev() {
            if strides[i] != expected_stride {
                return false;
            }
            expected_stride *= shape[i];
        }
        true
    }
}

/// Convert C tensor descriptor to Rust Tensor
///
/// # Safety
///
/// Caller must ensure:
/// - All pointers in `desc` are valid
/// - Data pointed to remains valid for the duration of the operation
/// - No concurrent mutations occur
pub unsafe fn c_tensor_to_rust(desc: &CTensorDescriptor) -> Result<Tensor, TensorError> {
    // Determine dtype
    let dtype = match desc.dtype {
        0 => DType::Float32,
        1 => DType::Float64,
        2 => DType::Int32,
        3 => DType::Int64,
        _ => {
            return Err(TensorError::InvalidArgument {
                parameter: "dtype".to_string(),
                reason: format!("Unknown dtype code: {}", desc.dtype),
            })
        }
    };

    // Extract shape
    let shape = std::slice::from_raw_parts(desc.shape, desc.ndim);

    // Handle non-contiguous tensors by copying
    // TODO: Optimize by handling strides in operations
    if !is_contiguous(desc) {
        // Copy to contiguous memory
        return match dtype {
            DType::Float32 => {
                let src = desc.data as *const f32;
                let mut data = Vec::with_capacity(desc.numel);
                // Simplified: assume row-major, proper stride handling needed
                for i in 0..desc.numel {
                    data.push(*src.add(i));
                }
                Ok(Tensor::from_vec(data, shape))
            }
            DType::Float64 => {
                let src = desc.data as *const f64;
                let mut data = Vec::with_capacity(desc.numel);
                for i in 0..desc.numel {
                    data.push(*src.add(i));
                }
                Ok(Tensor::from_vec(data, shape))
            }
            DType::Int32 => {
                let src = desc.data as *const i32;
                let mut data = Vec::with_capacity(desc.numel);
                for i in 0..desc.numel {
                    data.push(*src.add(i));
                }
                Ok(Tensor::from_vec(data, shape))
            }
            DType::Int64 => {
                let src = desc.data as *const i64;
                let mut data = Vec::with_capacity(desc.numel);
                for i in 0..desc.numel {
                    data.push(*src.add(i));
                }
                Ok(Tensor::from_vec(data, shape))
            }
        };
    }

    // Contiguous case: can use from_vec efficiently
    match dtype {
        DType::Float32 => {
            let src = std::slice::from_raw_parts(desc.data as *const f32, desc.numel);
            Ok(Tensor::from_vec(src.to_vec(), shape))
        }
        DType::Float64 => {
            let src = std::slice::from_raw_parts(desc.data as *const f64, desc.numel);
            Ok(Tensor::from_vec(src.to_vec(), shape))
        }
        DType::Int32 => {
            let src = std::slice::from_raw_parts(desc.data as *const i32, desc.numel);
            Ok(Tensor::from_vec(src.to_vec(), shape))
        }
        DType::Int64 => {
            let src = std::slice::from_raw_parts(desc.data as *const i64, desc.numel);
            Ok(Tensor::from_vec(src.to_vec(), shape))
        }
    }
}

/// Copy Rust tensor data back to C tensor
///
/// # Safety
///
/// Caller must ensure output buffer has sufficient capacity
pub unsafe fn rust_tensor_to_c(tensor: &Tensor, output: *mut c_void) {
    use crate::tensor::TensorData;

    match tensor.data() {
        TensorData::Float32(arr) => {
            let dst = output as *mut f32;
            let src = arr.as_slice().unwrap();
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len());
        }
        TensorData::Float64(arr) => {
            let dst = output as *mut f64;
            let src = arr.as_slice().unwrap();
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len());
        }
        TensorData::Int32(arr) => {
            let dst = output as *mut i32;
            let src = arr.as_slice().unwrap();
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len());
        }
        TensorData::Int64(arr) => {
            let dst = output as *mut i64;
            let src = arr.as_slice().unwrap();
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contiguous_detection() {
        // 2x3 matrix with row-major strides
        let shape = [2, 3];
        let strides = [3, 1]; // row-major

        let desc = CTensorDescriptor {
            data: std::ptr::null_mut(),
            shape: shape.as_ptr(),
            strides: strides.as_ptr(),
            ndim: 2,
            numel: 6,
            dtype: 0,
        };

        assert!(is_contiguous(&desc));

        // Non-contiguous (column-major)
        let strides_col = [1, 2];
        let desc_noncontig = CTensorDescriptor {
            strides: strides_col.as_ptr(),
            ..desc
        };

        assert!(!is_contiguous(&desc_noncontig));
    }
}
