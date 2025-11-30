//! FFI exports for tensor operations
//!
//! This module provides C-compatible function exports for all RustTorch operations.
//! Each function accepts raw pointers and performs the operation, writing results
//! to the output buffer.

use super::tensor::{CTensorDescriptor, CResult, c_tensor_to_rust, rust_tensor_to_c};
use crate::ops;

// ============================================================================
// Element-wise Operations
// ============================================================================

/// Element-wise addition: out = a + b
///
/// # Safety
///
/// - All tensor descriptors must be valid
/// - Output buffer must have capacity for `a.numel` elements
#[no_mangle]
pub unsafe extern "C" fn rusttorch_add(
    a: *const CTensorDescriptor,
    b: *const CTensorDescriptor,
    out: *const CTensorDescriptor,
) -> CResult {
    let a_desc = &*a;
    let b_desc = &*b;
    let out_desc = &*out;

    match (c_tensor_to_rust(a_desc), c_tensor_to_rust(b_desc)) {
        (Ok(tensor_a), Ok(tensor_b)) => {
            let result = ops::add(&tensor_a, &tensor_b);
            rust_tensor_to_c(&result, out_desc.data);
            CResult::ok()
        }
        (Err(e), _) | (_, Err(e)) => CResult::err(e),
    }
}

/// Element-wise multiplication: out = a * b
#[no_mangle]
pub unsafe extern "C" fn rusttorch_mul(
    a: *const CTensorDescriptor,
    b: *const CTensorDescriptor,
    out: *const CTensorDescriptor,
) -> CResult {
    let a_desc = &*a;
    let b_desc = &*b;
    let out_desc = &*out;

    match (c_tensor_to_rust(a_desc), c_tensor_to_rust(b_desc)) {
        (Ok(tensor_a), Ok(tensor_b)) => {
            let result = ops::mul(&tensor_a, &tensor_b);
            rust_tensor_to_c(&result, out_desc.data);
            CResult::ok()
        }
        (Err(e), _) | (_, Err(e)) => CResult::err(e),
    }
}

/// Element-wise subtraction: out = a - b
#[no_mangle]
pub unsafe extern "C" fn rusttorch_sub(
    a: *const CTensorDescriptor,
    b: *const CTensorDescriptor,
    out: *const CTensorDescriptor,
) -> CResult {
    let a_desc = &*a;
    let b_desc = &*b;
    let out_desc = &*out;

    match (c_tensor_to_rust(a_desc), c_tensor_to_rust(b_desc)) {
        (Ok(tensor_a), Ok(tensor_b)) => {
            let result = ops::sub(&tensor_a, &tensor_b);
            rust_tensor_to_c(&result, out_desc.data);
            CResult::ok()
        }
        (Err(e), _) | (_, Err(e)) => CResult::err(e),
    }
}

/// Element-wise division: out = a / b
#[no_mangle]
pub unsafe extern "C" fn rusttorch_div(
    a: *const CTensorDescriptor,
    b: *const CTensorDescriptor,
    out: *const CTensorDescriptor,
) -> CResult {
    let a_desc = &*a;
    let b_desc = &*b;
    let out_desc = &*out;

    match (c_tensor_to_rust(a_desc), c_tensor_to_rust(b_desc)) {
        (Ok(tensor_a), Ok(tensor_b)) => {
            let result = ops::div(&tensor_a, &tensor_b);
            rust_tensor_to_c(&result, out_desc.data);
            CResult::ok()
        }
        (Err(e), _) | (_, Err(e)) => CResult::err(e),
    }
}

// ============================================================================
// Activation Functions
// ============================================================================

/// ReLU activation: out = max(0, input)
#[no_mangle]
pub unsafe extern "C" fn rusttorch_relu(
    input: *const CTensorDescriptor,
    out: *const CTensorDescriptor,
) -> CResult {
    let in_desc = &*input;
    let out_desc = &*out;

    match c_tensor_to_rust(in_desc) {
        Ok(tensor) => {
            let result = ops::relu(&tensor);
            rust_tensor_to_c(&result, out_desc.data);
            CResult::ok()
        }
        Err(e) => CResult::err(e),
    }
}

/// Sigmoid activation: out = 1 / (1 + exp(-input))
#[no_mangle]
pub unsafe extern "C" fn rusttorch_sigmoid(
    input: *const CTensorDescriptor,
    out: *const CTensorDescriptor,
) -> CResult {
    let in_desc = &*input;
    let out_desc = &*out;

    match c_tensor_to_rust(in_desc) {
        Ok(tensor) => {
            let result = ops::sigmoid(&tensor);
            rust_tensor_to_c(&result, out_desc.data);
            CResult::ok()
        }
        Err(e) => CResult::err(e),
    }
}

/// Tanh activation
#[no_mangle]
pub unsafe extern "C" fn rusttorch_tanh(
    input: *const CTensorDescriptor,
    out: *const CTensorDescriptor,
) -> CResult {
    let in_desc = &*input;
    let out_desc = &*out;

    match c_tensor_to_rust(in_desc) {
        Ok(tensor) => {
            let result = ops::tanh(&tensor);
            rust_tensor_to_c(&result, out_desc.data);
            CResult::ok()
        }
        Err(e) => CResult::err(e),
    }
}

/// GELU activation
#[no_mangle]
pub unsafe extern "C" fn rusttorch_gelu(
    input: *const CTensorDescriptor,
    out: *const CTensorDescriptor,
) -> CResult {
    let in_desc = &*input;
    let out_desc = &*out;

    match c_tensor_to_rust(in_desc) {
        Ok(tensor) => {
            let result = ops::gelu(&tensor);
            rust_tensor_to_c(&result, out_desc.data);
            CResult::ok()
        }
        Err(e) => CResult::err(e),
    }
}

/// Leaky ReLU activation
#[no_mangle]
pub unsafe extern "C" fn rusttorch_leaky_relu(
    input: *const CTensorDescriptor,
    out: *const CTensorDescriptor,
    alpha: f32,
) -> CResult {
    let in_desc = &*input;
    let out_desc = &*out;

    match c_tensor_to_rust(in_desc) {
        Ok(tensor) => {
            let result = ops::leaky_relu(&tensor, alpha);
            rust_tensor_to_c(&result, out_desc.data);
            CResult::ok()
        }
        Err(e) => CResult::err(e),
    }
}

// ============================================================================
// Reduction Operations
// ============================================================================

/// Global sum reduction
#[no_mangle]
pub unsafe extern "C" fn rusttorch_sum(
    input: *const CTensorDescriptor,
    out_value: *mut f64,
) -> CResult {
    let in_desc = &*input;

    match c_tensor_to_rust(in_desc) {
        Ok(tensor) => {
            let result = ops::sum(&tensor);
            *out_value = result;
            CResult::ok()
        }
        Err(e) => CResult::err(e),
    }
}

/// Global mean reduction
#[no_mangle]
pub unsafe extern "C" fn rusttorch_mean(
    input: *const CTensorDescriptor,
    out_value: *mut f64,
) -> CResult {
    let in_desc = &*input;

    match c_tensor_to_rust(in_desc) {
        Ok(tensor) => {
            let result = ops::mean(&tensor);
            *out_value = result;
            CResult::ok()
        }
        Err(e) => CResult::err(e),
    }
}

/// Global max reduction
#[no_mangle]
pub unsafe extern "C" fn rusttorch_max(
    input: *const CTensorDescriptor,
    out_value: *mut f64,
) -> CResult {
    let in_desc = &*input;

    match c_tensor_to_rust(in_desc) {
        Ok(tensor) => {
            let result = ops::max(&tensor);
            *out_value = result;
            CResult::ok()
        }
        Err(e) => CResult::err(e),
    }
}

/// Global min reduction
#[no_mangle]
pub unsafe extern "C" fn rusttorch_min(
    input: *const CTensorDescriptor,
    out_value: *mut f64,
) -> CResult {
    let in_desc = &*input;

    match c_tensor_to_rust(in_desc) {
        Ok(tensor) => {
            let result = ops::min(&tensor);
            *out_value = result;
            CResult::ok()
        }
        Err(e) => CResult::err(e),
    }
}

// ============================================================================
// Matrix Operations
// ============================================================================

/// Matrix multiplication: out = a @ b
#[no_mangle]
pub unsafe extern "C" fn rusttorch_matmul(
    a: *const CTensorDescriptor,
    b: *const CTensorDescriptor,
    out: *const CTensorDescriptor,
) -> CResult {
    let a_desc = &*a;
    let b_desc = &*b;
    let out_desc = &*out;

    match (c_tensor_to_rust(a_desc), c_tensor_to_rust(b_desc)) {
        (Ok(tensor_a), Ok(tensor_b)) => {
            match ops::matmul(&tensor_a, &tensor_b) {
                Ok(result) => {
                    rust_tensor_to_c(&result, out_desc.data);
                    CResult::ok()
                }
                Err(e) => CResult::err(crate::error::TensorError::Other {
                    message: e,
                }),
            }
        }
        (Err(e), _) | (_, Err(e)) => CResult::err(e),
    }
}

/// Transpose operation
#[no_mangle]
pub unsafe extern "C" fn rusttorch_transpose(
    input: *const CTensorDescriptor,
    out: *const CTensorDescriptor,
) -> CResult {
    let in_desc = &*input;
    let out_desc = &*out;

    match c_tensor_to_rust(in_desc) {
        Ok(tensor) => {
            let result = ops::transpose(&tensor);
            rust_tensor_to_c(&result, out_desc.data);
            CResult::ok()
        }
        Err(e) => CResult::err(e),
    }
}

// ============================================================================
// Simplified Float32-only API (for initial integration)
// ============================================================================

/// Simplified ReLU for contiguous float32 tensors
///
/// This is a simpler version for initial C++ integration.
/// Assumes contiguous row-major float32 data.
#[no_mangle]
pub unsafe extern "C" fn rusttorch_relu_f32(
    input: *const f32,
    output: *mut f32,
    size: usize,
) {
    let input_slice = std::slice::from_raw_parts(input, size);
    let output_slice = std::slice::from_raw_parts_mut(output, size);

    for (i, &val) in input_slice.iter().enumerate() {
        output_slice[i] = val.max(0.0);
    }
}

/// Simplified element-wise add for contiguous float32 tensors
#[no_mangle]
pub unsafe extern "C" fn rusttorch_add_f32(
    a: *const f32,
    b: *const f32,
    output: *mut f32,
    size: usize,
) {
    let a_slice = std::slice::from_raw_parts(a, size);
    let b_slice = std::slice::from_raw_parts(b, size);
    let output_slice = std::slice::from_raw_parts_mut(output, size);

    for i in 0..size {
        output_slice[i] = a_slice[i] + b_slice[i];
    }
}

/// Simplified element-wise mul for contiguous float32 tensors
#[no_mangle]
pub unsafe extern "C" fn rusttorch_mul_f32(
    a: *const f32,
    b: *const f32,
    output: *mut f32,
    size: usize,
) {
    let a_slice = std::slice::from_raw_parts(a, size);
    let b_slice = std::slice::from_raw_parts(b, size);
    let output_slice = std::slice::from_raw_parts_mut(output, size);

    for i in 0..size {
        output_slice[i] = a_slice[i] * b_slice[i];
    }
}
