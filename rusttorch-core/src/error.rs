//! Error types for RustTorch operations
//!
//! This module defines the error types used throughout the RustTorch library.
//! All fallible operations should return `Result<T, TensorError>` instead of panicking.

use std::fmt;

/// The main error type for RustTorch operations
#[derive(Debug, Clone)]
pub enum TensorError {
    /// Shape mismatch between tensors
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
        context: String,
    },

    /// Data type mismatch
    DTypeMismatch {
        expected: String,
        actual: String,
        context: String,
    },

    /// Operation on empty tensor that requires non-empty data
    EmptyTensor {
        operation: String,
    },

    /// Invalid dimension index or size
    InvalidDimension {
        dimension: usize,
        max_dimension: usize,
        context: String,
    },

    /// Broadcasting error
    BroadcastError {
        shape_a: Vec<usize>,
        shape_b: Vec<usize>,
        reason: String,
    },

    /// Tensor size overflow (too many elements)
    SizeOverflow {
        dimensions: Vec<usize>,
    },

    /// Invalid argument or parameter
    InvalidArgument {
        parameter: String,
        reason: String,
    },

    /// Data validation error (e.g., predictions not in [0,1])
    ValidationError {
        field: String,
        constraint: String,
        actual: String,
    },

    /// I/O error when loading data
    IoError {
        message: String,
    },

    /// Parse error when loading data
    ParseError {
        line: usize,
        column: usize,
        message: String,
    },

    /// Generic error for cases not covered above
    Other {
        message: String,
    },
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TensorError::ShapeMismatch { expected, actual, context } => {
                write!(
                    f,
                    "Shape mismatch in {}: expected {:?}, got {:?}",
                    context, expected, actual
                )
            }
            TensorError::DTypeMismatch { expected, actual, context } => {
                write!(
                    f,
                    "Data type mismatch in {}: expected {}, got {}",
                    context, expected, actual
                )
            }
            TensorError::EmptyTensor { operation } => {
                write!(f, "Cannot perform {} on empty tensor", operation)
            }
            TensorError::InvalidDimension { dimension, max_dimension, context } => {
                write!(
                    f,
                    "Invalid dimension in {}: {} (max: {})",
                    context, dimension, max_dimension
                )
            }
            TensorError::BroadcastError { shape_a, shape_b, reason } => {
                write!(
                    f,
                    "Cannot broadcast shapes {:?} and {:?}: {}",
                    shape_a, shape_b, reason
                )
            }
            TensorError::SizeOverflow { dimensions } => {
                write!(
                    f,
                    "Tensor size overflow: dimensions {:?} would exceed maximum size",
                    dimensions
                )
            }
            TensorError::InvalidArgument { parameter, reason } => {
                write!(f, "Invalid argument '{}': {}", parameter, reason)
            }
            TensorError::ValidationError { field, constraint, actual } => {
                write!(
                    f,
                    "Validation error for '{}': expected {}, got {}",
                    field, constraint, actual
                )
            }
            TensorError::IoError { message } => {
                write!(f, "I/O error: {}", message)
            }
            TensorError::ParseError { line, column, message } => {
                write!(f, "Parse error at line {}, column {}: {}", line, column, message)
            }
            TensorError::Other { message } => {
                write!(f, "{}", message)
            }
        }
    }
}

impl std::error::Error for TensorError {}

/// Convert std::io::Error to TensorError
impl From<std::io::Error> for TensorError {
    fn from(error: std::io::Error) -> Self {
        TensorError::IoError {
            message: error.to_string(),
        }
    }
}

/// Convenience type alias for Results that use TensorError
pub type Result<T> = std::result::Result<T, TensorError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_mismatch_display() {
        let err = TensorError::ShapeMismatch {
            expected: vec![2, 3],
            actual: vec![3, 2],
            context: "matrix multiplication".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Shape mismatch"));
        assert!(msg.contains("matrix multiplication"));
    }

    #[test]
    fn test_empty_tensor_display() {
        let err = TensorError::EmptyTensor {
            operation: "mean".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("empty tensor"));
        assert!(msg.contains("mean"));
    }

    #[test]
    fn test_broadcast_error_display() {
        let err = TensorError::BroadcastError {
            shape_a: vec![2, 3],
            shape_b: vec![4, 5],
            reason: "incompatible shapes".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("broadcast"));
        assert!(msg.contains("incompatible"));
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let tensor_err: TensorError = io_err.into();
        match tensor_err {
            TensorError::IoError { message } => {
                assert!(message.contains("file not found"));
            }
            _ => panic!("Expected IoError variant"),
        }
    }
}
