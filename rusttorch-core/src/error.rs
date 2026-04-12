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
    EmptyTensor { operation: String },

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
    SizeOverflow { dimensions: Vec<usize> },

    /// Invalid argument or parameter
    InvalidArgument { parameter: String, reason: String },

    /// Data validation error (e.g., predictions not in [0,1])
    ValidationError {
        field: String,
        constraint: String,
        actual: String,
    },

    /// I/O error when loading data
    IoError { message: String },

    /// Parse error when loading data
    ParseError {
        line: usize,
        column: usize,
        message: String,
    },

    /// Generic error for cases not covered above
    Other { message: String },
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TensorError::ShapeMismatch {
                expected,
                actual,
                context,
            } => {
                write!(
                    f,
                    "Shape mismatch in {}: expected {:?}, got {:?}",
                    context, expected, actual
                )
            }
            TensorError::DTypeMismatch {
                expected,
                actual,
                context,
            } => {
                write!(
                    f,
                    "Data type mismatch in {}: expected {}, got {}",
                    context, expected, actual
                )
            }
            TensorError::EmptyTensor { operation } => {
                write!(f, "Cannot perform {} on empty tensor", operation)
            }
            TensorError::InvalidDimension {
                dimension,
                max_dimension,
                context,
            } => {
                write!(
                    f,
                    "Invalid dimension in {}: {} (max: {})",
                    context, dimension, max_dimension
                )
            }
            TensorError::BroadcastError {
                shape_a,
                shape_b,
                reason,
            } => {
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
            TensorError::ValidationError {
                field,
                constraint,
                actual,
            } => {
                write!(
                    f,
                    "Validation error for '{}': expected {}, got {}",
                    field, constraint, actual
                )
            }
            TensorError::IoError { message } => {
                write!(f, "I/O error: {}", message)
            }
            TensorError::ParseError {
                line,
                column,
                message,
            } => {
                write!(
                    f,
                    "Parse error at line {}, column {}: {}",
                    line, column, message
                )
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
        assert!(msg.contains("[2, 3]"));
        assert!(msg.contains("[3, 2]"));
    }

    #[test]
    fn test_dtype_mismatch_display() {
        let err = TensorError::DTypeMismatch {
            expected: "float32".to_string(),
            actual: "int64".to_string(),
            context: "add".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Data type mismatch"));
        assert!(msg.contains("float32"));
        assert!(msg.contains("int64"));
        assert!(msg.contains("add"));
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
    fn test_invalid_dimension_display() {
        let err = TensorError::InvalidDimension {
            dimension: 5,
            max_dimension: 3,
            context: "reshape".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Invalid dimension"));
        assert!(msg.contains("reshape"));
        assert!(msg.contains('5'));
        assert!(msg.contains('3'));
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
    fn test_size_overflow_display() {
        let err = TensorError::SizeOverflow {
            dimensions: vec![usize::MAX, 2],
        };
        let msg = format!("{}", err);
        assert!(msg.contains("size overflow"));
    }

    #[test]
    fn test_invalid_argument_display() {
        let err = TensorError::InvalidArgument {
            parameter: "lr".to_string(),
            reason: "must be positive".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("lr"));
        assert!(msg.contains("must be positive"));
    }

    #[test]
    fn test_validation_error_display() {
        let err = TensorError::ValidationError {
            field: "probs".to_string(),
            constraint: "within [0, 1]".to_string(),
            actual: "1.5".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("probs"));
        assert!(msg.contains("within [0, 1]"));
        assert!(msg.contains("1.5"));
    }

    #[test]
    fn test_io_error_display() {
        let err = TensorError::IoError {
            message: "disk full".to_string(),
        };
        assert!(format!("{}", err).contains("disk full"));
    }

    #[test]
    fn test_parse_error_display() {
        let err = TensorError::ParseError {
            line: 42,
            column: 7,
            message: "bad token".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("42"));
        assert!(msg.contains('7'));
        assert!(msg.contains("bad token"));
    }

    #[test]
    fn test_other_display() {
        let err = TensorError::Other {
            message: "something broke".to_string(),
        };
        assert_eq!(format!("{}", err), "something broke");
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

    #[test]
    fn test_error_is_std_error() {
        // Ensure TensorError implements std::error::Error.
        fn assert_std_error<E: std::error::Error>(_: &E) {}
        let err = TensorError::Other {
            message: "test".to_string(),
        };
        assert_std_error(&err);
    }

    #[test]
    fn test_error_is_clone_and_debug() {
        let err = TensorError::ShapeMismatch {
            expected: vec![1, 2],
            actual: vec![3, 4],
            context: "test".to_string(),
        };
        let cloned = err.clone();
        assert_eq!(format!("{err:?}"), format!("{cloned:?}"));
    }

    #[test]
    fn test_result_type_alias() {
        fn returns_ok() -> Result<i32> {
            Ok(42)
        }
        fn returns_err() -> Result<i32> {
            Err(TensorError::Other {
                message: "fail".to_string(),
            })
        }
        assert_eq!(returns_ok().unwrap(), 42);
        assert!(returns_err().is_err());
    }
}
