//! Data type definitions for tensors

/// Supported tensor data types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    /// 32-bit floating point
    Float32,
    /// 64-bit floating point
    Float64,
    /// 32-bit signed integer
    Int32,
    /// 64-bit signed integer
    Int64,
}

impl DType {
    /// Get the size of this dtype in bytes
    pub fn size_bytes(&self) -> usize {
        match self {
            DType::Float32 => 4,
            DType::Float64 => 8,
            DType::Int32 => 4,
            DType::Int64 => 8,
        }
    }

    /// Check if this is a floating-point type
    pub fn is_float(&self) -> bool {
        matches!(self, DType::Float32 | DType::Float64)
    }

    /// Check if this is an integer type
    pub fn is_int(&self) -> bool {
        matches!(self, DType::Int32 | DType::Int64)
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DType::Float32 => write!(f, "float32"),
            DType::Float64 => write!(f, "float64"),
            DType::Int32 => write!(f, "int32"),
            DType::Int64 => write!(f, "int64"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_size() {
        assert_eq!(DType::Float32.size_bytes(), 4);
        assert_eq!(DType::Float64.size_bytes(), 8);
        assert_eq!(DType::Int32.size_bytes(), 4);
        assert_eq!(DType::Int64.size_bytes(), 8);
    }

    #[test]
    fn test_dtype_predicates() {
        assert!(DType::Float32.is_float());
        assert!(!DType::Float32.is_int());
        assert!(DType::Int64.is_int());
        assert!(!DType::Int64.is_float());
    }
}
