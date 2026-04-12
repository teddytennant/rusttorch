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
    fn test_dtype_size_f32() {
        assert_eq!(DType::Float32.size_bytes(), std::mem::size_of::<f32>());
    }

    #[test]
    fn test_dtype_size_f64() {
        assert_eq!(DType::Float64.size_bytes(), std::mem::size_of::<f64>());
    }

    #[test]
    fn test_dtype_size_i32() {
        assert_eq!(DType::Int32.size_bytes(), std::mem::size_of::<i32>());
    }

    #[test]
    fn test_dtype_size_i64() {
        assert_eq!(DType::Int64.size_bytes(), std::mem::size_of::<i64>());
    }

    #[test]
    fn test_is_float_positive() {
        assert!(DType::Float32.is_float());
        assert!(DType::Float64.is_float());
    }

    #[test]
    fn test_is_float_negative() {
        assert!(!DType::Int32.is_float());
        assert!(!DType::Int64.is_float());
    }

    #[test]
    fn test_is_int_positive() {
        assert!(DType::Int32.is_int());
        assert!(DType::Int64.is_int());
    }

    #[test]
    fn test_is_int_negative() {
        assert!(!DType::Float32.is_int());
        assert!(!DType::Float64.is_int());
    }

    #[test]
    fn test_float_and_int_are_disjoint() {
        for dt in [
            DType::Float32,
            DType::Float64,
            DType::Int32,
            DType::Int64,
        ] {
            assert!(dt.is_float() ^ dt.is_int(), "{dt:?} should be exactly one");
        }
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", DType::Float32), "float32");
        assert_eq!(format!("{}", DType::Float64), "float64");
        assert_eq!(format!("{}", DType::Int32), "int32");
        assert_eq!(format!("{}", DType::Int64), "int64");
    }

    #[test]
    fn test_equality_and_copy() {
        let a = DType::Float32;
        let b = a; // Copy
        assert_eq!(a, b);
        assert_ne!(DType::Float32, DType::Float64);
    }

    #[test]
    fn test_debug_format() {
        assert_eq!(format!("{:?}", DType::Float32), "Float32");
        assert_eq!(format!("{:?}", DType::Int64), "Int64");
    }

    #[test]
    fn test_size_matches_native_types() {
        // Guard against someone changing the enum without updating size_bytes
        assert_eq!(DType::Float32.size_bytes() * 2, DType::Float64.size_bytes());
        assert_eq!(DType::Int32.size_bytes() * 2, DType::Int64.size_bytes());
    }
}
