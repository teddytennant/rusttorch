//! Utility functions and helpers

/// Convert a slice to a comma-separated string
pub fn slice_to_string<T: std::fmt::Display>(slice: &[T]) -> String {
    slice
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slice_to_string() {
        assert_eq!(slice_to_string(&[1, 2, 3]), "1, 2, 3");
    }
}
