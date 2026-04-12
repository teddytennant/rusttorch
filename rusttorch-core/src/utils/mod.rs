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
    fn test_integers() {
        assert_eq!(slice_to_string(&[1, 2, 3]), "1, 2, 3");
    }

    #[test]
    fn test_single_element() {
        assert_eq!(slice_to_string(&[42]), "42");
    }

    #[test]
    fn test_empty_slice() {
        let empty: [i32; 0] = [];
        assert_eq!(slice_to_string(&empty), "");
    }

    #[test]
    fn test_floats() {
        assert_eq!(slice_to_string(&[1.5_f32, 2.5, 3.5]), "1.5, 2.5, 3.5");
    }

    #[test]
    fn test_negative_integers() {
        assert_eq!(slice_to_string(&[-1, 0, 1]), "-1, 0, 1");
    }

    #[test]
    fn test_strings() {
        let words: Vec<String> = vec!["hello".into(), "world".into()];
        assert_eq!(slice_to_string(&words), "hello, world");
    }

    #[test]
    fn test_usize_shape_style() {
        assert_eq!(slice_to_string(&[2_usize, 3, 4]), "2, 3, 4");
    }

    #[test]
    fn test_large_slice() {
        let v: Vec<i32> = (0..5).collect();
        assert_eq!(slice_to_string(&v), "0, 1, 2, 3, 4");
    }

    #[test]
    fn test_bool_slice() {
        assert_eq!(slice_to_string(&[true, false, true]), "true, false, true");
    }

    #[test]
    fn test_char_slice() {
        assert_eq!(slice_to_string(&['a', 'b', 'c']), "a, b, c");
    }

    #[test]
    fn test_no_trailing_separator() {
        let s = slice_to_string(&[1, 2, 3]);
        assert!(!s.ends_with(','));
        assert!(!s.ends_with(", "));
    }
}
