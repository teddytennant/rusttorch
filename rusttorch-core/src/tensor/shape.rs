//! Shape and stride utilities for tensors

/// Compute strides from shape (row-major/C-contiguous)
pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Compute the total number of elements from a shape
pub fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Check if a shape is valid
pub fn is_valid_shape(shape: &[usize]) -> bool {
    !shape.is_empty() && shape.iter().all(|&dim| dim > 0)
}

/// Broadcast two shapes to a common shape
pub fn broadcast_shapes(shape1: &[usize], shape2: &[usize]) -> Option<Vec<usize>> {
    let max_len = shape1.len().max(shape2.len());
    let mut result = vec![1; max_len];

    for i in 0..max_len {
        let dim1 = if i < shape1.len() {
            shape1[shape1.len() - 1 - i]
        } else {
            1
        };
        let dim2 = if i < shape2.len() {
            shape2[shape2.len() - 1 - i]
        } else {
            1
        };

        if dim1 == dim2 {
            result[max_len - 1 - i] = dim1;
        } else if dim1 == 1 {
            result[max_len - 1 - i] = dim2;
        } else if dim2 == 1 {
            result[max_len - 1 - i] = dim1;
        } else {
            return None; // Shapes not broadcastable
        }
    }

    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_strides() {
        assert_eq!(compute_strides(&[3, 4, 5]), vec![20, 5, 1]);
        assert_eq!(compute_strides(&[2, 3]), vec![3, 1]);
        assert_eq!(compute_strides(&[5]), vec![1]);
    }

    #[test]
    fn test_numel() {
        assert_eq!(numel(&[2, 3, 4]), 24);
        assert_eq!(numel(&[5]), 5);
        assert_eq!(numel(&[1, 1, 1]), 1);
    }

    #[test]
    fn test_is_valid_shape() {
        assert!(is_valid_shape(&[1, 2, 3]));
        assert!(is_valid_shape(&[5]));
        assert!(!is_valid_shape(&[]));
        assert!(!is_valid_shape(&[0, 1]));
    }

    #[test]
    fn test_broadcast_shapes() {
        assert_eq!(broadcast_shapes(&[3, 1], &[1, 4]), Some(vec![3, 4]));
        assert_eq!(broadcast_shapes(&[2, 3, 4], &[4]), Some(vec![2, 3, 4]));
        assert_eq!(broadcast_shapes(&[3, 5], &[3, 4]), None);
    }
}
