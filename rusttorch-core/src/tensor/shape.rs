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
    use proptest::prelude::*;

    // ---- compute_strides ----

    #[test]
    fn test_compute_strides_3d() {
        assert_eq!(compute_strides(&[3, 4, 5]), vec![20, 5, 1]);
    }

    #[test]
    fn test_compute_strides_2d() {
        assert_eq!(compute_strides(&[2, 3]), vec![3, 1]);
    }

    #[test]
    fn test_compute_strides_1d() {
        assert_eq!(compute_strides(&[5]), vec![1]);
    }

    #[test]
    fn test_compute_strides_singleton_dims() {
        assert_eq!(compute_strides(&[1, 1, 4]), vec![4, 4, 1]);
    }

    #[test]
    fn test_compute_strides_4d_nchw() {
        // Standard NCHW layout: (N=2, C=3, H=4, W=5)
        assert_eq!(compute_strides(&[2, 3, 4, 5]), vec![60, 20, 5, 1]);
    }

    #[test]
    fn test_compute_strides_stride_product_property() {
        // For contiguous row-major: strides[i] * shape[i] should equal strides[i-1]
        let shape = [7, 11, 13, 17];
        let strides = compute_strides(&shape);
        for i in 1..shape.len() {
            assert_eq!(strides[i - 1], strides[i] * shape[i]);
        }
        assert_eq!(*strides.last().unwrap(), 1);
    }

    // ---- numel ----

    #[test]
    fn test_numel_3d() {
        assert_eq!(numel(&[2, 3, 4]), 24);
    }

    #[test]
    fn test_numel_1d() {
        assert_eq!(numel(&[5]), 5);
    }

    #[test]
    fn test_numel_all_ones() {
        assert_eq!(numel(&[1, 1, 1]), 1);
    }

    #[test]
    fn test_numel_empty_shape_is_one() {
        // The conventional definition: empty product = 1 (a scalar).
        assert_eq!(numel(&[]), 1);
    }

    #[test]
    fn test_numel_with_zero_dim_is_zero() {
        assert_eq!(numel(&[2, 0, 3]), 0);
    }

    // ---- is_valid_shape ----

    #[test]
    fn test_is_valid_shape_normal() {
        assert!(is_valid_shape(&[1, 2, 3]));
        assert!(is_valid_shape(&[5]));
    }

    #[test]
    fn test_is_valid_shape_empty_rejected() {
        assert!(!is_valid_shape(&[]));
    }

    #[test]
    fn test_is_valid_shape_zero_dim_rejected() {
        assert!(!is_valid_shape(&[0, 1]));
        assert!(!is_valid_shape(&[3, 0]));
        assert!(!is_valid_shape(&[1, 2, 0, 4]));
    }

    // ---- broadcast_shapes ----

    #[test]
    fn test_broadcast_shapes_same_rank() {
        assert_eq!(broadcast_shapes(&[3, 1], &[1, 4]), Some(vec![3, 4]));
    }

    #[test]
    fn test_broadcast_shapes_unequal_rank() {
        assert_eq!(broadcast_shapes(&[2, 3, 4], &[4]), Some(vec![2, 3, 4]));
    }

    #[test]
    fn test_broadcast_shapes_incompatible() {
        assert_eq!(broadcast_shapes(&[3, 5], &[3, 4]), None);
    }

    #[test]
    fn test_broadcast_shapes_scalar_to_tensor() {
        // A scalar (empty shape) broadcasts to any shape's identity row.
        assert_eq!(broadcast_shapes(&[], &[2, 3]), Some(vec![2, 3]));
    }

    #[test]
    fn test_broadcast_shapes_identical() {
        assert_eq!(broadcast_shapes(&[5, 4, 3], &[5, 4, 3]), Some(vec![5, 4, 3]));
    }

    #[test]
    fn test_broadcast_shapes_both_ones_promote() {
        assert_eq!(broadcast_shapes(&[1, 1], &[4, 5]), Some(vec![4, 5]));
    }

    #[test]
    fn test_broadcast_shapes_symmetric() {
        // Broadcast rules are commutative.
        let a = [2, 1, 5];
        let b = [3, 5];
        assert_eq!(broadcast_shapes(&a, &b), broadcast_shapes(&b, &a));
    }

    // ---- Property-based tests (fulfills RUSTTORCH_TODO Track B) ----
    proptest! {
        /// Strides for row-major: last stride is always 1; prefix product matches numel.
        #[test]
        fn prop_compute_strides_last_is_one_and_prefix_product(shape in prop::collection::vec(1usize..64, 1..5)) {
            let strides = compute_strides(&shape);
            prop_assert_eq!(*strides.last().unwrap(), 1);
            let n = numel(&shape);
            if !shape.is_empty() {
                prop_assert_eq!(strides[0] * shape[0], n);
            }
        }

        /// Broadcast is symmetric when valid: broadcast(a, b) == broadcast(b, a).
        #[test]
        fn prop_broadcast_symmetric(
            a in prop::collection::vec(1usize..32, 0..4),
            b in prop::collection::vec(1usize..32, 0..4)
        ) {
            let r1 = broadcast_shapes(&a, &b);
            let r2 = broadcast_shapes(&b, &a);
            prop_assert_eq!(r1, r2);
        }

        /// Valid broadcast result has max rank and each dim is max or the non-1 value.
        #[test]
        fn prop_broadcast_result_dims_valid(
            a in prop::collection::vec(1usize..16, 1..4),
            b in prop::collection::vec(1usize..16, 1..4)
        ) {
            if let Some(r) = broadcast_shapes(&a, &b) {
                prop_assert!(r.len() <= a.len().max(b.len()));
                // Spot check: every dim in r is >= corresponding in a/b (or 1-padded)
                for (i, &rd) in r.iter().rev().enumerate() {
                    let da = if i < a.len() { a[a.len()-1-i] } else { 1 };
                    let db = if i < b.len() { b[b.len()-1-i] } else { 1 };
                    let expected = da.max(db);
                    prop_assert_eq!(rd, expected);
                }
            }
        }
    }
}
