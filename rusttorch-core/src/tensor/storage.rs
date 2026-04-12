//! Memory storage for tensors
//!
//! This module handles the low-level memory management for tensors,
//! including allocation, deallocation, and reference counting.

use std::sync::Arc;

/// Reference-counted storage for tensor data
#[derive(Debug)]
pub struct Storage<T> {
    data: Arc<Vec<T>>,
}

impl<T> Storage<T> {
    /// Create a new storage with the given capacity
    pub fn new(capacity: usize) -> Self
    where
        T: Default + Clone,
    {
        Self {
            data: Arc::new(vec![T::default(); capacity]),
        }
    }

    /// Create storage from existing data
    pub fn from_vec(data: Vec<T>) -> Self {
        Self {
            data: Arc::new(data),
        }
    }

    /// Get the number of elements
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if storage is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get a reference to the underlying data
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }
}

impl<T: Clone> Clone for Storage<T> {
    fn clone(&self) -> Self {
        Self {
            data: Arc::clone(&self.data),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_new_zero_initialized() {
        let storage: Storage<f32> = Storage::new(10);
        assert_eq!(storage.len(), 10);
        for &x in storage.as_slice() {
            assert_eq!(x, 0.0);
        }
    }

    #[test]
    fn test_storage_new_nonempty_is_not_empty() {
        let storage: Storage<f32> = Storage::new(1);
        assert!(!storage.is_empty());
    }

    #[test]
    fn test_storage_new_zero_capacity_is_empty() {
        let storage: Storage<f32> = Storage::new(0);
        assert_eq!(storage.len(), 0);
        assert!(storage.is_empty());
    }

    #[test]
    fn test_storage_from_vec_preserves_data() {
        let data = vec![1.0_f32, 2.0, 3.0];
        let storage = Storage::from_vec(data);
        assert_eq!(storage.len(), 3);
        assert_eq!(storage.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_storage_from_empty_vec() {
        let storage: Storage<f32> = Storage::from_vec(Vec::new());
        assert_eq!(storage.len(), 0);
        assert!(storage.is_empty());
    }

    #[test]
    fn test_storage_clone_is_cheap_arc_share() {
        // Clone should share the underlying Arc, not copy the vector.
        let storage1 = Storage::from_vec(vec![1_i32; 1000]);
        let storage2 = storage1.clone();
        assert_eq!(storage1.len(), storage2.len());
        // Same underlying pointer means Arc shared rather than cloned.
        assert_eq!(
            storage1.as_slice().as_ptr(),
            storage2.as_slice().as_ptr()
        );
    }

    #[test]
    fn test_storage_generic_over_type() {
        let s_u8: Storage<u8> = Storage::from_vec(vec![0, 1, 2, 3]);
        assert_eq!(s_u8.len(), 4);
        let s_i64: Storage<i64> = Storage::from_vec(vec![-1, -2]);
        assert_eq!(s_i64.as_slice(), &[-1, -2]);
    }

    #[test]
    fn test_storage_as_slice_matches_length() {
        let storage = Storage::from_vec(vec![1.0_f64; 7]);
        assert_eq!(storage.as_slice().len(), storage.len());
    }

    #[test]
    fn test_storage_debug_format() {
        let storage = Storage::from_vec(vec![1_i32, 2, 3]);
        let dbg = format!("{:?}", storage);
        assert!(dbg.contains("Storage"));
    }

    #[test]
    fn test_storage_independent_instances() {
        // Different Storages constructed independently should be disjoint.
        let s1 = Storage::from_vec(vec![1.0_f32, 2.0]);
        let s2 = Storage::from_vec(vec![1.0_f32, 2.0]);
        // Same contents, but different Arc allocations.
        assert_ne!(s1.as_slice().as_ptr(), s2.as_slice().as_ptr());
        assert_eq!(s1.as_slice(), s2.as_slice());
    }

    #[test]
    fn test_storage_default_type_constraint() {
        // Storage::new requires T: Default + Clone.
        let s: Storage<u32> = Storage::new(5);
        assert_eq!(s.as_slice(), &[0, 0, 0, 0, 0]);
    }
}
