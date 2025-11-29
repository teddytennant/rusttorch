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
    fn test_storage_creation() {
        let storage: Storage<f32> = Storage::new(10);
        assert_eq!(storage.len(), 10);
        assert!(!storage.is_empty());
    }

    #[test]
    fn test_storage_from_vec() {
        let data = vec![1.0, 2.0, 3.0];
        let storage = Storage::from_vec(data);
        assert_eq!(storage.len(), 3);
        assert_eq!(storage.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_storage_clone() {
        let storage1 = Storage::from_vec(vec![1, 2, 3]);
        let storage2 = storage1.clone();
        assert_eq!(storage1.len(), storage2.len());
    }
}
