//! Zero-copy tensor views for PyTorch interop
//!
//! This module provides safe, zero-copy views into PyTorch tensor data.
//! Lifetimes ensure the data remains valid during operations.

use crate::DType;
use std::marker::PhantomData;
use std::slice;

/// A zero-copy view into tensor data with compile-time lifetime safety
///
/// This type provides safe access to PyTorch tensor data without copying.
/// The lifetime parameter `'a` ensures the underlying data remains valid.
///
/// # Safety
///
/// The borrow checker ensures:
/// - Data cannot be freed while view exists
/// - No mutable aliasing (if created as immutable)
/// - Compile-time memory safety guarantees
pub struct TensorView<'a, T> {
    /// Immutable slice view of the data
    data: &'a [T],
    /// Shape of the tensor (dimensions)
    shape: Vec<usize>,
    /// Strides for accessing elements
    strides: Vec<usize>,
    /// Data type
    dtype: DType,
    /// Phantom data for lifetime tracking
    _marker: PhantomData<&'a ()>,
}

/// Mutable zero-copy view into tensor data
pub struct TensorViewMut<'a, T> {
    /// Mutable slice view of the data
    data: &'a mut [T],
    /// Shape of the tensor (dimensions)
    shape: Vec<usize>,
    /// Strides for accessing elements
    strides: Vec<usize>,
    /// Data type
    dtype: DType,
    /// Phantom data for lifetime tracking
    _marker: PhantomData<&'a mut ()>,
}

impl<'a, T> TensorView<'a, T> {
    /// Create a zero-copy view from a raw pointer (UNSAFE)
    ///
    /// # Safety
    ///
    /// Caller must guarantee:
    /// - `ptr` points to valid memory for `len` elements
    /// - Data remains valid for lifetime `'a`
    /// - No mutable aliases exist
    /// - Pointer is properly aligned for type `T`
    /// - Shape and strides are consistent with data layout
    pub unsafe fn from_raw_parts(
        ptr: *const T,
        len: usize,
        shape: Vec<usize>,
        strides: Vec<usize>,
        dtype: DType,
    ) -> Self {
        let data = slice::from_raw_parts(ptr, len);
        Self {
            data,
            shape,
            strides,
            dtype,
            _marker: PhantomData,
        }
    }

    /// Get the shape of the tensor
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the strides of the tensor
    #[inline]
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Get the data type
    #[inline]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get the number of dimensions
    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Get total number of elements
    #[inline]
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get the underlying data as a slice (zero-copy)
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.data
    }

    /// Check if tensor is contiguous in memory
    pub fn is_contiguous(&self) -> bool {
        if self.ndim() == 0 {
            return true;
        }

        let mut expected_stride = 1;
        for i in (0..self.ndim()).rev() {
            if self.strides[i] != expected_stride {
                return false;
            }
            expected_stride *= self.shape[i];
        }
        true
    }

    /// Get element at index (panics if out of bounds or non-contiguous)
    pub fn get(&self, indices: &[usize]) -> Option<&T> {
        if indices.len() != self.ndim() {
            return None;
        }

        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.shape[i] {
                return None;
            }
        }

        // Calculate linear index using strides
        let linear_idx: usize = indices
            .iter()
            .zip(&self.strides)
            .map(|(&idx, &stride)| idx * stride)
            .sum();

        self.data.get(linear_idx)
    }
}

impl<'a, T> TensorViewMut<'a, T> {
    /// Create a mutable zero-copy view from a raw pointer (UNSAFE)
    ///
    /// # Safety
    ///
    /// Same requirements as `TensorView::from_raw_parts`, plus:
    /// - No other references (mutable or immutable) exist to this data
    pub unsafe fn from_raw_parts_mut(
        ptr: *mut T,
        len: usize,
        shape: Vec<usize>,
        strides: Vec<usize>,
        dtype: DType,
    ) -> Self {
        let data = slice::from_raw_parts_mut(ptr, len);
        Self {
            data,
            shape,
            strides,
            dtype,
            _marker: PhantomData,
        }
    }

    /// Get the shape of the tensor
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the strides of the tensor
    #[inline]
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Get the data type
    #[inline]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get the number of dimensions
    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Get total number of elements
    #[inline]
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get the underlying data as a mutable slice (zero-copy)
    #[inline]
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        self.data
    }

    /// Get immutable slice
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.data
    }

    /// Check if tensor is contiguous in memory
    pub fn is_contiguous(&self) -> bool {
        if self.ndim() == 0 {
            return true;
        }

        let mut expected_stride = 1;
        for i in (0..self.ndim()).rev() {
            if self.strides[i] != expected_stride {
                return false;
            }
            expected_stride *= self.shape[i];
        }
        true
    }
}

// Implement Send and Sync for parallel operations
unsafe impl<'a, T: Send> Send for TensorView<'a, T> {}
unsafe impl<'a, T: Sync> Sync for TensorView<'a, T> {}
unsafe impl<'a, T: Send> Send for TensorViewMut<'a, T> {}
unsafe impl<'a, T: Sync> Sync for TensorViewMut<'a, T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_view_creation() {
        let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];
        let strides = vec![3, 1]; // Row-major

        let view = unsafe {
            TensorView::from_raw_parts(
                data.as_ptr(),
                data.len(),
                shape.clone(),
                strides.clone(),
                DType::Float32,
            )
        };

        assert_eq!(view.shape(), &[2, 3]);
        assert_eq!(view.ndim(), 2);
        assert_eq!(view.numel(), 6);
        assert!(view.is_contiguous());
    }

    #[test]
    fn test_tensor_view_indexing() {
        let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];
        let strides = vec![3, 1];

        let view = unsafe {
            TensorView::from_raw_parts(data.as_ptr(), data.len(), shape, strides, DType::Float32)
        };

        assert_eq!(view.get(&[0, 0]), Some(&1.0));
        assert_eq!(view.get(&[0, 1]), Some(&2.0));
        assert_eq!(view.get(&[1, 2]), Some(&6.0));
        assert_eq!(view.get(&[2, 0]), None); // Out of bounds
    }

    #[test]
    fn test_mutable_view() {
        let mut data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let strides = vec![2, 1];

        let mut view = unsafe {
            TensorViewMut::from_raw_parts_mut(
                data.as_mut_ptr(),
                data.len(),
                shape,
                strides,
                DType::Float32,
            )
        };

        // Modify through view
        view.as_slice_mut()[0] = 10.0;

        assert_eq!(data[0], 10.0); // Original data modified
    }

    #[test]
    fn test_contiguous_check() {
        let data = vec![1.0_f32; 12];

        // Contiguous tensor
        let contiguous = unsafe {
            TensorView::from_raw_parts(
                data.as_ptr(),
                12,
                vec![3, 4],
                vec![4, 1], // Row-major
                DType::Float32,
            )
        };
        assert!(contiguous.is_contiguous());

        // Non-contiguous (transposed)
        let non_contiguous = unsafe {
            TensorView::from_raw_parts(
                data.as_ptr(),
                12,
                vec![4, 3],
                vec![1, 4], // Column-major
                DType::Float32,
            )
        };
        assert!(!non_contiguous.is_contiguous());
    }
}
