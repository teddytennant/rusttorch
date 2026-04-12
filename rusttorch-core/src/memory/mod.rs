//! Memory management utilities for tensor data.
//!
//! Two building blocks:
//!
//! 1. [`AlignedBuffer<T>`] — owned, heap-allocated buffer with an
//!    arbitrary power-of-two alignment. Used for SIMD-friendly
//!    storage (64-byte alignment lets AVX-512 loads/stores run on
//!    any element), and as the future drop-in for device-memory
//!    allocations when the GPU backend arrives.
//!
//! 2. [`BumpArena`] — a single-threaded bump allocator that hands
//!    out sequentially-placed `AlignedBuffer`s until the arena is
//!    reset. This targets the "many short-lived temporaries inside
//!    one training step" pattern: during a forward/backward pass
//!    the framework allocates a lot of activation/gradient tensors
//!    that all die at the end of the step. A bump arena turns the
//!    N per-step `malloc`/`free` pairs into one reset.
//!
//! The global per-allocator free-lists inside libc's malloc are
//! already fast, so the main reason to prefer a pool is predictability
//! (no fragmentation) and alignment control, not raw speed on tiny
//! allocations. See the `bench_memory` criterion suite for measured
//! numbers on your machine.

use std::alloc::{self, Layout};
use std::cell::UnsafeCell;
use std::marker::PhantomData;
use std::ptr::NonNull;

/// Default alignment used by [`AlignedBuffer::new`] and [`allocate_aligned`].
/// 64 bytes is the AVX-512 cache-line width and the widest SIMD vector
/// register on x86-64 today. Any alignment that is a multiple of 64 also
/// satisfies AVX2 (32-byte) and SSE (16-byte) alignment requirements.
pub const DEFAULT_ALIGN: usize = 64;

/// Allocate a zero-initialized `Vec<u8>` of the requested size. Alignment is
/// not guaranteed beyond `std::mem::align_of::<u8>() == 1`, which is fine for
/// raw byte storage but not for SIMD loads. Prefer [`AlignedBuffer`] when
/// alignment matters.
///
/// This function is kept for backward compatibility with existing callers;
/// new code should reach for [`AlignedBuffer`].
pub fn allocate_aligned(size: usize, _alignment: usize) -> Vec<u8> {
    vec![0; size]
}

/// An owned, heap-allocated buffer of `T` aligned to a power-of-two byte
/// boundary. Memory is zero-initialized.
///
/// Alignment is enforced at construction via [`std::alloc::alloc_zeroed`],
/// so the resulting pointer is guaranteed to be at least `align_bytes`-aligned
/// regardless of `std::mem::align_of::<T>()`.
///
/// # Invariants
///
/// - `ptr` is non-null and points to `len` valid, zero-initialized `T`s.
/// - `align_bytes` is a power of two and ≥ `std::mem::align_of::<T>()`.
/// - The allocation layout used for deallocation matches the one used for
///   allocation (same size, same alignment).
pub struct AlignedBuffer<T> {
    ptr: NonNull<T>,
    len: usize,
    align_bytes: usize,
    _marker: PhantomData<T>,
}

impl<T> AlignedBuffer<T> {
    /// Allocate a new buffer of `len` elements with the default 64-byte
    /// alignment. Contents are zero-initialized.
    ///
    /// Returns `None` if `len == 0` (zero-size allocations aren't useful
    /// and would require distinguishing a sentinel pointer from real memory).
    pub fn new(len: usize) -> Option<Self> {
        Self::with_alignment(len, DEFAULT_ALIGN)
    }

    /// Allocate a new buffer with a custom alignment. The alignment must be
    /// a power of two; otherwise this returns `None`. Contents are
    /// zero-initialized.
    pub fn with_alignment(len: usize, align_bytes: usize) -> Option<Self> {
        if len == 0 || !align_bytes.is_power_of_two() {
            return None;
        }
        let elem_size = std::mem::size_of::<T>();
        if elem_size == 0 {
            return None;
        }
        let effective_align = align_bytes.max(std::mem::align_of::<T>());
        let byte_size = len.checked_mul(elem_size)?;
        let layout = Layout::from_size_align(byte_size, effective_align).ok()?;

        // SAFETY: layout has non-zero size (len > 0 and elem_size > 0 above),
        // and `alloc_zeroed` is guaranteed to either return a valid,
        // properly-aligned pointer to zeroed memory or null on OOM.
        let raw = unsafe { alloc::alloc_zeroed(layout) };
        let ptr = NonNull::new(raw as *mut T)?;
        Some(Self {
            ptr,
            len,
            align_bytes: effective_align,
            _marker: PhantomData,
        })
    }

    /// Number of elements in the buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the buffer has no elements. Currently always false
    /// because `new` refuses to create empty buffers; kept for clippy.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Alignment in bytes of the underlying allocation.
    #[inline]
    pub fn alignment(&self) -> usize {
        self.align_bytes
    }

    /// Immutable slice over the buffer's contents.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: `ptr` is valid for `len` elements, properly aligned, and
        // the buffer cannot be aliased mutably while this borrow is live.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Mutable slice over the buffer's contents.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: `&mut self` grants unique access for the duration of the
        // returned borrow; the slice is in-bounds of the allocation.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Raw pointer to the first element. Unsafe to dereference beyond
    /// `len` elements.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Mutable raw pointer to the first element.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }
}

impl<T> Drop for AlignedBuffer<T> {
    fn drop(&mut self) {
        // SAFETY: layout matches what `with_alignment` used for allocation.
        let byte_size = self.len * std::mem::size_of::<T>();
        if byte_size == 0 {
            return;
        }
        let layout = Layout::from_size_align(byte_size, self.align_bytes)
            .expect("valid layout was used for allocation");
        unsafe {
            alloc::dealloc(self.ptr.as_ptr() as *mut u8, layout);
        }
    }
}

// SAFETY: `AlignedBuffer<T>` owns its allocation; it's `Send` whenever `T` is
// `Send`, and `Sync` whenever `T` is `Sync`, just like `Box<[T]>`.
unsafe impl<T: Send> Send for AlignedBuffer<T> {}
unsafe impl<T: Sync> Sync for AlignedBuffer<T> {}

impl<T: std::fmt::Debug> std::fmt::Debug for AlignedBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AlignedBuffer")
            .field("len", &self.len)
            .field("align_bytes", &self.align_bytes)
            .finish()
    }
}

/// A single-threaded bump-allocation arena for `f32` scratch storage.
///
/// Hands out sequentially-placed slices from a single large
/// [`AlignedBuffer<f32>`]. All slices are invalidated on [`reset`], which
/// returns the high-water mark to zero. This matches the "allocate a bunch
/// of activation/gradient tensors during a forward-backward step, free them
/// all at the end" pattern.
///
/// [`reset`]: BumpArena::reset
///
/// # Thread safety
///
/// Not `Sync`: the arena uses interior mutability without synchronization.
/// Create one per thread, or wrap in a `Mutex` if you really must share.
pub struct BumpArena {
    buf: UnsafeCell<AlignedBuffer<f32>>,
    offset: UnsafeCell<usize>,
}

impl BumpArena {
    /// Create a new arena with capacity for `capacity` f32 elements,
    /// allocated with 64-byte alignment. Returns `None` if the underlying
    /// allocation fails or `capacity == 0`.
    pub fn new(capacity: usize) -> Option<Self> {
        let buf = AlignedBuffer::<f32>::new(capacity)?;
        Some(Self {
            buf: UnsafeCell::new(buf),
            offset: UnsafeCell::new(0),
        })
    }

    /// Total capacity of the arena in elements.
    pub fn capacity(&self) -> usize {
        // SAFETY: we only read `len`; no references are handed out.
        unsafe { (*self.buf.get()).len() }
    }

    /// Current high-water mark in elements — the number of elements handed
    /// out since the last [`reset`].
    ///
    /// [`reset`]: BumpArena::reset
    pub fn used(&self) -> usize {
        unsafe { *self.offset.get() }
    }

    /// Remaining capacity in elements.
    pub fn remaining(&self) -> usize {
        self.capacity() - self.used()
    }

    /// Allocate a zero-initialized slice of `n` f32s from the arena.
    /// Returns `None` if the arena doesn't have enough remaining capacity.
    ///
    /// # Safety contract for the caller
    ///
    /// The returned slice is valid until the next call to [`reset`] on
    /// this arena. Holding the slice across a `reset` invocation is UB.
    /// Because `alloc` takes `&self` (not `&mut self`) we cannot enforce
    /// this via the borrow checker.
    ///
    /// [`reset`]: BumpArena::reset
    #[allow(clippy::mut_from_ref)] // interior mutability is the whole point
    pub fn alloc(&self, n: usize) -> Option<&mut [f32]> {
        if n == 0 {
            return Some(&mut []);
        }
        // SAFETY: not `Sync`, so only one thread at a time can touch these
        // UnsafeCells. We uphold the invariant that each call hands out a
        // disjoint sub-slice of the backing buffer.
        unsafe {
            let offset = *self.offset.get();
            let buf = &mut *self.buf.get();
            let new_offset = offset.checked_add(n)?;
            if new_offset > buf.len() {
                return None;
            }
            *self.offset.get() = new_offset;
            let base = buf.as_mut_ptr().add(offset);
            // Zero out the freshly-exposed region so callers see a clean slate.
            std::ptr::write_bytes(base, 0, n);
            Some(std::slice::from_raw_parts_mut(base, n))
        }
    }

    /// Reset the arena, invalidating every slice previously handed out by
    /// [`alloc`]. The underlying allocation is kept for reuse.
    ///
    /// [`alloc`]: BumpArena::alloc
    pub fn reset(&mut self) {
        // SAFETY: &mut self ensures no live borrows into the arena remain.
        unsafe {
            *self.offset.get() = 0;
        }
    }
}

// Not Sync, but Send is fine: transferring ownership across threads is OK
// as long as it's one thread at a time.
unsafe impl Send for BumpArena {}

impl std::fmt::Debug for BumpArena {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BumpArena")
            .field("capacity", &self.capacity())
            .field("used", &self.used())
            .finish()
    }
}

/// Legacy name kept for source compatibility. Prefer [`BumpArena`].
pub struct MemoryPool {
    inner: BumpArena,
}

impl MemoryPool {
    /// Create a new pool with a 4 MiB default scratch arena.
    pub fn new() -> Self {
        Self {
            inner: BumpArena::new(1024 * 1024).expect("default pool allocation"),
        }
    }

    /// Create a pool backed by an arena of the given f32 capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: BumpArena::new(capacity).expect("pool allocation"),
        }
    }

    /// Delegated to the inner [`BumpArena`].
    #[allow(clippy::mut_from_ref)]
    pub fn alloc(&self, n: usize) -> Option<&mut [f32]> {
        self.inner.alloc(n)
    }

    /// Delegated to the inner [`BumpArena`].
    pub fn reset(&mut self) {
        self.inner.reset();
    }

    /// Delegated to the inner [`BumpArena`].
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Delegated to the inner [`BumpArena`].
    pub fn used(&self) -> usize {
        self.inner.used()
    }
}

impl Default for MemoryPool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- AlignedBuffer ----

    #[test]
    fn aligned_buffer_default_alignment() {
        let buf = AlignedBuffer::<f32>::new(16).unwrap();
        assert_eq!(buf.len(), 16);
        assert!(buf.alignment() >= DEFAULT_ALIGN);
        let ptr = buf.as_ptr() as usize;
        assert_eq!(ptr % DEFAULT_ALIGN, 0, "ptr {ptr:#x} not 64-aligned");
    }

    #[test]
    fn aligned_buffer_zeroed() {
        let buf = AlignedBuffer::<f32>::new(32).unwrap();
        for &x in buf.as_slice() {
            assert_eq!(x, 0.0);
        }
    }

    #[test]
    fn aligned_buffer_mutable() {
        let mut buf = AlignedBuffer::<f32>::new(4).unwrap();
        for (i, slot) in buf.as_mut_slice().iter_mut().enumerate() {
            *slot = i as f32;
        }
        assert_eq!(buf.as_slice(), &[0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn aligned_buffer_rejects_zero_len() {
        assert!(AlignedBuffer::<f32>::new(0).is_none());
    }

    #[test]
    fn aligned_buffer_rejects_non_power_of_two_alignment() {
        assert!(AlignedBuffer::<f32>::with_alignment(8, 48).is_none());
    }

    #[test]
    fn aligned_buffer_honors_larger_alignment() {
        let buf = AlignedBuffer::<f32>::with_alignment(4, 128).unwrap();
        assert_eq!(buf.alignment(), 128);
        assert_eq!(buf.as_ptr() as usize % 128, 0);
    }

    #[test]
    fn aligned_buffer_minimum_type_alignment_respected() {
        // Requesting align=1 on an f64 should still produce an 8-byte-aligned
        // pointer because align_of::<f64>() == 8.
        let buf = AlignedBuffer::<f64>::with_alignment(2, 1).unwrap();
        assert!(buf.alignment() >= std::mem::align_of::<f64>());
        assert_eq!(buf.as_ptr() as usize % std::mem::align_of::<f64>(), 0);
    }

    #[test]
    fn aligned_buffer_debug() {
        let buf = AlignedBuffer::<f32>::new(8).unwrap();
        let s = format!("{buf:?}");
        assert!(s.contains("AlignedBuffer"));
        assert!(s.contains("len"));
    }

    // ---- BumpArena ----

    #[test]
    fn bump_arena_sequential_allocs_are_disjoint() {
        let arena = BumpArena::new(32).unwrap();
        let a = arena.alloc(8).unwrap();
        let a_ptr = a.as_ptr();
        let b = arena.alloc(8).unwrap();
        let b_ptr = b.as_ptr();
        // The two sub-slices should be adjacent but non-overlapping.
        assert_eq!(
            unsafe { b_ptr.offset_from(a_ptr) },
            8,
            "bump arena should hand out contiguous slices"
        );
    }

    #[test]
    fn bump_arena_alloc_is_zeroed() {
        let mut arena = BumpArena::new(64).unwrap();
        // Write garbage, reset, and re-allocate — the re-exposed region
        // should be zeroed again.
        let s = arena.alloc(16).unwrap();
        for slot in s.iter_mut() {
            *slot = 42.0;
        }
        arena.reset();
        let s2 = arena.alloc(16).unwrap();
        for &x in s2.iter() {
            assert_eq!(x, 0.0, "arena should re-zero reused memory");
        }
    }

    #[test]
    fn bump_arena_out_of_capacity() {
        let arena = BumpArena::new(4).unwrap();
        assert!(arena.alloc(8).is_none());
        assert_eq!(arena.used(), 0);
    }

    #[test]
    fn bump_arena_zero_len_alloc_is_empty_slice() {
        let arena = BumpArena::new(4).unwrap();
        let s = arena.alloc(0).unwrap();
        assert_eq!(s.len(), 0);
        assert_eq!(arena.used(), 0);
    }

    #[test]
    fn bump_arena_reset_returns_capacity() {
        let mut arena = BumpArena::new(16).unwrap();
        let _ = arena.alloc(8).unwrap();
        assert_eq!(arena.used(), 8);
        arena.reset();
        assert_eq!(arena.used(), 0);
        assert_eq!(arena.remaining(), 16);
    }

    #[test]
    fn bump_arena_capacity_accessor() {
        let arena = BumpArena::new(128).unwrap();
        assert_eq!(arena.capacity(), 128);
        assert_eq!(arena.remaining(), 128);
    }

    #[test]
    fn bump_arena_debug() {
        let arena = BumpArena::new(8).unwrap();
        let _ = arena.alloc(3).unwrap();
        let s = format!("{arena:?}");
        assert!(s.contains("BumpArena"));
        assert!(s.contains("capacity"));
        assert!(s.contains("used"));
    }

    // ---- Legacy MemoryPool wrapper ----

    #[test]
    fn memory_pool_default_has_capacity() {
        let pool = MemoryPool::default();
        assert!(pool.capacity() > 0);
        assert_eq!(pool.used(), 0);
    }

    #[test]
    fn memory_pool_with_custom_capacity() {
        let mut pool = MemoryPool::with_capacity(256);
        let s = pool.alloc(64).unwrap();
        assert_eq!(s.len(), 64);
        assert_eq!(pool.used(), 64);
        pool.reset();
        assert_eq!(pool.used(), 0);
    }

    #[test]
    fn allocate_aligned_backward_compat() {
        let buf = allocate_aligned(128, DEFAULT_ALIGN);
        assert_eq!(buf.len(), 128);
        for b in buf {
            assert_eq!(b, 0);
        }
    }
}
