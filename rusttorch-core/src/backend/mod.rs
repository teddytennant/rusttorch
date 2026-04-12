//! Backend abstraction — the contract every tensor execution target must
//! implement.
//!
//! Today the only backend is [`NdArrayBackend`] (CPU via ndarray). The
//! goal of this module is to nail down the *interface* before CUDA work
//! starts, so that:
//!
//! 1. Client code can already be written device-agnostically.
//! 2. When cudarc + hand-written kernels land, they fit into a stable
//!    surface rather than reshaping the whole library.
//!
//! ## Why delegate to `Tensor`?
//!
//! For the first iteration the trait methods accept and return
//! `Tensor`s rather than a typed `Storage` handle. That keeps the
//! refactor small: `NdArrayBackend` simply forwards to the existing
//! ops, with no data-layout changes. A later pass will hoist the
//! per-backend storage behind an associated type so GPU buffers don't
//! have to round-trip through the ndarray `TensorData` enum.
//!
//! ## Kernel surface
//!
//! The method set here is the minimum needed to run GPT-2 inference
//! end-to-end on a backend:
//!
//! - dense ops: `matmul`
//! - elementwise: `add`, `mul`
//! - activations: `relu`, `gelu`
//! - attention pieces: `softmax`, `layer_norm`
//! - embedding: `embedding_lookup` (not yet wired, see GPU_KERNELS.md)
//!
//! Additional kernels (conv2d, batchnorm, dropout, reductions) will be
//! added as the GPU side matures.

use crate::autograd::Variable;
use crate::error::Result;
use crate::tensor::{Device, Tensor};

pub mod ndarray_backend;
pub use ndarray_backend::NdArrayBackend;

/// A backend is a concrete implementation of the tensor kernel surface
/// on a particular device.
///
/// Every method takes and returns `Tensor`s. For CPU that's free; for
/// CUDA it will imply staging data through a device-side storage type,
/// which will be added as an associated type in a follow-up pass.
pub trait Backend {
    /// The device this backend executes on.
    fn device(&self) -> Device;

    // ---- Dense linear algebra ----

    /// Matrix multiply. Operates on the autograd-tracked `Variable` so
    /// callers can opt into graph recording; implementations that do not
    /// need to participate in autograd should just dispatch to the
    /// tensor-level kernel and wrap the result in `Variable::detach`.
    fn matmul(&self, a: &Variable, b: &Variable) -> Result<Variable>;

    // ---- Elementwise ----

    fn add(&self, a: &Variable, b: &Variable) -> Result<Variable>;
    fn mul(&self, a: &Variable, b: &Variable) -> Result<Variable>;

    // ---- Activations ----

    fn relu(&self, x: &Variable) -> Variable;
    fn gelu(&self, x: &Variable) -> Variable;

    // ---- Transformer primitives ----

    /// Numerically stable log-softmax along `dim`. Used in attention and
    /// in cross-entropy loss.
    fn log_softmax(&self, x: &Variable, dim: usize) -> Result<Variable>;

    /// LayerNorm over the last `norm_size` elements. `weight` and `bias`
    /// are optional; `eps` is the usual small constant.
    fn layer_norm(
        &self,
        x: &Variable,
        norm_size: usize,
        weight: Option<&Variable>,
        bias: Option<&Variable>,
        eps: f32,
    ) -> Result<Variable>;

    // ---- Materialization ----

    /// Create a scalar f32 tensor on this backend's device.
    fn scalar(&self, value: f32) -> Tensor;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ndarray_backend_reports_cpu_device() {
        let b = NdArrayBackend::new();
        assert_eq!(b.device(), Device::Cpu);
    }

    #[test]
    fn ndarray_backend_matmul_matches_direct_call() {
        let b = NdArrayBackend::new();
        let a = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]), false);
        let m = Variable::new(Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]), false);
        let out = b.matmul(&a, &m).unwrap();
        assert_eq!(out.tensor().to_vec_f32(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn ndarray_backend_add_mul_relu_round_trip() {
        let b = NdArrayBackend::new();
        let x = Variable::new(Tensor::from_vec(vec![-1.0, 0.5, 2.0], &[3]), false);
        let y = Variable::new(Tensor::from_vec(vec![1.0, 1.0, 1.0], &[3]), false);
        let sum = b.add(&x, &y).unwrap();
        assert_eq!(sum.tensor().to_vec_f32(), vec![0.0, 1.5, 3.0]);
        let prod = b.mul(&x, &y).unwrap();
        assert_eq!(prod.tensor().to_vec_f32(), vec![-1.0, 0.5, 2.0]);
        let r = b.relu(&x);
        assert_eq!(r.tensor().to_vec_f32(), vec![0.0, 0.5, 2.0]);
    }

    #[test]
    fn ndarray_backend_scalar_lives_on_cpu() {
        let b = NdArrayBackend::new();
        let s = b.scalar(3.5);
        assert_eq!(s.device(), Device::Cpu);
        assert_eq!(s.to_vec_f32(), vec![3.5]);
    }
}
