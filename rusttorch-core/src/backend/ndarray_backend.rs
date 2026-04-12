//! CPU backend implemented on top of ndarray.
//!
//! This is a thin adapter over the existing op implementations; it does
//! not change data layout or introduce a new storage type. The value it
//! provides is a stable implementation of the [`Backend`] trait so that
//! future GPU backends have a reference to compare against.

use super::Backend;
use crate::autograd::Variable;
use crate::error::Result;
use crate::tensor::{Device, Tensor};

/// CPU execution target backed by ndarray. Zero-sized today — all state
/// lives on the `Tensor`s it operates over.
#[derive(Debug, Clone, Copy, Default)]
pub struct NdArrayBackend;

impl NdArrayBackend {
    /// Create a new CPU backend handle. Free — no state.
    pub const fn new() -> Self {
        NdArrayBackend
    }
}

impl Backend for NdArrayBackend {
    fn device(&self) -> Device {
        Device::Cpu
    }

    fn matmul(&self, a: &Variable, b: &Variable) -> Result<Variable> {
        a.matmul(b)
    }

    fn add(&self, a: &Variable, b: &Variable) -> Result<Variable> {
        a.add(b)
    }

    fn mul(&self, a: &Variable, b: &Variable) -> Result<Variable> {
        a.mul(b)
    }

    fn relu(&self, x: &Variable) -> Variable {
        x.relu()
    }

    fn gelu(&self, x: &Variable) -> Variable {
        x.gelu()
    }

    fn log_softmax(&self, x: &Variable, dim: usize) -> Result<Variable> {
        x.log_softmax(dim)
    }

    fn layer_norm(
        &self,
        x: &Variable,
        norm_size: usize,
        weight: Option<&Variable>,
        bias: Option<&Variable>,
        eps: f32,
    ) -> Result<Variable> {
        x.layer_norm(norm_size, weight, bias, eps)
    }

    fn scalar(&self, value: f32) -> Tensor {
        Tensor::scalar(value)
    }
}
