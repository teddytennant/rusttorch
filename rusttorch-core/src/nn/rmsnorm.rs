//! RMSNorm — Root Mean Square Layer Normalization.
//!
//! Introduced in Zhang & Sennrich 2019, and made famous by Llama and most
//! modern language models. Cheaper than LayerNorm (skips mean subtraction
//! and has no bias) and empirically equally effective.
//!
//! ```text
//! y = x * (1 / sqrt(mean(x^2) + eps)) * weight
//! ```
//!
//! Weight (gamma) is initialized to 1. There is no bias parameter.

use crate::autograd::Variable;
use crate::error::Result;
use crate::nn::module::Module;
use crate::nn::parameter::Parameter;
use crate::tensor::{DType, Tensor};

/// Root-mean-square layer normalization module.
pub struct RmsNorm {
    pub weight: Parameter,
    pub normalized_shape: usize,
    pub eps: f32,
}

impl RmsNorm {
    /// Create a new RmsNorm that normalizes over the last `normalized_shape` elements.
    pub fn new(normalized_shape: usize) -> Self {
        let weight = Parameter::new(
            Tensor::ones(&[normalized_shape], DType::Float32),
            "weight",
        );
        RmsNorm {
            weight,
            normalized_shape,
            eps: 1e-6,
        }
    }

    /// Create with custom epsilon.
    pub fn with_eps(normalized_shape: usize, eps: f32) -> Self {
        let mut n = Self::new(normalized_shape);
        n.eps = eps;
        n
    }
}

impl Module for RmsNorm {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        input.rms_norm(self.normalized_shape, Some(self.weight.var()), self.eps)
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![self.weight.clone()]
    }
}

impl std::fmt::Debug for RmsNorm {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "RmsNorm(normalized_shape={}, eps={})",
            self.normalized_shape, self.eps
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_default_weight_is_ones() {
        let rms = RmsNorm::new(4);
        assert_eq!(rms.weight.tensor().to_vec_f32(), vec![1.0, 1.0, 1.0, 1.0]);
        assert_eq!(rms.normalized_shape, 4);
        assert!((rms.eps - 1e-6).abs() < 1e-12);
    }

    #[test]
    fn rms_norm_with_eps_overrides_default() {
        let rms = RmsNorm::with_eps(8, 1e-5);
        assert!((rms.eps - 1e-5).abs() < 1e-12);
    }

    #[test]
    fn rms_norm_parameters_contain_weight_only() {
        let rms = RmsNorm::new(16);
        let params = rms.parameters();
        assert_eq!(params.len(), 1, "RMSNorm should have weight only, no bias");
        assert_eq!(params[0].shape(), vec![16]);
    }

    #[test]
    fn rms_norm_forward_with_default_weight_ones() {
        // With weight=1s, RMSNorm(x) should produce x / rms(x).
        let rms = RmsNorm::new(3);
        let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 2.0], &[1, 3]), false);
        let y = rms.forward(&x).unwrap();
        let got = y.tensor().to_vec_f32();

        let mean_sq = (1.0 + 4.0 + 4.0) / 3.0;
        let inv_rms = 1.0 / (mean_sq + 1e-6_f32).sqrt();
        let expected = [1.0 * inv_rms, 2.0 * inv_rms, 2.0 * inv_rms];
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-5, "got {g}, want {e}");
        }
    }

    #[test]
    fn rms_norm_forward_shape_preserved() {
        let rms = RmsNorm::new(5);
        let x = Variable::new(
            Tensor::from_vec((0..20).map(|i| i as f32 * 0.1).collect(), &[4, 5]),
            false,
        );
        let y = rms.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![4, 5]);
    }

    #[test]
    fn rms_norm_invalid_size_errors() {
        let rms = RmsNorm::new(3);
        // Tensor size 7 is not divisible by norm_size 3.
        let x = Variable::new(Tensor::from_vec(vec![0.0; 7], &[7]), false);
        assert!(rms.forward(&x).is_err());
    }

    #[test]
    fn rms_norm_debug_format() {
        let rms = RmsNorm::new(12);
        let s = format!("{:?}", rms);
        assert!(s.contains("RmsNorm"));
        assert!(s.contains("12"));
    }
}
