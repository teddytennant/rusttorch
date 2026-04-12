//! GroupNorm (Wu & He 2018) — normalize per-sample, per-group.
//!
//! GroupNorm splits the channel dimension into `num_groups` groups and
//! normalizes each group's elements independently. It behaves like
//! LayerNorm when `num_groups=1` and like InstanceNorm when
//! `num_groups=num_channels`.
//!
//! Unlike BatchNorm, GroupNorm's statistics are per-sample, so it
//! works well with small batches (batch=1 is fine).
//!
//! Input must be 3-D `[N, C, L]`. For 4-D `[N, C, H, W]` convolutional
//! features, reshape to `[N, C, H*W]` before passing in, and reshape
//! back afterwards.

use crate::autograd::Variable;
use crate::error::Result;
use crate::nn::module::Module;
use crate::nn::parameter::Parameter;
use crate::tensor::{DType, Tensor};

/// Group normalization module.
pub struct GroupNorm {
    pub weight: Parameter,
    pub bias: Parameter,
    pub num_groups: usize,
    pub num_channels: usize,
    pub eps: f32,
}

impl GroupNorm {
    /// Create a new GroupNorm with learnable per-channel affine params.
    ///
    /// Panics if `num_channels` is not divisible by `num_groups`.
    pub fn new(num_groups: usize, num_channels: usize) -> Self {
        assert!(
            num_groups > 0 && num_channels.is_multiple_of(num_groups),
            "num_groups ({num_groups}) must divide num_channels ({num_channels})"
        );
        let weight = Parameter::new(
            Tensor::ones(&[num_channels], DType::Float32),
            "weight",
        );
        let bias = Parameter::zeros(&[num_channels], "bias");
        GroupNorm {
            weight,
            bias,
            num_groups,
            num_channels,
            eps: 1e-5,
        }
    }

    /// Create with a custom epsilon.
    pub fn with_eps(num_groups: usize, num_channels: usize, eps: f32) -> Self {
        let mut gn = Self::new(num_groups, num_channels);
        gn.eps = eps;
        gn
    }
}

impl Module for GroupNorm {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        input.group_norm(
            self.num_groups,
            Some(self.weight.var()),
            Some(self.bias.var()),
            self.eps,
        )
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}

impl std::fmt::Debug for GroupNorm {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "GroupNorm(num_groups={}, num_channels={}, eps={})",
            self.num_groups, self.num_channels, self.eps
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_norm_construction() {
        let gn = GroupNorm::new(4, 16);
        assert_eq!(gn.num_groups, 4);
        assert_eq!(gn.num_channels, 16);
        assert_eq!(gn.weight.shape(), vec![16]);
        assert_eq!(gn.bias.shape(), vec![16]);
    }

    #[test]
    #[should_panic(expected = "must divide")]
    fn group_norm_panics_on_bad_groups() {
        let _ = GroupNorm::new(3, 8);
    }

    #[test]
    fn group_norm_parameters_return_weight_and_bias() {
        let gn = GroupNorm::new(2, 8);
        let params = gn.parameters();
        assert_eq!(params.len(), 2);
    }

    #[test]
    fn group_norm_forward_unit_variance_zero_mean_per_group() {
        // With weight=1, bias=0, each group should have mean~0, var~1.
        let gn = GroupNorm::new(2, 4);
        let data = (0..4 * 3).map(|i| i as f32 * 0.3).collect::<Vec<_>>();
        // [N=1, C=4, L=3]
        let x = Variable::new(Tensor::from_vec(data, &[1, 4, 3]), false);
        let y = gn.forward(&x).unwrap().tensor().to_vec_f32();

        // Group 0 = channels 0..2, group 1 = channels 2..4.
        for g in 0..2 {
            let start = g * 2 * 3;
            let end = start + 2 * 3;
            let slice = &y[start..end];
            let mean: f32 = slice.iter().sum::<f32>() / slice.len() as f32;
            let var: f32 = slice.iter().map(|v| (v - mean).powi(2)).sum::<f32>()
                / slice.len() as f32;
            assert!(mean.abs() < 1e-5, "group {g} mean should be 0, got {mean}");
            assert!(
                (var - 1.0).abs() < 1e-3,
                "group {g} var should be 1, got {var}"
            );
        }
    }

    #[test]
    fn group_norm_rejects_non_3d_input() {
        let gn = GroupNorm::new(1, 4);
        let x = Variable::new(Tensor::from_vec(vec![0.0; 8], &[2, 4]), false);
        assert!(gn.forward(&x).is_err());
    }

    #[test]
    fn group_norm_layernorm_equivalence_when_groups_is_one() {
        // GroupNorm with num_groups=1 should produce the same result as
        // LayerNorm over C*L for each sample.
        let gn = GroupNorm::new(1, 4);
        let x = Variable::new(
            Tensor::from_vec(
                vec![
                    1.0, 2.0, 3.0, -1.0, -2.0, 0.5, //
                    0.5, -0.5, 2.0, 1.5, -3.0, 1.0,
                ],
                &[1, 4, 3],
            ),
            false,
        );
        let gn_out = gn.forward(&x).unwrap().tensor().to_vec_f32();

        // Manually verify mean ≈ 0 and variance ≈ 1 across the whole sample.
        let mean: f32 = gn_out.iter().sum::<f32>() / gn_out.len() as f32;
        let var: f32 = gn_out.iter().map(|v| (v - mean).powi(2)).sum::<f32>()
            / gn_out.len() as f32;
        assert!(mean.abs() < 1e-5);
        assert!((var - 1.0).abs() < 1e-3);
    }

    #[test]
    fn group_norm_instance_norm_equivalence_when_groups_equals_channels() {
        // GroupNorm(C, C) = InstanceNorm: normalizes each channel
        // independently per sample.
        let gn = GroupNorm::new(4, 4);
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let x = Variable::new(Tensor::from_vec(data, &[1, 4, 3]), false);
        let y = gn.forward(&x).unwrap().tensor().to_vec_f32();

        // Each channel (4 channels, each 3 elements) should be zero-mean
        // unit-variance after normalization.
        for c in 0..4 {
            let slice = &y[c * 3..c * 3 + 3];
            let mean: f32 = slice.iter().sum::<f32>() / 3.0;
            assert!(mean.abs() < 1e-5, "channel {c} mean = {mean}");
        }
    }

    #[test]
    fn group_norm_debug_format() {
        let gn = GroupNorm::new(8, 32);
        let s = format!("{:?}", gn);
        assert!(s.contains("GroupNorm"));
        assert!(s.contains("num_groups=8"));
        assert!(s.contains("num_channels=32"));
    }
}
