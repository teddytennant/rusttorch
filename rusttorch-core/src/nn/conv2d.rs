//! Conv2d — 2D convolution layer.

use crate::autograd::Variable;
use crate::error::Result;
use crate::nn::module::Module;
use crate::nn::parameter::Parameter;

/// 2D convolution layer.
///
/// Applies a 2D convolution over input with shape [B, C_in, H, W].
/// Output shape: [B, C_out, oH, oW] where:
///   oH = (H + 2*padding - kernel_size) / stride + 1
///   oW = (W + 2*padding - kernel_size) / stride + 1
pub struct Conv2d {
    pub weight: Parameter,
    pub bias: Option<Parameter>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
}

impl Conv2d {
    /// Create a new Conv2d layer with Kaiming uniform weight init.
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        let fan_in = in_channels * kernel_size * kernel_size;
        let weight = Parameter::kaiming_uniform(
            &[out_channels, in_channels, kernel_size, kernel_size],
            fan_in,
            "weight",
        );
        let bound = 1.0 / (fan_in as f32).sqrt();
        let bias = Some(Parameter::uniform(&[out_channels], bound, "bias"));

        Conv2d {
            weight,
            bias,
            in_channels,
            out_channels,
            kernel_size,
            stride: 1,
            padding: 0,
        }
    }

    /// Create Conv2d with custom stride and padding.
    pub fn with_options(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
        let fan_in = in_channels * kernel_size * kernel_size;
        let weight = Parameter::kaiming_uniform(
            &[out_channels, in_channels, kernel_size, kernel_size],
            fan_in,
            "weight",
        );
        let bound = 1.0 / (fan_in as f32).sqrt();
        let bias = Some(Parameter::uniform(&[out_channels], bound, "bias"));

        Conv2d {
            weight,
            bias,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
        }
    }
}

impl Module for Conv2d {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        crate::autograd::ops::conv2d_forward(
            input,
            self.weight.var(),
            self.bias.as_ref().map(|b| b.var()),
            self.stride,
            self.padding,
        )
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }
}

impl std::fmt::Debug for Conv2d {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "Conv2d(in={}, out={}, kernel={}, stride={}, padding={}, bias={})",
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.bias.is_some()
        )
    }
}
