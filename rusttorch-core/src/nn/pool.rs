//! Pooling layers.

use crate::autograd::Variable;
use crate::error::Result;
use crate::nn::module::Module;
use crate::nn::parameter::Parameter;

/// 2D max pooling layer.
///
/// Applies max pooling over input with shape [B, C, H, W].
/// Output shape: [B, C, oH, oW] where:
///   oH = (H - kernel_size) / stride + 1
///   oW = (W - kernel_size) / stride + 1
pub struct MaxPool2d {
    pub kernel_size: usize,
    pub stride: usize,
}

impl MaxPool2d {
    /// Create a new MaxPool2d layer. Stride defaults to kernel_size.
    pub fn new(kernel_size: usize) -> Self {
        MaxPool2d {
            kernel_size,
            stride: kernel_size,
        }
    }

    /// Create MaxPool2d with custom stride.
    pub fn with_stride(kernel_size: usize, stride: usize) -> Self {
        MaxPool2d {
            kernel_size,
            stride,
        }
    }
}

impl Module for MaxPool2d {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        crate::autograd::ops::max_pool2d_forward(input, self.kernel_size, self.stride)
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![] // No learnable parameters
    }
}

impl std::fmt::Debug for MaxPool2d {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "MaxPool2d(kernel_size={}, stride={})",
            self.kernel_size, self.stride
        )
    }
}

/// 2D average pooling layer.
///
/// Applies average pooling over input with shape [B, C, H, W].
/// Output shape: [B, C, oH, oW] where:
///   oH = (H - kernel_size) / stride + 1
///   oW = (W - kernel_size) / stride + 1
pub struct AvgPool2d {
    pub kernel_size: usize,
    pub stride: usize,
}

impl AvgPool2d {
    /// Create a new AvgPool2d layer. Stride defaults to kernel_size.
    pub fn new(kernel_size: usize) -> Self {
        AvgPool2d {
            kernel_size,
            stride: kernel_size,
        }
    }

    /// Create AvgPool2d with custom stride.
    pub fn with_stride(kernel_size: usize, stride: usize) -> Self {
        AvgPool2d {
            kernel_size,
            stride,
        }
    }
}

impl Module for AvgPool2d {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        crate::autograd::ops::avg_pool2d_forward(input, self.kernel_size, self.stride)
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![]
    }
}

impl std::fmt::Debug for AvgPool2d {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "AvgPool2d(kernel_size={}, stride={})",
            self.kernel_size, self.stride
        )
    }
}

/// Adaptive 2D average pooling layer.
///
/// Maps any spatial input size to a fixed output size by dynamically
/// computing kernel windows. Essential for architectures like ResNet
/// where global average pooling maps arbitrary spatial dims to 1x1.
///
/// Input shape: [B, C, H, W] (any H, W)
/// Output shape: [B, C, output_h, output_w]
pub struct AdaptiveAvgPool2d {
    pub output_h: usize,
    pub output_w: usize,
}

impl AdaptiveAvgPool2d {
    /// Create a new AdaptiveAvgPool2d layer with target output size.
    pub fn new(output_size: usize) -> Self {
        AdaptiveAvgPool2d {
            output_h: output_size,
            output_w: output_size,
        }
    }

    /// Create with different height and width targets.
    pub fn new_rect(output_h: usize, output_w: usize) -> Self {
        AdaptiveAvgPool2d { output_h, output_w }
    }
}

impl Module for AdaptiveAvgPool2d {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        crate::autograd::ops::adaptive_avg_pool2d_forward(input, self.output_h, self.output_w)
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![]
    }
}

impl std::fmt::Debug for AdaptiveAvgPool2d {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "AdaptiveAvgPool2d(output_size=({}, {}))",
            self.output_h, self.output_w
        )
    }
}
