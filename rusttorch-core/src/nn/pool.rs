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
