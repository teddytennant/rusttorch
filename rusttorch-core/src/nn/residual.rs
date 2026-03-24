//! Residual (skip connection) blocks for building ResNet-style architectures.
//!
//! A residual block computes `output = relu(x + F(x))` where F is typically
//! two conv layers with batch normalization. The skip connection enables
//! training of much deeper networks by providing gradient shortcuts.

use crate::autograd::Variable;
use crate::error::Result;
use crate::nn::batchnorm::BatchNorm2d;
use crate::nn::conv2d::Conv2d;
use crate::nn::module::Module;
use crate::nn::parameter::Parameter;
use crate::nn::state_dict::StateDict;

/// A basic residual block: two 3x3 convolutions with batch normalization
/// and a skip connection.
///
/// ```text
///     x ─────────────────────── identity
///     │                              │
///     ├── Conv2d(3x3) ──┐           │
///     │                  │           │
///     ├── BatchNorm2d ───┤           │
///     │                  │           │
///     ├── ReLU ──────────┤           │
///     │                  │           │
///     ├── Conv2d(3x3) ──┤           │
///     │                  │           │
///     ├── BatchNorm2d ───┤           │
///     │                  │           │
///     └── + ◄────────────┘───────────┘
///         │
///         ReLU
///         │
///       output
/// ```
///
/// When `in_channels != out_channels` or `stride != 1`, a 1x1 downsample
/// convolution + batchnorm is applied to the identity path to match dimensions.
pub struct ResidualBlock {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    conv2: Conv2d,
    bn2: BatchNorm2d,
    /// Optional 1x1 downsample for the skip connection when dimensions change.
    downsample_conv: Option<Conv2d>,
    downsample_bn: Option<BatchNorm2d>,
    stride: usize,
}

impl ResidualBlock {
    /// Create a basic residual block.
    ///
    /// # Arguments
    /// * `in_channels` — number of input feature channels
    /// * `out_channels` — number of output feature channels
    /// * `stride` — stride for the first convolution (2 for downsampling)
    pub fn new(in_channels: usize, out_channels: usize, stride: usize) -> Self {
        let conv1 = Conv2d::with_options(in_channels, out_channels, 3, stride, 1);
        let bn1 = BatchNorm2d::new(out_channels);
        let conv2 = Conv2d::with_options(out_channels, out_channels, 3, 1, 1);
        let bn2 = BatchNorm2d::new(out_channels);

        let (downsample_conv, downsample_bn) = if stride != 1 || in_channels != out_channels {
            (
                Some(Conv2d::with_options(
                    in_channels,
                    out_channels,
                    1,
                    stride,
                    0,
                )),
                Some(BatchNorm2d::new(out_channels)),
            )
        } else {
            (None, None)
        };

        ResidualBlock {
            conv1,
            bn1,
            conv2,
            bn2,
            downsample_conv,
            downsample_bn,
            stride,
        }
    }

    /// Set training mode for all batch norm layers.
    pub fn train(&self) {
        self.bn1.train();
        self.bn2.train();
        if let Some(ref bn) = self.downsample_bn {
            bn.train();
        }
    }

    /// Set eval mode for all batch norm layers.
    pub fn eval(&self) {
        self.bn1.eval();
        self.bn2.eval();
        if let Some(ref bn) = self.downsample_bn {
            bn.eval();
        }
    }
}

impl Module for ResidualBlock {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        // Main path: conv1 -> bn1 -> relu -> conv2 -> bn2
        let out = self.conv1.forward(input)?;
        let out = self.bn1.forward(&out)?;
        let out = out.relu();
        let out = self.conv2.forward(&out)?;
        let out = self.bn2.forward(&out)?;

        // Skip connection: identity or downsample
        let identity =
            if let (Some(ref conv), Some(ref bn)) = (&self.downsample_conv, &self.downsample_bn) {
                let x = conv.forward(input)?;
                bn.forward(&x)?
            } else {
                input.clone()
            };

        // Residual addition + activation
        let out = out.add(&identity)?;
        Ok(out.relu())
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters());
        params.extend(self.bn1.parameters());
        params.extend(self.conv2.parameters());
        params.extend(self.bn2.parameters());
        if let Some(ref conv) = self.downsample_conv {
            params.extend(conv.parameters());
        }
        if let Some(ref bn) = self.downsample_bn {
            params.extend(bn.parameters());
        }
        params
    }

    fn state_dict(&self) -> StateDict {
        let mut sd = StateDict::new();
        sd.merge_prefixed("conv1", &self.conv1.state_dict());
        sd.merge_prefixed("bn1", &self.bn1.state_dict());
        sd.merge_prefixed("conv2", &self.conv2.state_dict());
        sd.merge_prefixed("bn2", &self.bn2.state_dict());
        if let Some(ref conv) = self.downsample_conv {
            sd.merge_prefixed("downsample.0", &conv.state_dict());
        }
        if let Some(ref bn) = self.downsample_bn {
            sd.merge_prefixed("downsample.1", &bn.state_dict());
        }
        sd
    }

    fn load_state_dict(&self, state_dict: &StateDict) {
        self.conv1.load_state_dict(&state_dict.sub_dict("conv1"));
        self.bn1.load_state_dict(&state_dict.sub_dict("bn1"));
        self.conv2.load_state_dict(&state_dict.sub_dict("conv2"));
        self.bn2.load_state_dict(&state_dict.sub_dict("bn2"));
        if let Some(ref conv) = self.downsample_conv {
            conv.load_state_dict(&state_dict.sub_dict("downsample.0"));
        }
        if let Some(ref bn) = self.downsample_bn {
            bn.load_state_dict(&state_dict.sub_dict("downsample.1"));
        }
    }
}

impl std::fmt::Debug for ResidualBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "ResidualBlock(stride={}, downsample={})",
            self.stride,
            self.downsample_conv.is_some()
        )
    }
}
