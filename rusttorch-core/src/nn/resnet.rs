//! ResNet architectures for image classification.
//!
//! Implements ResNet-18 and ResNet-34 variants. The CIFAR-10 variant uses a 3x3 initial
//! convolution (instead of 7x7 + maxpool for ImageNet) since the input is 32x32.
//!
//! # Architecture (ResNet-18 for CIFAR-10)
//!
//! ```text
//! Input [B, 3, 32, 32]
//!   → Conv2d(3, 64, k=3, s=1, p=1) → BN → ReLU           [B, 64, 32, 32]
//!   → Layer1: 2× ResBlock(64, 64, s=1)                     [B, 64, 32, 32]
//!   → Layer2: ResBlock(64, 128, s=2) + ResBlock(128, s=1)   [B, 128, 16, 16]
//!   → Layer3: ResBlock(128, 256, s=2) + ResBlock(256, s=1)  [B, 256, 8, 8]
//!   → Layer4: ResBlock(256, 512, s=2) + ResBlock(512, s=1)  [B, 512, 4, 4]
//!   → AdaptiveAvgPool2d(1)                                  [B, 512, 1, 1]
//!   → Flatten                                               [B, 512]
//!   → Linear(512, num_classes)                              [B, 10]
//! ```

use crate::autograd::Variable;
use crate::error::Result;
use crate::nn::batchnorm::BatchNorm2d;
use crate::nn::conv2d::Conv2d;
use crate::nn::flatten::Flatten;
use crate::nn::linear::Linear;
use crate::nn::module::Module;
use crate::nn::parameter::Parameter;
use crate::nn::pool::AdaptiveAvgPool2d;
use crate::nn::residual::ResidualBlock;
use crate::nn::state_dict::StateDict;

/// ResNet model with configurable depth and number of classes.
pub struct ResNet {
    /// Initial convolution
    conv1: Conv2d,
    bn1: BatchNorm2d,
    /// 4 layer groups, each containing multiple residual blocks
    layer1: Vec<ResidualBlock>,
    layer2: Vec<ResidualBlock>,
    layer3: Vec<ResidualBlock>,
    layer4: Vec<ResidualBlock>,
    /// Global average pooling → flatten → classifier
    avgpool: AdaptiveAvgPool2d,
    flatten: Flatten,
    fc: Linear,
}

impl ResNet {
    /// Create a ResNet with the given block counts per layer group.
    ///
    /// # Arguments
    /// * `layers` — number of residual blocks in each of the 4 groups
    /// * `num_classes` — number of output classes
    /// * `in_channels` — number of input image channels (3 for RGB)
    fn new(layers: [usize; 4], num_classes: usize, in_channels: usize) -> Self {
        // CIFAR-10 variant: 3x3 conv with stride 1 and padding 1 (no 7x7, no maxpool)
        let conv1 = Conv2d::with_options(in_channels, 64, 3, 1, 1);
        let bn1 = BatchNorm2d::new(64);

        let layer1 = Self::make_layer(64, 64, layers[0], 1);
        let layer2 = Self::make_layer(64, 128, layers[1], 2);
        let layer3 = Self::make_layer(128, 256, layers[2], 2);
        let layer4 = Self::make_layer(256, 512, layers[3], 2);

        let avgpool = AdaptiveAvgPool2d::new(1);
        let flatten = Flatten::new();
        let fc = Linear::new(512, num_classes);

        ResNet {
            conv1,
            bn1,
            layer1,
            layer2,
            layer3,
            layer4,
            avgpool,
            flatten,
            fc,
        }
    }

    /// ResNet-18: [2, 2, 2, 2] blocks.
    pub fn resnet18(num_classes: usize) -> Self {
        Self::new([2, 2, 2, 2], num_classes, 3)
    }

    /// ResNet-34: [3, 4, 6, 3] blocks.
    pub fn resnet34(num_classes: usize) -> Self {
        Self::new([3, 4, 6, 3], num_classes, 3)
    }

    /// Create a layer group of residual blocks.
    ///
    /// The first block may downsample (stride=2) and change channels.
    /// Subsequent blocks maintain the same dimensions.
    fn make_layer(
        in_channels: usize,
        out_channels: usize,
        num_blocks: usize,
        stride: usize,
    ) -> Vec<ResidualBlock> {
        let mut blocks = Vec::with_capacity(num_blocks);

        // First block: may change channels and/or downsample
        blocks.push(ResidualBlock::new(in_channels, out_channels, stride));

        // Remaining blocks: same channels, stride 1
        for _ in 1..num_blocks {
            blocks.push(ResidualBlock::new(out_channels, out_channels, 1));
        }

        blocks
    }

    /// Set training mode for all batch normalization layers.
    pub fn train(&self) {
        self.bn1.train();
        for block in &self.layer1 {
            block.train();
        }
        for block in &self.layer2 {
            block.train();
        }
        for block in &self.layer3 {
            block.train();
        }
        for block in &self.layer4 {
            block.train();
        }
    }

    /// Set eval mode for all batch normalization layers.
    pub fn eval(&self) {
        self.bn1.eval();
        for block in &self.layer1 {
            block.eval();
        }
        for block in &self.layer2 {
            block.eval();
        }
        for block in &self.layer3 {
            block.eval();
        }
        for block in &self.layer4 {
            block.eval();
        }
    }

    /// Total number of trainable parameters.
    pub fn num_params(&self) -> usize {
        self.parameters().iter().map(|p| p.tensor().numel()).sum()
    }

    fn forward_layer(blocks: &[ResidualBlock], x: &Variable) -> Result<Variable> {
        let mut out = blocks[0].forward(x)?;
        for block in &blocks[1..] {
            out = block.forward(&out)?;
        }
        Ok(out)
    }

    fn layer_params(blocks: &[ResidualBlock]) -> Vec<Parameter> {
        blocks.iter().flat_map(|b| b.parameters()).collect()
    }

    fn layer_state_dict(blocks: &[ResidualBlock], prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        for (i, block) in blocks.iter().enumerate() {
            sd.merge_prefixed(&format!("{}.{}", prefix, i), &block.state_dict());
        }
        sd
    }

    fn load_layer_state_dict(blocks: &[ResidualBlock], state_dict: &StateDict, prefix: &str) {
        for (i, block) in blocks.iter().enumerate() {
            block.load_state_dict(&state_dict.sub_dict(&format!("{}.{}", prefix, i)));
        }
    }
}

impl Module for ResNet {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        // Initial conv → bn → relu
        let x = self.conv1.forward(input)?;
        let x = self.bn1.forward(&x)?;
        let x = x.relu();

        // 4 residual layer groups
        let x = Self::forward_layer(&self.layer1, &x)?;
        let x = Self::forward_layer(&self.layer2, &x)?;
        let x = Self::forward_layer(&self.layer3, &x)?;
        let x = Self::forward_layer(&self.layer4, &x)?;

        // Global average pool → flatten → FC
        let x = self.avgpool.forward(&x)?;
        let x = self.flatten.forward(&x)?;
        self.fc.forward(&x)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters());
        params.extend(self.bn1.parameters());
        params.extend(Self::layer_params(&self.layer1));
        params.extend(Self::layer_params(&self.layer2));
        params.extend(Self::layer_params(&self.layer3));
        params.extend(Self::layer_params(&self.layer4));
        params.extend(self.fc.parameters());
        params
    }

    fn state_dict(&self) -> StateDict {
        let mut sd = StateDict::new();
        sd.merge_prefixed("conv1", &self.conv1.state_dict());
        sd.merge_prefixed("bn1", &self.bn1.state_dict());
        sd.merge(Self::layer_state_dict(&self.layer1, "layer1"));
        sd.merge(Self::layer_state_dict(&self.layer2, "layer2"));
        sd.merge(Self::layer_state_dict(&self.layer3, "layer3"));
        sd.merge(Self::layer_state_dict(&self.layer4, "layer4"));
        sd.merge_prefixed("fc", &self.fc.state_dict());
        sd
    }

    fn load_state_dict(&self, state_dict: &StateDict) {
        self.conv1.load_state_dict(&state_dict.sub_dict("conv1"));
        self.bn1.load_state_dict(&state_dict.sub_dict("bn1"));
        Self::load_layer_state_dict(&self.layer1, state_dict, "layer1");
        Self::load_layer_state_dict(&self.layer2, state_dict, "layer2");
        Self::load_layer_state_dict(&self.layer3, state_dict, "layer3");
        Self::load_layer_state_dict(&self.layer4, state_dict, "layer4");
        self.fc.load_state_dict(&state_dict.sub_dict("fc"));
    }
}

impl std::fmt::Debug for ResNet {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let blocks: Vec<usize> = [&self.layer1, &self.layer2, &self.layer3, &self.layer4]
            .iter()
            .map(|l| l.len())
            .collect();
        write!(
            f,
            "ResNet(blocks={:?}, params={})",
            blocks,
            self.num_params()
        )
    }
}
