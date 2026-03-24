//! Neural network module — PyTorch-style `nn` for RustTorch.
//!
//! Provides the building blocks for constructing and training neural networks:
//! - `Module` trait — base abstraction for all layers
//! - `Parameter` — learnable weight wrapper around `Variable`
//! - `Linear` — fully connected layer
//! - `Conv2d` — 2D convolution layer
//! - `MaxPool2d` — 2D max pooling
//! - `Flatten` — flatten spatial dimensions
//! - `ReLU`, `Sigmoid`, `Tanh` — activation modules
//! - `Sequential` — layer container
//! - `MSELoss` — loss function
//! - `SGD`, `Adam` — optimizers
//!
//! # Example
//!
//! ```ignore
//! use rusttorch_core::nn::*;
//!
//! // CNN example
//! let model = Sequential::new(vec![
//!     Box::new(Conv2d::new(1, 16, 3)),      // [B, 1, 28, 28] -> [B, 16, 26, 26]
//!     Box::new(ReLU::new()),
//!     Box::new(MaxPool2d::new(2)),           // -> [B, 16, 13, 13]
//!     Box::new(Conv2d::new(16, 32, 3)),      // -> [B, 32, 11, 11]
//!     Box::new(ReLU::new()),
//!     Box::new(MaxPool2d::new(2)),           // -> [B, 32, 5, 5]
//!     Box::new(Flatten::new()),              // -> [B, 800]
//!     Box::new(Linear::new(800, 10)),        // -> [B, 10]
//! ]);
//! ```

pub mod activation;
pub mod conv2d;
pub mod flatten;
pub mod linear;
pub mod loss;
pub mod module;
pub mod optim;
pub mod parameter;
pub mod pool;
pub mod sequential;

#[cfg(test)]
mod tests;

pub use activation::{ReLU, Sigmoid, Tanh};
pub use conv2d::Conv2d;
pub use flatten::Flatten;
pub use linear::Linear;
pub use loss::MSELoss;
pub use module::Module;
pub use optim::{Adam, Optimizer, SGD};
pub use parameter::Parameter;
pub use pool::MaxPool2d;
pub use sequential::Sequential;
