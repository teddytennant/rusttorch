//! Neural network module — PyTorch-style `nn` for RustTorch.
//!
//! Provides the building blocks for constructing and training neural networks:
//! - `Module` trait — base abstraction for all layers
//! - `Parameter` — learnable weight wrapper around `Variable`
//! - `Linear` — fully connected layer
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
//! let model = Sequential::new(vec![
//!     Box::new(Linear::new(2, 8)),
//!     Box::new(ReLU::new()),
//!     Box::new(Linear::new(8, 1)),
//!     Box::new(Sigmoid::new()),
//! ]);
//!
//! let loss_fn = MSELoss::new();
//! let mut optimizer = SGD::new(model.parameters(), 0.1);
//!
//! // Training loop
//! optimizer.zero_grad();
//! let output = model.forward(&input)?;
//! let loss = loss_fn.forward(&output, &target)?;
//! loss.backward()?;
//! optimizer.step()?;
//! ```

pub mod activation;
pub mod linear;
pub mod loss;
pub mod module;
pub mod optim;
pub mod parameter;
pub mod sequential;

#[cfg(test)]
mod tests;

pub use activation::{ReLU, Sigmoid, Tanh};
pub use linear::Linear;
pub use loss::MSELoss;
pub use module::Module;
pub use optim::{Adam, Optimizer, SGD};
pub use parameter::Parameter;
pub use sequential::Sequential;
