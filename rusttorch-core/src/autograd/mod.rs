//! Automatic differentiation engine for RustTorch
//!
//! This module implements reverse-mode automatic differentiation (backpropagation).
//! It provides `Variable` — a tensor wrapper that tracks computation history and
//! can compute gradients via `backward()`.
//!
//! # Design
//!
//! - **Dynamic computational graph** (PyTorch-style, not static like TensorFlow)
//! - **Variable wraps Tensor** — autograd is opt-in via `requires_grad`
//! - **Reverse-mode AD** — efficient for scalar outputs (loss functions)
//! - **Gradient accumulation** — multiple paths to the same variable accumulate
//!
//! # Example
//!
//! ```
//! use rusttorch_core::autograd::Variable;
//! use rusttorch_core::Tensor;
//!
//! let x = Variable::new(Tensor::from_vec(vec![2.0, 3.0], &[2]), true);
//! let w = Variable::new(Tensor::from_vec(vec![1.0, -1.0], &[2]), true);
//! let y = x.mul(&w).unwrap();
//! let loss = y.sum().unwrap();
//! loss.backward().unwrap();
//!
//! assert!(x.grad().is_some());
//! assert!(w.grad().is_some());
//! ```

pub mod graph;
pub mod ops;
pub mod variable;

#[cfg(test)]
mod tests;

pub use variable::Variable;
