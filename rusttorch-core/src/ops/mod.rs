//! Tensor operations module
//!
//! This module contains all tensor operations including:
//! - Element-wise operations (add, mul, etc.)
//! - Reduction operations (sum, mean, etc.)
//! - Activation functions (relu, sigmoid, etc.)
//! - Matrix operations (matmul, transpose, reshape)
//! - Loss functions (MSE, cross-entropy, etc.)
//! - Optimizer update rules (SGD, Adam, etc.)
//! - Broadcasting utilities for automatic shape expansion

pub mod elementwise;
pub mod reduction;
pub mod activation;
pub mod matrix;
pub mod loss;
pub mod optimizer;
pub mod broadcast;
pub mod simd;

pub use elementwise::*;
pub use reduction::*;
pub use activation::*;
pub use matrix::*;
pub use loss::*;
pub use optimizer::*;
pub use broadcast::*;
pub use simd::*;
