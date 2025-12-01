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

pub mod activation;
pub mod broadcast;
pub mod elementwise;
pub mod loss;
pub mod matrix;
pub mod optimizer;
pub mod reduction;
pub mod simd;

pub use activation::*;
pub use broadcast::*;
pub use elementwise::*;
pub use loss::*;
pub use matrix::*;
pub use optimizer::*;
pub use reduction::*;
pub use simd::*;
