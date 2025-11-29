//! Activation functions for neural networks

use crate::tensor::Tensor;

/// ReLU activation: max(0, x)
pub fn relu(tensor: &Tensor) -> Tensor {
    // TODO: Implement
    Tensor::zeros(&[1], crate::DType::Float32)
}

/// Sigmoid activation: 1 / (1 + exp(-x))
pub fn sigmoid(tensor: &Tensor) -> Tensor {
    // TODO: Implement
    Tensor::zeros(&[1], crate::DType::Float32)
}

/// Tanh activation
pub fn tanh(tensor: &Tensor) -> Tensor {
    // TODO: Implement
    Tensor::zeros(&[1], crate::DType::Float32)
}

/// GELU activation (Gaussian Error Linear Unit)
pub fn gelu(tensor: &Tensor) -> Tensor {
    // TODO: Implement
    Tensor::zeros(&[1], crate::DType::Float32)
}

/// Softmax activation along a dimension
pub fn softmax(tensor: &Tensor, dim: usize) -> Tensor {
    // TODO: Implement
    Tensor::zeros(&[1], crate::DType::Float32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DType;

    #[test]
    fn test_relu_stub() {
        let t = Tensor::ones(&[2, 2], DType::Float32);
        let r = relu(&t);
        assert_eq!(r.shape(), &[1]); // Stub implementation
    }
}
