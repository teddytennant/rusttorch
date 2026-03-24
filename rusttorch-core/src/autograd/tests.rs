//! Comprehensive tests for the autograd system.
//!
//! Uses numerical gradient checking to verify analytical gradients.

#[cfg(test)]
mod tests {
    use crate::autograd::Variable;
    use crate::tensor::Tensor;

    /// Numerical gradient checking helper.
    ///
    /// Perturbs each element of the input by ±epsilon, recomputes the output,
    /// and estimates the gradient via finite differences:
    ///   grad_i ≈ (f(x + eps*e_i) - f(x - eps*e_i)) / (2 * eps)
    ///
    /// Then compares with the analytical gradient from autograd.
    fn check_gradient(
        input_data: Vec<f32>,
        shape: &[usize],
        forward_fn: impl Fn(&Variable) -> Variable,
        tolerance: f32,
    ) {
        let eps = 1e-3_f32;

        // Compute analytical gradient
        let x = Variable::new(Tensor::from_vec(input_data.clone(), shape), true);
        let y = forward_fn(&x);
        y.backward().unwrap();
        let analytical = x.grad().expect("gradient should exist");
        let analytical_data = analytical.to_vec_f32();

        // Compute numerical gradient for each element
        for i in 0..input_data.len() {
            let mut plus = input_data.clone();
            plus[i] += eps;
            let x_plus = Variable::new(Tensor::from_vec(plus, shape), false);
            let y_plus = forward_fn(&x_plus);
            let val_plus = y_plus.tensor().item().unwrap();

            let mut minus = input_data.clone();
            minus[i] -= eps;
            let x_minus = Variable::new(Tensor::from_vec(minus, shape), false);
            let y_minus = forward_fn(&x_minus);
            let val_minus = y_minus.tensor().item().unwrap();

            let numerical = (val_plus - val_minus) / (2.0 * eps as f64);
            let diff = (analytical_data[i] as f64 - numerical).abs();
            assert!(
                diff < tolerance as f64,
                "Gradient mismatch at index {}: analytical={}, numerical={}, diff={}",
                i,
                analytical_data[i],
                numerical,
                diff
            );
        }
    }

    // ========== Basic backward tests ==========

    #[test]
    fn test_add_backward() {
        let a = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]), true);
        let b = Variable::new(Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3]), true);
        let c = a.add(&b).unwrap();
        let loss = c.sum().unwrap();
        loss.backward().unwrap();

        // d(sum(a+b))/da = [1, 1, 1], d(sum(a+b))/db = [1, 1, 1]
        let a_grad = a.grad().unwrap().to_vec_f32();
        let b_grad = b.grad().unwrap().to_vec_f32();
        assert_eq!(a_grad, vec![1.0, 1.0, 1.0]);
        assert_eq!(b_grad, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_sub_backward() {
        let a = Variable::new(Tensor::from_vec(vec![5.0, 6.0], &[2]), true);
        let b = Variable::new(Tensor::from_vec(vec![1.0, 2.0], &[2]), true);
        let c = a.sub(&b).unwrap();
        let loss = c.sum().unwrap();
        loss.backward().unwrap();

        // d(sum(a-b))/da = [1, 1], d(sum(a-b))/db = [-1, -1]
        let a_grad = a.grad().unwrap().to_vec_f32();
        let b_grad = b.grad().unwrap().to_vec_f32();
        assert_eq!(a_grad, vec![1.0, 1.0]);
        assert_eq!(b_grad, vec![-1.0, -1.0]);
    }

    #[test]
    fn test_mul_backward() {
        let a = Variable::new(Tensor::from_vec(vec![2.0, 3.0], &[2]), true);
        let b = Variable::new(Tensor::from_vec(vec![4.0, 5.0], &[2]), true);
        let c = a.mul(&b).unwrap();
        let loss = c.sum().unwrap();
        loss.backward().unwrap();

        // d(sum(a*b))/da = b = [4, 5], d(sum(a*b))/db = a = [2, 3]
        let a_grad = a.grad().unwrap().to_vec_f32();
        let b_grad = b.grad().unwrap().to_vec_f32();
        assert_eq!(a_grad, vec![4.0, 5.0]);
        assert_eq!(b_grad, vec![2.0, 3.0]);
    }

    #[test]
    fn test_div_backward() {
        let a = Variable::new(Tensor::from_vec(vec![6.0, 8.0], &[2]), true);
        let b = Variable::new(Tensor::from_vec(vec![2.0, 4.0], &[2]), true);
        let c = a.div(&b).unwrap();
        let loss = c.sum().unwrap();
        loss.backward().unwrap();

        // d(sum(a/b))/da = 1/b = [0.5, 0.25]
        // d(sum(a/b))/db = -a/b² = [-6/4, -8/16] = [-1.5, -0.5]
        let a_grad = a.grad().unwrap().to_vec_f32();
        let b_grad = b.grad().unwrap().to_vec_f32();
        assert!((a_grad[0] - 0.5).abs() < 1e-5);
        assert!((a_grad[1] - 0.25).abs() < 1e-5);
        assert!((b_grad[0] - (-1.5)).abs() < 1e-5);
        assert!((b_grad[1] - (-0.5)).abs() < 1e-5);
    }

    #[test]
    fn test_matmul_backward() {
        // A(2x3) @ B(3x2) = C(2x2), then sum to scalar
        let a = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]),
            true,
        );
        let b = Variable::new(
            Tensor::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]),
            true,
        );
        let c = a.matmul(&b).unwrap();
        let loss = c.sum().unwrap();
        loss.backward().unwrap();

        // d(sum(A@B))/dA = ones(2,2) @ B^T = [[7+8, 9+10, 11+12], [7+8, 9+10, 11+12]]
        //                                   = [[15, 19, 23], [15, 19, 23]]
        let a_grad = a.grad().unwrap().to_vec_f32();
        assert!((a_grad[0] - 15.0).abs() < 1e-4);
        assert!((a_grad[1] - 19.0).abs() < 1e-4);
        assert!((a_grad[2] - 23.0).abs() < 1e-4);

        // d(sum(A@B))/dB = A^T @ ones(2,2) = [[1+4, 1+4], [2+5, 2+5], [3+6, 3+6]]
        //                                   = [[5, 5], [7, 7], [9, 9]]
        let b_grad = b.grad().unwrap().to_vec_f32();
        assert!((b_grad[0] - 5.0).abs() < 1e-4);
        assert!((b_grad[1] - 5.0).abs() < 1e-4);
        assert!((b_grad[2] - 7.0).abs() < 1e-4);
        assert!((b_grad[3] - 7.0).abs() < 1e-4);
    }

    #[test]
    fn test_relu_backward() {
        let x = Variable::new(
            Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]),
            true,
        );
        let y = x.relu();
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        // d(relu(x))/dx = 0 if x <= 0, 1 if x > 0
        let grad = x.grad().unwrap().to_vec_f32();
        assert_eq!(grad, vec![0.0, 0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn test_sigmoid_backward() {
        let x = Variable::new(Tensor::from_vec(vec![0.0], &[1]), true);
        let y = x.sigmoid().unwrap();
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        // sigmoid(0) = 0.5, d(sigmoid)/dx = 0.5 * 0.5 = 0.25
        let grad = x.grad().unwrap().to_vec_f32();
        assert!((grad[0] - 0.25).abs() < 1e-5);
    }

    #[test]
    fn test_tanh_backward() {
        let x = Variable::new(Tensor::from_vec(vec![0.0], &[1]), true);
        let y = x.tanh_act().unwrap();
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        // tanh(0) = 0, d(tanh)/dx = 1 - 0² = 1
        let grad = x.grad().unwrap().to_vec_f32();
        assert!((grad[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_sum_backward() {
        let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]), true);
        let loss = x.sum().unwrap();
        loss.backward().unwrap();

        let grad = x.grad().unwrap().to_vec_f32();
        assert_eq!(grad, vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_mean_backward() {
        let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]), true);
        let loss = x.mean().unwrap();
        loss.backward().unwrap();

        // d(mean)/dx_i = 1/n = 0.25
        let grad = x.grad().unwrap().to_vec_f32();
        for &g in &grad {
            assert!((g - 0.25).abs() < 1e-5);
        }
    }

    #[test]
    fn test_mul_scalar_backward() {
        let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]), true);
        let y = x.mul_scalar(5.0);
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        let grad = x.grad().unwrap().to_vec_f32();
        assert_eq!(grad, vec![5.0, 5.0, 5.0]);
    }

    // ========== Composition / chain rule tests ==========

    #[test]
    fn test_chain_mul_add() {
        // f(x) = sum((x * x) + x) = sum(x² + x)
        // df/dx_i = 2*x_i + 1
        let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]), true);
        let x2 = x.mul(&x).unwrap();
        let y = x2.add(&x).unwrap();
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        let grad = x.grad().unwrap().to_vec_f32();
        assert!((grad[0] - 3.0).abs() < 1e-5); // 2*1+1 = 3
        assert!((grad[1] - 5.0).abs() < 1e-5); // 2*2+1 = 5
        assert!((grad[2] - 7.0).abs() < 1e-5); // 2*3+1 = 7
    }

    #[test]
    fn test_chain_relu_sum() {
        // f(x) = sum(relu(x * 2 - 3))
        let x = Variable::new(Tensor::from_vec(vec![0.0, 1.0, 2.0, 3.0], &[4]), true);
        let two = Variable::new(Tensor::from_vec(vec![2.0, 2.0, 2.0, 2.0], &[4]), false);
        let three = Variable::new(Tensor::from_vec(vec![3.0, 3.0, 3.0, 3.0], &[4]), false);
        let y = x.mul(&two).unwrap();
        let z = y.sub(&three).unwrap();
        let r = z.relu();
        let loss = r.sum().unwrap();
        loss.backward().unwrap();

        // 2x - 3: [-3, -1, 1, 3]
        // relu: [0, 0, 1, 3]
        // grad of relu: [0, 0, 1, 1] (mask)
        // grad through sub: same
        // grad through mul by 2: [0, 0, 2, 2]
        let grad = x.grad().unwrap().to_vec_f32();
        assert_eq!(grad, vec![0.0, 0.0, 2.0, 2.0]);
    }

    #[test]
    fn test_linear_layer() {
        // y = relu(x @ W + b), loss = sum(y)
        // x: [1, 3], W: [3, 2], b: [1, 2]
        let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]), true);
        let w = Variable::new(
            Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[3, 2]),
            true,
        );
        let b = Variable::new(Tensor::from_vec(vec![0.1, -0.1], &[1, 2]), true);

        let xw = x.matmul(&w).unwrap();
        let xw_b = xw.add(&b).unwrap();
        let y = xw_b.relu();
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        // Just verify gradients exist and have correct shapes
        assert_eq!(x.grad().unwrap().shape(), &[1, 3]);
        assert_eq!(w.grad().unwrap().shape(), &[3, 2]);
        assert_eq!(b.grad().unwrap().shape(), &[1, 2]);
    }

    #[test]
    fn test_diamond_graph() {
        // Diamond: a -> mul_scalar(3) -> b, a -> mul_scalar(5) -> c, b + c -> d
        // d = 3a + 5a = 8a, so dd/da = 8
        let a = Variable::new(Tensor::from_vec(vec![2.0], &[1]), true);
        let b = a.mul_scalar(3.0);
        let c = a.mul_scalar(5.0);
        let d = b.add(&c).unwrap();
        let loss = d.sum().unwrap();
        loss.backward().unwrap();

        let grad = a.grad().unwrap().to_vec_f32();
        assert!((grad[0] - 8.0).abs() < 1e-5);
    }

    #[test]
    fn test_no_grad_propagation() {
        // b doesn't require grad, so it shouldn't get one
        let a = Variable::new(Tensor::from_vec(vec![1.0, 2.0], &[2]), true);
        let b = Variable::new(Tensor::from_vec(vec![3.0, 4.0], &[2]), false);
        let c = a.mul(&b).unwrap();
        let loss = c.sum().unwrap();
        loss.backward().unwrap();

        assert!(a.grad().is_some());
        assert!(b.grad().is_none());
    }

    #[test]
    fn test_zero_grad() {
        let x = Variable::new(Tensor::from_vec(vec![2.0, 3.0], &[2]), true);
        let y = x.mul_scalar(3.0);
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        assert!(x.grad().is_some());
        x.zero_grad();
        assert!(x.grad().is_none());
    }

    #[test]
    fn test_backward_requires_scalar() {
        let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]), true);
        let y = x.mul_scalar(2.0);
        // y is not scalar, backward should fail
        let result = y.backward();
        assert!(result.is_err());
    }

    // ========== Numerical gradient checking ==========

    #[test]
    fn test_numerical_add_sum() {
        check_gradient(
            vec![1.0, 2.0, 3.0, 4.0],
            &[4],
            |x| {
                let two = Variable::new(Tensor::from_vec(vec![2.0, 2.0, 2.0, 2.0], &[4]), false);
                let y = x.add(&two).unwrap();
                y.sum().unwrap()
            },
            1e-3,
        );
    }

    #[test]
    fn test_numerical_mul_sum() {
        check_gradient(
            vec![1.0, 2.0, 3.0],
            &[3],
            |x| {
                let y = x.mul(x).unwrap(); // x²
                y.sum().unwrap()
            },
            1e-2,
        );
    }

    #[test]
    fn test_numerical_sigmoid() {
        check_gradient(
            vec![-2.0, -1.0, 0.0, 1.0, 2.0],
            &[5],
            |x| {
                let y = x.sigmoid().unwrap();
                y.sum().unwrap()
            },
            1e-3,
        );
    }

    #[test]
    fn test_numerical_tanh() {
        check_gradient(
            vec![-2.0, -1.0, 0.0, 1.0, 2.0],
            &[5],
            |x| {
                let y = x.tanh_act().unwrap();
                y.sum().unwrap()
            },
            1e-3,
        );
    }

    #[test]
    fn test_numerical_relu() {
        check_gradient(
            vec![-2.0, -1.0, 0.5, 1.0, 2.0],
            &[5],
            |x| {
                let y = x.relu();
                y.sum().unwrap()
            },
            1e-3,
        );
    }

    #[test]
    fn test_numerical_div() {
        check_gradient(
            vec![1.0, 2.0, 3.0, 4.0],
            &[4],
            |x| {
                let denom = Variable::new(Tensor::from_vec(vec![2.0, 3.0, 4.0, 5.0], &[4]), false);
                let y = x.div(&denom).unwrap();
                y.sum().unwrap()
            },
            1e-3,
        );
    }

    #[test]
    fn test_numerical_mean() {
        check_gradient(vec![1.0, 2.0, 3.0, 4.0], &[4], |x| x.mean().unwrap(), 1e-3);
    }

    #[test]
    fn test_numerical_chain() {
        // f(x) = sum(sigmoid(x * 2 + 1))
        check_gradient(
            vec![-1.0, 0.0, 1.0],
            &[3],
            |x| {
                let y = x.mul_scalar(2.0);
                let one = Variable::new(Tensor::from_vec(vec![1.0, 1.0, 1.0], &[3]), false);
                let z = y.add(&one).unwrap();
                let s = z.sigmoid().unwrap();
                s.sum().unwrap()
            },
            1e-3,
        );
    }

    #[test]
    fn test_numerical_matmul() {
        // f(X) = sum(X @ W) where W is fixed
        check_gradient(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
            |x| {
                let w = Variable::new(
                    Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[3, 2]),
                    false,
                );
                let y = x.matmul(&w).unwrap();
                y.sum().unwrap()
            },
            1e-3,
        );
    }

    #[test]
    fn test_numerical_deep_chain() {
        // f(x) = mean(relu(tanh(sigmoid(x * 3))))
        check_gradient(
            vec![0.1, 0.5, -0.3, 0.8],
            &[4],
            |x| {
                let y = x.mul_scalar(3.0);
                let s = y.sigmoid().unwrap();
                let t = s.tanh_act().unwrap();
                let r = t.relu();
                r.mean().unwrap()
            },
            1e-2,
        );
    }

    // ========== Edge cases ==========

    #[test]
    fn test_single_element() {
        let x = Variable::new(Tensor::from_vec(vec![3.0], &[1]), true);
        let y = x.mul_scalar(2.0);
        y.backward().unwrap();

        let grad = x.grad().unwrap().to_vec_f32();
        assert!((grad[0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_large_tensor() {
        let data: Vec<f32> = (0..1000).map(|i| i as f32 * 0.001).collect();
        let x = Variable::new(Tensor::from_vec(data, &[1000]), true);
        let y = x.mul_scalar(2.0);
        let loss = y.mean().unwrap();
        loss.backward().unwrap();

        let grad = x.grad().unwrap().to_vec_f32();
        // d(mean(2x))/dx_i = 2/1000 = 0.002
        for &g in &grad {
            assert!((g - 0.002).abs() < 1e-5);
        }
    }

    #[test]
    fn test_detach_no_grad() {
        let x = Variable::detach(Tensor::from_vec(vec![1.0, 2.0], &[2]));
        assert!(!x.requires_grad());
    }

    #[test]
    fn test_variable_debug_display() {
        let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0], &[2]), true);
        let debug = format!("{:?}", x);
        assert!(debug.contains("Variable"));
        assert!(debug.contains("requires_grad=true"));

        let display = format!("{}", x);
        assert!(display.contains("Variable"));
    }

    // ========== Multi-layer neural network test ==========

    #[test]
    fn test_two_layer_network() {
        // Two-layer network: y = sigmoid(relu(x @ W1 + b1) @ W2 + b2)
        let x = Variable::new(Tensor::from_vec(vec![1.0, 0.5], &[1, 2]), true);
        let w1 = Variable::new(
            Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3]),
            true,
        );
        let b1 = Variable::new(Tensor::from_vec(vec![0.01, 0.02, 0.03], &[1, 3]), true);
        let w2 = Variable::new(Tensor::from_vec(vec![0.1, 0.2, 0.3], &[3, 1]), true);
        let b2 = Variable::new(Tensor::from_vec(vec![0.01], &[1, 1]), true);

        // Forward
        let h = x.matmul(&w1).unwrap().add(&b1).unwrap().relu();
        let out = h.matmul(&w2).unwrap().add(&b2).unwrap().sigmoid().unwrap();
        let loss = out.sum().unwrap();
        loss.backward().unwrap();

        // All parameters should have gradients with correct shapes
        assert_eq!(x.grad().unwrap().shape(), &[1, 2]);
        assert_eq!(w1.grad().unwrap().shape(), &[2, 3]);
        assert_eq!(b1.grad().unwrap().shape(), &[1, 3]);
        assert_eq!(w2.grad().unwrap().shape(), &[3, 1]);
        assert_eq!(b2.grad().unwrap().shape(), &[1, 1]);

        // Gradients should be finite and non-zero (for this input)
        let w1_grad = w1.grad().unwrap().to_vec_f32();
        for &g in &w1_grad {
            assert!(g.is_finite());
        }
    }

    #[test]
    fn test_mse_loss_backward() {
        // MSE loss: loss = mean((pred - target)²)
        let pred = Variable::new(Tensor::from_vec(vec![0.5, 0.8, 0.2], &[3]), true);
        let target = Variable::new(Tensor::from_vec(vec![1.0, 0.0, 0.0], &[3]), false);

        let diff = pred.sub(&target).unwrap();
        let sq = diff.mul(&diff).unwrap();
        let loss = sq.mean().unwrap();
        loss.backward().unwrap();

        // d(mean((p-t)²))/dp_i = 2*(p_i - t_i)/n
        let grad = pred.grad().unwrap().to_vec_f32();
        let expected = vec![
            2.0 * (0.5 - 1.0) / 3.0, // -0.333...
            2.0 * (0.8 - 0.0) / 3.0, // 0.533...
            2.0 * (0.2 - 0.0) / 3.0, // 0.133...
        ];
        for (g, e) in grad.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-4, "got {}, expected {}", g, e);
        }
    }
}
