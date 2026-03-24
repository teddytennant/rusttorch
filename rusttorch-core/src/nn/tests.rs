//! Tests for the nn module.

use crate::autograd::Variable;
use crate::nn::*;
use crate::tensor::Tensor;

// ---- Parameter tests ----

#[test]
fn test_parameter_new() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
    let p = Parameter::new(t, "test");
    assert_eq!(p.name(), "test");
    assert_eq!(p.shape(), vec![3]);
    assert!(p.var().requires_grad());
    assert!(p.var().is_leaf());
}

#[test]
fn test_parameter_kaiming_uniform() {
    let p = Parameter::kaiming_uniform(&[4, 3], 3, "weight");
    assert_eq!(p.shape(), vec![4, 3]);
    // Values should be within bound = sqrt(6/3) ≈ 1.414
    let data = p.tensor().to_vec_f32();
    let bound = (6.0_f32 / 3.0).sqrt();
    for &v in &data {
        assert!(v >= -bound && v <= bound, "Value {} outside bounds", v);
    }
}

#[test]
fn test_parameter_uniform() {
    let p = Parameter::uniform(&[10], 0.5, "bias");
    assert_eq!(p.shape(), vec![10]);
    let data = p.tensor().to_vec_f32();
    for &v in &data {
        assert!(v >= -0.5 && v <= 0.5, "Value {} outside bounds", v);
    }
}

#[test]
fn test_parameter_zeros() {
    let p = Parameter::zeros(&[3, 2], "zero_param");
    let data = p.tensor().to_vec_f32();
    for &v in &data {
        assert_eq!(v, 0.0);
    }
}

#[test]
fn test_parameter_update() {
    let p = Parameter::new(Tensor::from_vec(vec![1.0, 2.0], &[2]), "w");
    p.update(Tensor::from_vec(vec![3.0, 4.0], &[2]));
    let data = p.tensor().to_vec_f32();
    assert_eq!(data, vec![3.0, 4.0]);
}

#[test]
fn test_parameter_grad_lifecycle() {
    let p = Parameter::new(Tensor::from_vec(vec![2.0], &[1]), "w");
    assert!(p.grad().is_none());
    // Compute a gradient
    let x = Variable::new(Tensor::from_vec(vec![3.0], &[1]), false);
    let y = p.var().mul(&x).unwrap();
    let loss = y.sum().unwrap();
    loss.backward().unwrap();
    assert!(p.grad().is_some());
    p.zero_grad();
    assert!(p.grad().is_none());
}

// ---- Linear tests ----

#[test]
fn test_linear_shape() {
    let layer = Linear::new(3, 5);
    assert_eq!(layer.in_features, 3);
    assert_eq!(layer.out_features, 5);
    assert_eq!(layer.weight.shape(), vec![5, 3]);
    assert!(layer.bias.is_some());
    assert_eq!(layer.bias.as_ref().unwrap().shape(), vec![1, 5]);
}

#[test]
fn test_linear_forward() {
    let layer = Linear::new(3, 5);
    let input = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]), false);
    let output = layer.forward(&input).unwrap();
    assert_eq!(output.shape(), vec![1, 5]);
}

#[test]
fn test_linear_batch() {
    let layer = Linear::new(4, 2);
    let input = Variable::new(Tensor::from_vec(vec![1.0; 12], &[3, 4]), false);
    let output = layer.forward(&input).unwrap();
    assert_eq!(output.shape(), vec![3, 2]);
}

#[test]
fn test_linear_no_bias() {
    let layer = Linear::no_bias(3, 5);
    assert!(layer.bias.is_none());
    let input = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]), false);
    let output = layer.forward(&input).unwrap();
    assert_eq!(output.shape(), vec![1, 5]);
}

#[test]
fn test_linear_parameters_count() {
    let layer = Linear::new(3, 5);
    let params = layer.parameters();
    assert_eq!(params.len(), 2); // weight + bias
    assert_eq!(layer.num_parameters(), 3 * 5 + 1 * 5); // 15 + 5 = 20
}

#[test]
fn test_linear_no_bias_parameters() {
    let layer = Linear::no_bias(3, 5);
    assert_eq!(layer.parameters().len(), 1);
    assert_eq!(layer.num_parameters(), 15);
}

#[test]
fn test_linear_backward() {
    let layer = Linear::new(2, 1);
    let input = Variable::new(Tensor::from_vec(vec![1.0, 2.0], &[1, 2]), true);
    let output = layer.forward(&input).unwrap();
    let loss = output.sum().unwrap();
    loss.backward().unwrap();
    // Weight should have a gradient
    assert!(layer.weight.grad().is_some());
}

// ---- Activation module tests ----

#[test]
fn test_relu_module() {
    let relu = ReLU::new();
    let input = Variable::new(Tensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0], &[1, 4]), true);
    let output = relu.forward(&input).unwrap();
    let data = output.tensor().to_vec_f32();
    assert_eq!(data, vec![0.0, 0.0, 1.0, 2.0]);
    assert!(relu.parameters().is_empty());
}

#[test]
fn test_sigmoid_module() {
    let sigmoid = Sigmoid::new();
    let input = Variable::new(Tensor::from_vec(vec![0.0], &[1, 1]), false);
    let output = sigmoid.forward(&input).unwrap();
    let data = output.tensor().to_vec_f32();
    assert!((data[0] - 0.5).abs() < 1e-6);
}

#[test]
fn test_tanh_module() {
    let tanh = Tanh::new();
    let input = Variable::new(Tensor::from_vec(vec![0.0], &[1, 1]), false);
    let output = tanh.forward(&input).unwrap();
    let data = output.tensor().to_vec_f32();
    assert!((data[0] - 0.0).abs() < 1e-6);
}

// ---- Sequential tests ----

#[test]
fn test_sequential_forward() {
    let model = Sequential::new(vec![
        Box::new(Linear::new(2, 4)),
        Box::new(ReLU::new()),
        Box::new(Linear::new(4, 1)),
    ]);

    let input = Variable::new(Tensor::from_vec(vec![1.0, 2.0], &[1, 2]), false);
    let output = model.forward(&input).unwrap();
    assert_eq!(output.shape(), vec![1, 1]);
}

#[test]
fn test_sequential_parameters() {
    let model = Sequential::new(vec![
        Box::new(Linear::new(2, 4)),
        Box::new(ReLU::new()),
        Box::new(Linear::new(4, 1)),
    ]);

    // Linear(2,4): weight 8 + bias 4 = 12
    // ReLU: 0
    // Linear(4,1): weight 4 + bias 1 = 5
    // Total: 17
    assert_eq!(model.num_parameters(), 17);
    assert_eq!(model.parameters().len(), 4); // 2 weights + 2 biases
}

#[test]
fn test_sequential_len() {
    let model = Sequential::new(vec![Box::new(Linear::new(2, 4)), Box::new(ReLU::new())]);
    assert_eq!(model.len(), 2);
    assert!(!model.is_empty());
}

#[test]
fn test_sequential_empty() {
    let model = Sequential::new(vec![]);
    assert!(model.is_empty());
    let input = Variable::new(Tensor::from_vec(vec![1.0], &[1, 1]), false);
    let output = model.forward(&input).unwrap();
    // Empty sequential returns input unchanged
    assert_eq!(output.shape(), vec![1, 1]);
}

#[test]
fn test_sequential_backward() {
    let model = Sequential::new(vec![
        Box::new(Linear::new(2, 4)),
        Box::new(ReLU::new()),
        Box::new(Linear::new(4, 1)),
        Box::new(Sigmoid::new()),
    ]);

    let input = Variable::new(Tensor::from_vec(vec![1.0, 2.0], &[1, 2]), false);
    let output = model.forward(&input).unwrap();
    let loss = output.sum().unwrap();
    loss.backward().unwrap();

    // All weight parameters should have gradients
    for param in model.parameters() {
        assert!(
            param.grad().is_some(),
            "Parameter {} missing gradient",
            param.name()
        );
    }
}

// ---- MSELoss tests ----

#[test]
fn test_mse_loss_zero() {
    let loss_fn = MSELoss::new();
    let pred = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]), true);
    let target = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]), false);
    let loss = loss_fn.forward(&pred, &target).unwrap();
    let val = loss.tensor().to_vec_f32()[0];
    assert!(
        val.abs() < 1e-6,
        "MSE of identical tensors should be 0, got {}",
        val
    );
}

#[test]
fn test_mse_loss_value() {
    let loss_fn = MSELoss::new();
    let pred = Variable::new(Tensor::from_vec(vec![1.0, 2.0], &[1, 2]), true);
    let target = Variable::new(Tensor::from_vec(vec![3.0, 4.0], &[1, 2]), false);
    let loss = loss_fn.forward(&pred, &target).unwrap();
    let val = loss.tensor().to_vec_f32()[0];
    // MSE = mean((1-3)^2 + (2-4)^2) = mean(4 + 4) = 4.0
    assert!((val - 4.0).abs() < 1e-5, "Expected MSE ≈ 4.0, got {}", val);
}

#[test]
fn test_mse_loss_backward() {
    let loss_fn = MSELoss::new();
    let pred = Variable::new(Tensor::from_vec(vec![1.0, 2.0], &[1, 2]), true);
    let target = Variable::new(Tensor::from_vec(vec![3.0, 4.0], &[1, 2]), false);
    let loss = loss_fn.forward(&pred, &target).unwrap();
    loss.backward().unwrap();
    assert!(pred.grad().is_some());
}

// ---- SGD optimizer tests ----

#[test]
fn test_sgd_step() {
    let p = Parameter::new(Tensor::from_vec(vec![1.0, 2.0], &[1, 2]), "w");
    let params = vec![p.clone()];
    let mut opt = SGD::new(params, 0.1);

    // Simulate a gradient
    let x = Variable::new(Tensor::from_vec(vec![1.0, 1.0], &[1, 2]), false);
    let y = p.var().mul(&x).unwrap();
    let loss = y.sum().unwrap();
    loss.backward().unwrap();

    opt.step().unwrap();

    // After SGD step with lr=0.1, grad=[1,1]:
    // new_w = [1.0 - 0.1*1.0, 2.0 - 0.1*1.0] = [0.9, 1.9]
    let new_data = p.tensor().to_vec_f32();
    assert!((new_data[0] - 0.9).abs() < 1e-5);
    assert!((new_data[1] - 1.9).abs() < 1e-5);
}

#[test]
fn test_sgd_zero_grad() {
    let p = Parameter::new(Tensor::from_vec(vec![1.0], &[1, 1]), "w");
    let params = vec![p.clone()];
    let opt = SGD::new(params, 0.1);

    // Create a gradient
    let x = Variable::new(Tensor::from_vec(vec![2.0], &[1, 1]), false);
    let y = p.var().mul(&x).unwrap();
    let loss = y.sum().unwrap();
    loss.backward().unwrap();
    assert!(p.grad().is_some());

    opt.zero_grad();
    assert!(p.grad().is_none());
}

#[test]
fn test_sgd_momentum() {
    let p = Parameter::new(Tensor::from_vec(vec![1.0], &[1, 1]), "w");
    let params = vec![p.clone()];
    let mut opt = SGD::with_momentum(params, 0.1, 0.9);

    // Step 1: simulate grad = 1.0
    let x = Variable::new(Tensor::from_vec(vec![1.0], &[1, 1]), false);
    let y = p.var().mul(&x).unwrap();
    let loss = y.sum().unwrap();
    loss.backward().unwrap();
    opt.step().unwrap();

    // v1 = 0.9*0 + 1.0 = 1.0; w1 = 1.0 - 0.1*1.0 = 0.9
    let w1 = p.tensor().to_vec_f32()[0];
    assert!((w1 - 0.9).abs() < 1e-5);
}

// ---- Adam optimizer tests ----

#[test]
fn test_adam_step() {
    let p = Parameter::new(Tensor::from_vec(vec![1.0], &[1, 1]), "w");
    let params = vec![p.clone()];
    let mut opt = Adam::new(params, 0.01);

    let x = Variable::new(Tensor::from_vec(vec![1.0], &[1, 1]), false);
    let y = p.var().mul(&x).unwrap();
    let loss = y.sum().unwrap();
    loss.backward().unwrap();
    opt.step().unwrap();

    // Parameter should have changed
    let new_val = p.tensor().to_vec_f32()[0];
    assert!((new_val - 1.0).abs() > 1e-6, "Adam should update parameter");
    assert!(new_val < 1.0, "Positive gradient should decrease parameter");
}

#[test]
fn test_adam_multiple_steps() {
    let p = Parameter::new(Tensor::from_vec(vec![5.0], &[1, 1]), "w");
    let params = vec![p.clone()];
    let mut opt = Adam::new(params, 0.1);

    let initial_val = p.tensor().to_vec_f32()[0];

    // Multiple optimization steps toward 0
    for _ in 0..10 {
        opt.zero_grad();
        // Loss = w^2, grad = 2w (we approximate by using w directly)
        let target = Variable::new(Tensor::from_vec(vec![0.0], &[1, 1]), false);
        let loss_fn = MSELoss::new();
        let loss = loss_fn.forward(p.var(), &target).unwrap();
        loss.backward().unwrap();
        opt.step().unwrap();
    }

    let final_val = p.tensor().to_vec_f32()[0];
    assert!(
        final_val.abs() < initial_val.abs(),
        "Adam should move parameter toward 0: {} -> {}",
        initial_val,
        final_val
    );
}

// ---- Integration tests: full training pipeline ----

#[test]
fn test_full_training_step() {
    // Build a simple model: Linear(2,4) -> ReLU -> Linear(4,1)
    let model = Sequential::new(vec![
        Box::new(Linear::new(2, 4)),
        Box::new(ReLU::new()),
        Box::new(Linear::new(4, 1)),
    ]);

    let loss_fn = MSELoss::new();
    let mut optimizer = SGD::new(model.parameters(), 0.01);

    let input = Variable::new(Tensor::from_vec(vec![1.0, 0.0], &[1, 2]), false);
    let target = Variable::new(Tensor::from_vec(vec![1.0], &[1, 1]), false);

    // Forward
    let output = model.forward(&input).unwrap();
    assert_eq!(output.shape(), vec![1, 1]);

    // Loss
    let loss = loss_fn.forward(&output, &target).unwrap();
    let loss_val = loss.tensor().to_vec_f32()[0];
    assert!(loss_val.is_finite(), "Loss should be finite");

    // Backward
    loss.backward().unwrap();

    // Step
    optimizer.step().unwrap();

    // Forward again — loss should change
    optimizer.zero_grad();
    let output2 = model.forward(&input).unwrap();
    let loss2 = loss_fn.forward(&output2, &target).unwrap();
    let loss_val2 = loss2.tensor().to_vec_f32()[0];
    assert!(loss_val2.is_finite());
    // We can't guarantee loss decreases in one step due to random init,
    // but it should at least be different
    assert!(
        (loss_val - loss_val2).abs() > 1e-10,
        "Loss should change after a step"
    );
}

#[test]
fn test_zero_grad_clears_all() {
    let model = Sequential::new(vec![
        Box::new(Linear::new(2, 3)),
        Box::new(Linear::new(3, 1)),
    ]);

    let input = Variable::new(Tensor::from_vec(vec![1.0, 2.0], &[1, 2]), false);
    let output = model.forward(&input).unwrap();
    let loss = output.sum().unwrap();
    loss.backward().unwrap();

    // All params should have gradients
    for p in model.parameters() {
        assert!(p.grad().is_some());
    }

    model.zero_grad();

    // All params should now have no gradients
    for p in model.parameters() {
        assert!(p.grad().is_none());
    }
}

#[test]
fn test_module_debug() {
    let model = Sequential::new(vec![Box::new(Linear::new(2, 4)), Box::new(ReLU::new())]);
    let dbg = format!("{:?}", model);
    assert!(dbg.contains("Sequential"));
    assert!(dbg.contains("2 layers"));
}

#[test]
fn test_parameter_debug() {
    let p = Parameter::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]), "my_weight");
    let dbg = format!("{:?}", p);
    assert!(dbg.contains("my_weight"));
    assert!(dbg.contains("[3]"));
}

#[test]
fn test_linear_debug() {
    let layer = Linear::new(3, 5);
    let dbg = format!("{:?}", layer);
    assert!(dbg.contains("Linear"));
    assert!(dbg.contains("in=3"));
    assert!(dbg.contains("out=5"));
}

// ---- Conv2d tests ----

#[test]
fn test_conv2d_output_shape() {
    let conv = Conv2d::new(1, 4, 3); // 1 in, 4 out, 3x3 kernel
    let input = Variable::new(
        Tensor::from_vec(vec![0.0; 1 * 1 * 8 * 8], &[1, 1, 8, 8]),
        false,
    );
    let output = conv.forward(&input).unwrap();
    // oH = (8 - 3) / 1 + 1 = 6, oW = 6
    assert_eq!(output.shape(), vec![1, 4, 6, 6]);
}

#[test]
fn test_conv2d_with_padding() {
    let conv = Conv2d::with_options(1, 4, 3, 1, 1); // padding=1
    let input = Variable::new(
        Tensor::from_vec(vec![0.0; 1 * 1 * 8 * 8], &[1, 1, 8, 8]),
        false,
    );
    let output = conv.forward(&input).unwrap();
    // oH = (8 + 2 - 3) / 1 + 1 = 8 (same padding)
    assert_eq!(output.shape(), vec![1, 4, 8, 8]);
}

#[test]
fn test_conv2d_with_stride() {
    let conv = Conv2d::with_options(1, 4, 3, 2, 0); // stride=2
    let input = Variable::new(
        Tensor::from_vec(vec![0.0; 1 * 1 * 8 * 8], &[1, 1, 8, 8]),
        false,
    );
    let output = conv.forward(&input).unwrap();
    // oH = (8 - 3) / 2 + 1 = 3
    assert_eq!(output.shape(), vec![1, 4, 3, 3]);
}

#[test]
fn test_conv2d_batch() {
    let conv = Conv2d::new(3, 8, 3);
    let input = Variable::new(
        Tensor::from_vec(vec![0.5; 4 * 3 * 10 * 10], &[4, 3, 10, 10]),
        false,
    );
    let output = conv.forward(&input).unwrap();
    assert_eq!(output.shape(), vec![4, 8, 8, 8]);
}

#[test]
fn test_conv2d_parameters_count() {
    let conv = Conv2d::new(3, 16, 5);
    let params = conv.parameters();
    assert_eq!(params.len(), 2); // weight + bias
                                 // weight: 16 * 3 * 5 * 5 = 1200, bias: 16
    assert_eq!(conv.num_parameters(), 1200 + 16);
}

#[test]
fn test_conv2d_known_values() {
    // 1x1 convolution with known weight to verify correctness
    let conv = Conv2d::new(1, 1, 1);
    // Set weight to 2.0, bias to 1.0
    conv.weight
        .update(Tensor::from_vec(vec![2.0], &[1, 1, 1, 1]));
    conv.bias
        .as_ref()
        .unwrap()
        .update(Tensor::from_vec(vec![1.0], &[1]));

    let input = Variable::new(
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]),
        false,
    );
    let output = conv.forward(&input).unwrap();
    let data = output.tensor().to_vec_f32();
    // output = 2 * input + 1
    assert_eq!(data, vec![3.0, 5.0, 7.0, 9.0]);
}

#[test]
fn test_conv2d_backward() {
    let conv = Conv2d::new(1, 2, 3);
    let input = Variable::new(
        Tensor::from_vec(vec![1.0; 1 * 1 * 5 * 5], &[1, 1, 5, 5]),
        true,
    );
    let output = conv.forward(&input).unwrap();
    let loss = output.sum().unwrap();
    loss.backward().unwrap();

    // Weight should have gradient
    assert!(conv.weight.grad().is_some());
    assert_eq!(conv.weight.grad().unwrap().shape(), &[2, 1, 3, 3]);

    // Bias should have gradient
    assert!(conv.bias.as_ref().unwrap().grad().is_some());

    // Input should have gradient
    assert!(input.grad().is_some());
    assert_eq!(input.grad().unwrap().shape(), &[1, 1, 5, 5]);
}

#[test]
fn test_conv2d_numerical_gradient() {
    // Numerical gradient check for conv2d
    let eps = 1e-4;

    // Simple 1-in, 1-out, 2x2 kernel
    let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let weight_data = vec![0.1, 0.2, 0.3, 0.4];

    // Analytical gradient
    let input_var = Variable::new(Tensor::from_vec(input_data.clone(), &[1, 1, 3, 3]), true);
    let weight_var = Variable::new(Tensor::from_vec(weight_data.clone(), &[1, 1, 2, 2]), true);
    let output = crate::autograd::ops::conv2d_forward(&input_var, &weight_var, None, 1, 0).unwrap();
    let loss = output.sum().unwrap();
    loss.backward().unwrap();
    let analytical_grad = weight_var.grad().unwrap().to_vec_f32();

    // Numerical gradient for each weight element
    for i in 0..4 {
        let mut w_plus = weight_data.clone();
        w_plus[i] += eps;
        let mut w_minus = weight_data.clone();
        w_minus[i] -= eps;

        let inp = Variable::new(Tensor::from_vec(input_data.clone(), &[1, 1, 3, 3]), false);
        let wp = Variable::new(Tensor::from_vec(w_plus, &[1, 1, 2, 2]), false);
        let out_p = crate::autograd::ops::conv2d_forward(&inp, &wp, None, 1, 0).unwrap();
        let loss_p = crate::ops::sum(&out_p.tensor()) as f32;

        let inp2 = Variable::new(Tensor::from_vec(input_data.clone(), &[1, 1, 3, 3]), false);
        let wm = Variable::new(Tensor::from_vec(w_minus, &[1, 1, 2, 2]), false);
        let out_m = crate::autograd::ops::conv2d_forward(&inp2, &wm, None, 1, 0).unwrap();
        let loss_m = crate::ops::sum(&out_m.tensor()) as f32;

        let numerical = (loss_p - loss_m) / (2.0 * eps);
        let diff = (analytical_grad[i] - numerical).abs();
        assert!(
            diff < 1e-2,
            "Weight grad[{}]: analytical={}, numerical={}, diff={}",
            i,
            analytical_grad[i],
            numerical,
            diff
        );
    }
}

// ---- MaxPool2d tests ----

#[test]
fn test_maxpool2d_output_shape() {
    let pool = MaxPool2d::new(2);
    let input = Variable::new(
        Tensor::from_vec(vec![0.0; 1 * 1 * 4 * 4], &[1, 1, 4, 4]),
        false,
    );
    let output = pool.forward(&input).unwrap();
    assert_eq!(output.shape(), vec![1, 1, 2, 2]);
}

#[test]
fn test_maxpool2d_values() {
    let pool = MaxPool2d::new(2);
    #[rustfmt::skip]
    let data = vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ];
    let input = Variable::new(Tensor::from_vec(data, &[1, 1, 4, 4]), false);
    let output = pool.forward(&input).unwrap();
    let result = output.tensor().to_vec_f32();
    // Max of each 2x2 block
    assert_eq!(result, vec![6.0, 8.0, 14.0, 16.0]);
}

#[test]
fn test_maxpool2d_backward() {
    let pool = MaxPool2d::new(2);
    #[rustfmt::skip]
    let data = vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ];
    let input = Variable::new(Tensor::from_vec(data, &[1, 1, 4, 4]), true);
    let output = pool.forward(&input).unwrap();
    let loss = output.sum().unwrap();
    loss.backward().unwrap();

    let grad = input.grad().unwrap().to_vec_f32();
    // Gradients should only be at max positions (6, 8, 14, 16)
    // Position 5 (idx 5), 7 (idx 7), 13 (idx 13), 15 (idx 15)
    assert_eq!(grad[5], 1.0); // position of 6.0
    assert_eq!(grad[7], 1.0); // position of 8.0
    assert_eq!(grad[13], 1.0); // position of 14.0
    assert_eq!(grad[15], 1.0); // position of 16.0
                               // Non-max positions should be 0
    assert_eq!(grad[0], 0.0);
    assert_eq!(grad[4], 0.0);
}

#[test]
fn test_maxpool2d_no_parameters() {
    let pool = MaxPool2d::new(2);
    assert!(pool.parameters().is_empty());
}

#[test]
fn test_maxpool2d_with_stride() {
    let pool = MaxPool2d::with_stride(3, 1); // overlapping pools
    let input = Variable::new(
        Tensor::from_vec(vec![0.0; 1 * 1 * 5 * 5], &[1, 1, 5, 5]),
        false,
    );
    let output = pool.forward(&input).unwrap();
    // oH = (5 - 3) / 1 + 1 = 3
    assert_eq!(output.shape(), vec![1, 1, 3, 3]);
}

// ---- Flatten tests ----

#[test]
fn test_flatten_4d() {
    let flatten = Flatten::new();
    let input = Variable::new(
        Tensor::from_vec(vec![0.0; 2 * 3 * 4 * 5], &[2, 3, 4, 5]),
        false,
    );
    let output = flatten.forward(&input).unwrap();
    assert_eq!(output.shape(), vec![2, 60]); // 3*4*5 = 60
}

#[test]
fn test_flatten_already_flat() {
    let flatten = Flatten::new();
    let input = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]), false);
    let output = flatten.forward(&input).unwrap();
    assert_eq!(output.shape(), vec![1, 3]);
}

#[test]
fn test_flatten_backward() {
    let flatten = Flatten::new();
    let input = Variable::new(
        Tensor::from_vec(vec![1.0; 1 * 2 * 3 * 3], &[1, 2, 3, 3]),
        true,
    );
    let output = flatten.forward(&input).unwrap();
    assert_eq!(output.shape(), vec![1, 18]);
    let loss = output.sum().unwrap();
    loss.backward().unwrap();
    // Gradient should be 1.0 everywhere, reshaped back to original shape
    let grad = input.grad().unwrap();
    assert_eq!(grad.shape(), &[1, 2, 3, 3]);
}

#[test]
fn test_flatten_no_parameters() {
    let flatten = Flatten::new();
    assert!(flatten.parameters().is_empty());
}

// ---- Reshape autograd tests ----

#[test]
fn test_reshape_forward() {
    let x = Variable::new(
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]),
        true,
    );
    let y = x.reshape(&[3, 2]).unwrap();
    assert_eq!(y.shape(), vec![3, 2]);
    let data = y.tensor().to_vec_f32();
    assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_reshape_backward() {
    let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]), true);
    let y = x.reshape(&[4]).unwrap();
    let loss = y.sum().unwrap();
    loss.backward().unwrap();
    let grad = x.grad().unwrap();
    assert_eq!(grad.shape(), &[2, 2]); // gradient reshaped back
    assert_eq!(grad.to_vec_f32(), vec![1.0, 1.0, 1.0, 1.0]);
}

// ---- CNN integration tests ----

#[test]
fn test_conv_pool_flatten_pipeline() {
    // Test the full CNN pipeline: Conv2d -> ReLU -> MaxPool2d -> Flatten -> Linear
    let conv = Conv2d::new(1, 4, 3); // [1, 1, 8, 8] -> [1, 4, 6, 6]
    let relu = ReLU::new();
    let pool = MaxPool2d::new(2); // [1, 4, 6, 6] -> [1, 4, 3, 3]
    let flatten = Flatten::new(); // [1, 4, 3, 3] -> [1, 36]
    let fc = Linear::new(36, 2); // [1, 36] -> [1, 2]

    let input = Variable::new(Tensor::from_vec(vec![1.0; 64], &[1, 1, 8, 8]), true);
    let x = conv.forward(&input).unwrap();
    assert_eq!(x.shape(), vec![1, 4, 6, 6]);
    let x = relu.forward(&x).unwrap();
    let x = pool.forward(&x).unwrap();
    assert_eq!(x.shape(), vec![1, 4, 3, 3]);
    let x = flatten.forward(&x).unwrap();
    assert_eq!(x.shape(), vec![1, 36]);
    let x = fc.forward(&x).unwrap();
    assert_eq!(x.shape(), vec![1, 2]);
}

#[test]
fn test_cnn_backward() {
    // Full backward through CNN pipeline
    let conv = Conv2d::new(1, 2, 3);
    let pool = MaxPool2d::new(2);
    let flatten = Flatten::new();
    let fc = Linear::new(2, 1); // After conv 5x5->3x3, pool->1x1, 2 channels = 2

    let input = Variable::new(Tensor::from_vec(vec![1.0; 25], &[1, 1, 5, 5]), true);
    let x = conv.forward(&input).unwrap();
    let x = x.relu();
    let x = pool.forward(&x).unwrap();
    let x = flatten.forward(&x).unwrap();
    let x = fc.forward(&x).unwrap();
    let loss = x.sum().unwrap();
    loss.backward().unwrap();

    // Conv weight should have gradient
    assert!(conv.weight.grad().is_some());
    // FC weight should have gradient
    assert!(fc.weight.grad().is_some());
    // Input should have gradient
    assert!(input.grad().is_some());
}

#[test]
fn test_cnn_training_synthetic() {
    // Train a tiny CNN to distinguish two synthetic patterns:
    // Pattern A (label 0): all zeros
    // Pattern B (label 1): all ones
    // This is trivially separable — the CNN should learn it fast.

    let conv = Conv2d::new(1, 2, 3); // [B, 1, 5, 5] -> [B, 2, 3, 3]
    let pool = MaxPool2d::new(3); // [B, 2, 3, 3] -> [B, 2, 1, 1]
    let flatten = Flatten::new(); // [B, 2, 1, 1] -> [B, 2]
    let fc = Linear::new(2, 1); // [B, 2] -> [B, 1]

    let loss_fn = MSELoss::new();

    let mut all_params = vec![];
    all_params.extend(conv.parameters());
    all_params.extend(fc.parameters());
    let mut optimizer = Adam::new(all_params, 0.02);

    let pattern_a = vec![0.0f32; 25]; // all zeros
    let pattern_b = vec![1.0f32; 25]; // all ones

    let mut last_loss = f32::MAX;

    for _epoch in 0..200 {
        let mut epoch_loss = 0.0;

        for (pattern, target_val) in [(&pattern_a, 0.0f32), (&pattern_b, 1.0f32)] {
            optimizer.zero_grad();

            let input = Variable::new(Tensor::from_vec(pattern.clone(), &[1, 1, 5, 5]), false);
            let target = Variable::new(Tensor::from_vec(vec![target_val], &[1, 1]), false);

            let x = conv.forward(&input).unwrap();
            let x = x.relu();
            let x = pool.forward(&x).unwrap();
            let x = flatten.forward(&x).unwrap();
            let x = fc.forward(&x).unwrap();
            let x = x.sigmoid().unwrap();

            let loss = loss_fn.forward(&x, &target).unwrap();
            epoch_loss += loss.tensor().to_vec_f32()[0];

            loss.backward().unwrap();
            optimizer.step().unwrap();
        }

        last_loss = epoch_loss / 2.0;
    }

    assert!(
        last_loss < 0.1,
        "CNN should learn synthetic patterns, but final loss = {}",
        last_loss
    );

    // Verify predictions
    let input_a = Variable::new(Tensor::from_vec(pattern_a, &[1, 1, 5, 5]), false);
    let x = conv.forward(&input_a).unwrap();
    let x = x.relu();
    let x = pool.forward(&x).unwrap();
    let x = flatten.forward(&x).unwrap();
    let x = fc.forward(&x).unwrap();
    let pred_a = x.sigmoid().unwrap().tensor().to_vec_f32()[0];

    let input_b = Variable::new(Tensor::from_vec(pattern_b, &[1, 1, 5, 5]), false);
    let x = conv.forward(&input_b).unwrap();
    let x = x.relu();
    let x = pool.forward(&x).unwrap();
    let x = flatten.forward(&x).unwrap();
    let x = fc.forward(&x).unwrap();
    let pred_b = x.sigmoid().unwrap().tensor().to_vec_f32()[0];

    assert!(
        pred_a < 0.3,
        "Pattern A (zeros) should predict ~0, got {}",
        pred_a
    );
    assert!(
        pred_b > 0.7,
        "Pattern B (ones) should predict ~1, got {}",
        pred_b
    );
}

// ---- Debug format tests ----

#[test]
fn test_conv2d_debug() {
    let conv = Conv2d::with_options(3, 16, 5, 2, 1);
    let dbg = format!("{:?}", conv);
    assert!(dbg.contains("Conv2d"));
    assert!(dbg.contains("in=3"));
    assert!(dbg.contains("out=16"));
    assert!(dbg.contains("kernel=5"));
}

#[test]
fn test_maxpool2d_debug() {
    let pool = MaxPool2d::new(2);
    let dbg = format!("{:?}", pool);
    assert!(dbg.contains("MaxPool2d"));
    assert!(dbg.contains("kernel_size=2"));
}

#[test]
fn test_flatten_debug() {
    let flatten = Flatten::new();
    let dbg = format!("{:?}", flatten);
    assert!(dbg.contains("Flatten"));
}

// ---- XOR learning test (the classic) ----

#[test]
fn test_xor_training() {
    // XOR: the simplest problem that requires a hidden layer.
    // Inputs: [0,0] -> 0, [0,1] -> 1, [1,0] -> 1, [1,1] -> 0
    let model = Sequential::new(vec![
        Box::new(Linear::new(2, 8)),
        Box::new(Tanh::new()),
        Box::new(Linear::new(8, 1)),
        Box::new(Sigmoid::new()),
    ]);

    let loss_fn = MSELoss::new();
    let mut optimizer = Adam::new(model.parameters(), 0.05);

    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let targets = vec![0.0, 1.0, 1.0, 0.0];

    let mut last_loss = f32::MAX;

    // Train for enough epochs to learn XOR
    for _epoch in 0..500 {
        let mut epoch_loss = 0.0;

        for (inp, &tgt) in inputs.iter().zip(targets.iter()) {
            optimizer.zero_grad();

            let x = Variable::new(Tensor::from_vec(inp.clone(), &[1, 2]), false);
            let y = Variable::new(Tensor::from_vec(vec![tgt], &[1, 1]), false);

            let pred = model.forward(&x).unwrap();
            let loss = loss_fn.forward(&pred, &y).unwrap();
            epoch_loss += loss.tensor().to_vec_f32()[0];

            loss.backward().unwrap();
            optimizer.step().unwrap();
        }

        last_loss = epoch_loss / 4.0;
    }

    // After 500 epochs, loss should be very low
    assert!(
        last_loss < 0.05,
        "XOR should be learned, but final loss = {}",
        last_loss
    );

    // Verify predictions
    for (inp, &expected) in inputs.iter().zip(targets.iter()) {
        let x = Variable::new(Tensor::from_vec(inp.clone(), &[1, 2]), false);
        let pred = model.forward(&x).unwrap();
        let val = pred.tensor().to_vec_f32()[0];
        assert!(
            (val - expected).abs() < 0.2,
            "XOR({:?}) = {}, expected ~{}",
            inp,
            val,
            expected
        );
    }
}
