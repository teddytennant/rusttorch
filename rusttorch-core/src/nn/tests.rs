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

    let conv = Conv2d::new(1, 4, 3); // [B, 1, 5, 5] -> [B, 4, 3, 3]
    let pool = MaxPool2d::new(3); // [B, 4, 3, 3] -> [B, 4, 1, 1]
    let flatten = Flatten::new(); // [B, 4, 1, 1] -> [B, 4]
    let fc = Linear::new(4, 1); // [B, 4] -> [B, 1]

    let loss_fn = MSELoss::new();

    let mut all_params = vec![];
    all_params.extend(conv.parameters());
    all_params.extend(fc.parameters());
    let mut optimizer = Adam::new(all_params, 0.02);

    let pattern_a = vec![0.0f32; 25]; // all zeros
    let pattern_b = vec![1.0f32; 25]; // all ones

    let mut last_loss = f32::MAX;

    for _epoch in 0..400 {
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
        last_loss < 0.15,
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
        pred_a < 0.4,
        "Pattern A (zeros) should predict ~0, got {}",
        pred_a
    );
    assert!(
        pred_b > 0.6,
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

// ---- BatchNorm2d tests ----

#[test]
fn test_batchnorm2d_output_shape() {
    let bn = BatchNorm2d::new(3);
    let input = Variable::new(
        Tensor::from_vec(vec![1.0; 2 * 3 * 4 * 4], &[2, 3, 4, 4]),
        false,
    );
    let output = bn.forward(&input).unwrap();
    assert_eq!(output.shape(), vec![2, 3, 4, 4]);
}

#[test]
fn test_batchnorm2d_normalizes_to_zero_mean() {
    let bn = BatchNorm2d::new(1);
    // Input: 2 samples, 1 channel, 2x2 spatial
    // Values: [1, 2, 3, 4] and [5, 6, 7, 8]
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let input = Variable::new(Tensor::from_vec(data, &[2, 1, 2, 2]), false);
    let output = bn.forward(&input).unwrap();
    let out_data = output.tensor().to_vec_f32();

    // The output should have approximately zero mean across the batch
    let mean: f32 = out_data.iter().sum::<f32>() / out_data.len() as f32;
    assert!(
        mean.abs() < 1e-5,
        "BatchNorm output mean should be ~0, got {}",
        mean
    );
}

#[test]
fn test_batchnorm2d_unit_variance() {
    let bn = BatchNorm2d::new(1);
    let data: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let input = Variable::new(Tensor::from_vec(data, &[2, 1, 4, 4]), false);
    let output = bn.forward(&input).unwrap();
    let out_data = output.tensor().to_vec_f32();

    // With default gamma=1, beta=0, variance of output should be ~1
    let mean: f32 = out_data.iter().sum::<f32>() / out_data.len() as f32;
    let var: f32 = out_data
        .iter()
        .map(|&x| (x - mean) * (x - mean))
        .sum::<f32>()
        / out_data.len() as f32;
    assert!(
        (var - 1.0).abs() < 0.1,
        "BatchNorm output variance should be ~1, got {}",
        var
    );
}

#[test]
fn test_batchnorm2d_parameters() {
    let bn = BatchNorm2d::new(16);
    let params = bn.parameters();
    assert_eq!(params.len(), 2); // weight (gamma) + bias (beta)
    assert_eq!(params[0].shape(), vec![16]);
    assert_eq!(params[1].shape(), vec![16]);
}

#[test]
fn test_batchnorm2d_training_vs_eval() {
    let bn = BatchNorm2d::new(1);

    // Run a few training forward passes to accumulate running stats
    for i in 0..5 {
        let data: Vec<f32> = (0..8).map(|j| (i * 8 + j) as f32).collect();
        let input = Variable::new(Tensor::from_vec(data, &[2, 1, 2, 2]), false);
        bn.forward(&input).unwrap();
    }

    // Same input, different modes should give different outputs
    let test_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let test_input = Variable::new(Tensor::from_vec(test_data.clone(), &[2, 1, 2, 2]), false);

    bn.train();
    let train_output = bn.forward(&test_input).unwrap().tensor().to_vec_f32();

    bn.eval();
    let eval_input = Variable::new(Tensor::from_vec(test_data, &[2, 1, 2, 2]), false);
    let eval_output = bn.forward(&eval_input).unwrap().tensor().to_vec_f32();

    // They should differ because training uses batch stats while eval uses running stats
    let differs = train_output
        .iter()
        .zip(eval_output.iter())
        .any(|(a, b)| (a - b).abs() > 1e-5);
    assert!(
        differs,
        "Training and eval outputs should differ (different normalization stats)"
    );
}

#[test]
fn test_batchnorm2d_backward() {
    let bn = BatchNorm2d::new(2);
    let data: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
    let input = Variable::new(Tensor::from_vec(data, &[2, 2, 2, 2]), true);

    let output = bn.forward(&input).unwrap();
    let loss = output.sum().unwrap();
    loss.backward().unwrap();

    // Input should have gradients
    let input_grad = input.grad();
    assert!(input_grad.is_some(), "Input should have gradients");
    assert_eq!(input_grad.unwrap().shape(), &[2, 2, 2, 2]);

    // Weight and bias should have gradients
    let weight_grad = bn.weight.grad();
    assert!(
        weight_grad.is_some(),
        "Weight (gamma) should have gradients"
    );
    let bias_grad = bn.bias.grad();
    assert!(bias_grad.is_some(), "Bias (beta) should have gradients");
}

#[test]
fn test_batchnorm2d_gradient_correctness() {
    // Numerical gradient check for BatchNorm2d
    let eps_num = 1e-3;

    let bn = BatchNorm2d::new(1);
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    // Analytical gradient
    let input = Variable::new(Tensor::from_vec(data.clone(), &[2, 1, 2, 2]), true);
    let output = bn.forward(&input).unwrap();
    let loss = output.sum().unwrap();
    loss.backward().unwrap();
    let analytical_grad = input.grad().unwrap().to_vec_f32();

    // Numerical gradient (central differences)
    for i in 0..data.len() {
        let mut data_plus = data.clone();
        data_plus[i] += eps_num;
        let bn_plus = BatchNorm2d::new(1);
        let input_plus = Variable::new(Tensor::from_vec(data_plus, &[2, 1, 2, 2]), false);
        let loss_plus = bn_plus.forward(&input_plus).unwrap().sum().unwrap();
        let lp = loss_plus.tensor().to_vec_f32()[0];

        let mut data_minus = data.clone();
        data_minus[i] -= eps_num;
        let bn_minus = BatchNorm2d::new(1);
        let input_minus = Variable::new(Tensor::from_vec(data_minus, &[2, 1, 2, 2]), false);
        let loss_minus = bn_minus.forward(&input_minus).unwrap().sum().unwrap();
        let lm = loss_minus.tensor().to_vec_f32()[0];

        let numerical = (lp - lm) / (2.0 * eps_num);
        assert!(
            (analytical_grad[i] - numerical).abs() < 0.05,
            "Gradient mismatch at index {}: analytical={}, numerical={}",
            i,
            analytical_grad[i],
            numerical
        );
    }
}

#[test]
fn test_batchnorm2d_running_stats_update() {
    let bn = BatchNorm2d::new(2);

    // Initially running_mean = 0, running_var = 1
    assert_eq!(bn.running_mean(), vec![0.0, 0.0]);
    assert_eq!(bn.running_var(), vec![1.0, 1.0]);

    // Forward pass should update running stats
    let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let input = Variable::new(Tensor::from_vec(data, &[2, 2, 2, 2]), false);
    bn.forward(&input).unwrap();

    // Running mean should no longer be all zeros
    let rm = bn.running_mean();
    assert!(
        rm[0].abs() > 1e-6 || rm[1].abs() > 1e-6,
        "Running mean should be updated after forward pass"
    );
}

#[test]
fn test_batchnorm2d_rejects_non_4d() {
    let bn = BatchNorm2d::new(3);
    let input = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]), false);
    assert!(bn.forward(&input).is_err());
}

#[test]
fn test_batchnorm2d_debug() {
    let bn = BatchNorm2d::new(16);
    let dbg = format!("{:?}", bn);
    assert!(dbg.contains("BatchNorm2d"));
    assert!(dbg.contains("num_features=16"));
    assert!(dbg.contains("training=true"));
}

// ---- Dropout tests ----

#[test]
fn test_dropout_output_shape() {
    let dropout = Dropout::new(0.5);
    let input = Variable::new(Tensor::from_vec(vec![1.0; 12], &[3, 4]), false);
    let output = dropout.forward(&input).unwrap();
    assert_eq!(output.shape(), vec![3, 4]);
}

#[test]
fn test_dropout_zeros_elements() {
    let dropout = Dropout::new(0.5);
    // Large input so we can statistically verify dropout
    let data: Vec<f32> = vec![1.0; 1000];
    let input = Variable::new(Tensor::from_vec(data, &[1, 1000]), false);
    let output = dropout.forward(&input).unwrap();
    let out_data = output.tensor().to_vec_f32();

    let num_zeros = out_data.iter().filter(|&&x| x == 0.0).count();
    // With p=0.5, expect ~500 zeros. Allow wide margin for randomness.
    assert!(
        num_zeros > 300 && num_zeros < 700,
        "Expected ~500 zeros with p=0.5, got {}",
        num_zeros
    );

    // Non-zero elements should be scaled by 1/(1-0.5) = 2.0
    let non_zeros: Vec<&f32> = out_data.iter().filter(|&&x| x != 0.0).collect();
    assert!(!non_zeros.is_empty());
    for &&v in &non_zeros {
        assert!(
            (v - 2.0).abs() < 1e-5,
            "Non-zero elements should be scaled to 2.0, got {}",
            v
        );
    }
}

#[test]
fn test_dropout_eval_passthrough() {
    let dropout = Dropout::new(0.5);
    dropout.eval();

    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let input = Variable::new(Tensor::from_vec(data.clone(), &[2, 2]), false);
    let output = dropout.forward(&input).unwrap();
    let out_data = output.tensor().to_vec_f32();

    // In eval mode, output should equal input exactly
    assert_eq!(out_data, data);
}

#[test]
fn test_dropout_p_zero_passthrough() {
    let dropout = Dropout::new(0.0);
    let data: Vec<f32> = vec![1.0, 2.0, 3.0];
    let input = Variable::new(Tensor::from_vec(data.clone(), &[3]), false);
    let output = dropout.forward(&input).unwrap();
    assert_eq!(output.tensor().to_vec_f32(), data);
}

#[test]
fn test_dropout_no_parameters() {
    let dropout = Dropout::new(0.5);
    assert_eq!(dropout.parameters().len(), 0);
}

#[test]
fn test_dropout_backward() {
    let dropout = Dropout::new(0.3);
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let input = Variable::new(Tensor::from_vec(data, &[2, 4]), true);

    let output = dropout.forward(&input).unwrap();
    let loss = output.sum().unwrap();
    loss.backward().unwrap();

    let grad = input.grad().unwrap().to_vec_f32();
    let out_data = output.tensor().to_vec_f32();

    // Gradient should match the mask: zero where output was zero, scale where output was nonzero
    let scale = 1.0 / (1.0 - 0.3);
    for (i, (&g, &o)) in grad.iter().zip(out_data.iter()).enumerate() {
        if o == 0.0 {
            assert_eq!(
                g, 0.0,
                "Gradient should be 0 where output was dropped (idx {})",
                i
            );
        } else {
            assert!(
                (g - scale).abs() < 1e-5,
                "Gradient should be {} where kept (idx {}), got {}",
                scale,
                i,
                g
            );
        }
    }
}

#[test]
fn test_dropout_training_mode_toggle() {
    let dropout = Dropout::new(0.5);
    assert!(dropout.is_training());

    dropout.eval();
    assert!(!dropout.is_training());

    dropout.train();
    assert!(dropout.is_training());
}

#[test]
fn test_dropout_debug() {
    let dropout = Dropout::new(0.3);
    let dbg = format!("{:?}", dropout);
    assert!(dbg.contains("Dropout"));
    assert!(dbg.contains("p=0.3"));
    assert!(dbg.contains("training=true"));
}

#[test]
#[should_panic(expected = "Dropout probability must be in [0, 1)")]
fn test_dropout_invalid_p() {
    Dropout::new(1.0);
}

// ---- CNN with BatchNorm training test ----

#[test]
fn test_cnn_with_batchnorm_training() {
    // Train a CNN with BatchNorm to distinguish two synthetic patterns.
    // BatchNorm should help training converge faster.
    let conv = Conv2d::new(1, 2, 3); // [B, 1, 5, 5] -> [B, 2, 3, 3]
    let bn = BatchNorm2d::new(2);
    let pool = MaxPool2d::new(3); // [B, 2, 3, 3] -> [B, 2, 1, 1]
    let flatten = Flatten::new();
    let fc = Linear::new(2, 1);

    let loss_fn = MSELoss::new();
    let mut all_params = vec![];
    all_params.extend(conv.parameters());
    all_params.extend(bn.parameters());
    all_params.extend(fc.parameters());
    let mut optimizer = Adam::new(all_params, 0.02);

    // Use different patterns to ensure learnability
    let pattern_a = vec![0.0f32; 25];
    let pattern_b = vec![1.0f32; 25];

    let mut last_loss = f32::MAX;

    for _epoch in 0..300 {
        // Batch of 2 (both patterns together) so BatchNorm has real statistics
        let batch_data: Vec<f32> = pattern_a.iter().chain(pattern_b.iter()).copied().collect();
        let batch_targets = vec![0.0f32, 1.0f32];

        optimizer.zero_grad();

        let input = Variable::new(Tensor::from_vec(batch_data, &[2, 1, 5, 5]), false);
        let target = Variable::new(Tensor::from_vec(batch_targets, &[2, 1]), false);

        let x = conv.forward(&input).unwrap();
        let x = bn.forward(&x).unwrap();
        let x = x.relu();
        let x = pool.forward(&x).unwrap();
        let x = flatten.forward(&x).unwrap();
        let x = fc.forward(&x).unwrap();
        let x = x.sigmoid().unwrap();

        let loss = loss_fn.forward(&x, &target).unwrap();
        last_loss = loss.tensor().to_vec_f32()[0];

        loss.backward().unwrap();
        optimizer.step().unwrap();
    }

    assert!(
        last_loss < 0.15,
        "CNN+BatchNorm should learn synthetic patterns, but final loss = {}",
        last_loss
    );
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

// ---- LogSoftmax tests ----

#[test]
fn test_log_softmax_sums_to_one() {
    // exp(log_softmax(x)) should sum to 1
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let input = Variable::new(Tensor::from_vec(data, &[1, 4]), false);
    let output = input.log_softmax(1).unwrap();
    let out_data = output.tensor().to_vec_f32();

    // exp of log_softmax should sum to 1
    let sum: f32 = out_data.iter().map(|&x| x.exp()).sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "exp(log_softmax) should sum to 1, got {}",
        sum
    );
}

#[test]
fn test_log_softmax_values_negative() {
    // log_softmax values should all be <= 0
    let data = vec![1.0f32, 2.0, 3.0];
    let input = Variable::new(Tensor::from_vec(data, &[1, 3]), false);
    let output = input.log_softmax(1).unwrap();
    let out_data = output.tensor().to_vec_f32();

    for &v in &out_data {
        assert!(v <= 0.0, "log_softmax values should be <= 0, got {}", v);
    }
}

#[test]
fn test_log_softmax_numerical_stability() {
    // Large values shouldn't cause overflow
    let data = vec![1000.0f32, 1001.0, 1002.0];
    let input = Variable::new(Tensor::from_vec(data, &[1, 3]), false);
    let output = input.log_softmax(1).unwrap();
    let out_data = output.tensor().to_vec_f32();

    for &v in &out_data {
        assert!(v.is_finite(), "log_softmax should be finite, got {}", v);
    }
    let sum: f32 = out_data.iter().map(|&x| x.exp()).sum();
    assert!((sum - 1.0).abs() < 1e-4);
}

#[test]
fn test_log_softmax_batch() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input = Variable::new(Tensor::from_vec(data, &[2, 3]), false);
    let output = input.log_softmax(1).unwrap();
    let out_data = output.tensor().to_vec_f32();

    // Each row should independently sum (exp) to 1
    let sum1: f32 = out_data[0..3].iter().map(|&x| x.exp()).sum();
    let sum2: f32 = out_data[3..6].iter().map(|&x| x.exp()).sum();
    assert!((sum1 - 1.0).abs() < 1e-5);
    assert!((sum2 - 1.0).abs() < 1e-5);
}

#[test]
fn test_log_softmax_backward() {
    let data = vec![1.0f32, 2.0, 3.0];
    let input = Variable::new(Tensor::from_vec(data, &[1, 3]), true);
    let output = input.log_softmax(1).unwrap();
    let loss = output.sum().unwrap();
    loss.backward().unwrap();

    let grad = input.grad();
    assert!(grad.is_some(), "Should have gradients");

    // For log_softmax, sum of all outputs = sum(x_i - log(sum(exp)))
    // d/dx_i (sum log_softmax) = 1 - n * softmax_i
    // where n is the number of classes
    let grad_data = grad.unwrap().to_vec_f32();
    assert_eq!(grad_data.len(), 3);
}

#[test]
fn test_log_softmax_gradient_numerical() {
    let eps = 1e-3;
    let data = vec![1.0f32, 2.0, 3.0, 0.5, 1.5, 2.5];
    let input = Variable::new(Tensor::from_vec(data.clone(), &[2, 3]), true);
    let output = input.log_softmax(1).unwrap();
    let loss = output.sum().unwrap();
    loss.backward().unwrap();
    let analytical = input.grad().unwrap().to_vec_f32();

    for i in 0..data.len() {
        let mut dp = data.clone();
        dp[i] += eps;
        let inp_p = Variable::new(Tensor::from_vec(dp, &[2, 3]), false);
        let lp = inp_p.log_softmax(1).unwrap().sum().unwrap();

        let mut dm = data.clone();
        dm[i] -= eps;
        let inp_m = Variable::new(Tensor::from_vec(dm, &[2, 3]), false);
        let lm = inp_m.log_softmax(1).unwrap().sum().unwrap();

        let numerical = (lp.tensor().to_vec_f32()[0] - lm.tensor().to_vec_f32()[0]) / (2.0 * eps);
        assert!(
            (analytical[i] - numerical).abs() < 0.01,
            "LogSoftmax grad mismatch at {}: analytical={}, numerical={}",
            i,
            analytical[i],
            numerical
        );
    }
}

// ---- CrossEntropyLoss tests ----

#[test]
fn test_cross_entropy_loss_basic() {
    let loss_fn = CrossEntropyLoss::new();

    // Perfect prediction: logits strongly favor class 0
    let logits = Variable::new(Tensor::from_vec(vec![10.0, -10.0, -10.0], &[1, 3]), false);
    let target = Variable::new(Tensor::from_vec(vec![1.0, 0.0, 0.0], &[1, 3]), false);
    let loss = loss_fn.forward(&logits, &target).unwrap();
    let loss_val = loss.tensor().to_vec_f32()[0];

    // Loss should be very small (correct prediction)
    assert!(
        loss_val < 0.01,
        "Loss for correct prediction should be ~0, got {}",
        loss_val
    );
}

#[test]
fn test_cross_entropy_loss_wrong_prediction() {
    let loss_fn = CrossEntropyLoss::new();

    // Wrong prediction: logits favor class 2 but target is class 0
    let logits = Variable::new(Tensor::from_vec(vec![-10.0, -10.0, 10.0], &[1, 3]), false);
    let target = Variable::new(Tensor::from_vec(vec![1.0, 0.0, 0.0], &[1, 3]), false);
    let loss = loss_fn.forward(&logits, &target).unwrap();
    let loss_val = loss.tensor().to_vec_f32()[0];

    // Loss should be large (wrong prediction)
    assert!(
        loss_val > 1.0,
        "Loss for wrong prediction should be large, got {}",
        loss_val
    );
}

#[test]
fn test_cross_entropy_loss_batch() {
    let loss_fn = CrossEntropyLoss::new();

    // Batch of 2: one correct, one wrong
    let logits = Variable::new(
        Tensor::from_vec(vec![5.0, -5.0, -5.0, -5.0, -5.0, 5.0], &[2, 3]),
        false,
    );
    // Both target class 0
    let target = Variable::new(
        Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0], &[2, 3]),
        false,
    );
    let loss = loss_fn.forward(&logits, &target).unwrap();
    let loss_val = loss.tensor().to_vec_f32()[0];

    // Average of low loss + high loss
    assert!(
        loss_val > 0.1,
        "Mixed batch loss should be moderate, got {}",
        loss_val
    );
}

#[test]
fn test_cross_entropy_loss_backward() {
    let loss_fn = CrossEntropyLoss::new();

    let logits = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]), true);
    let target = Variable::new(Tensor::from_vec(vec![1.0, 0.0, 0.0], &[1, 3]), false);

    let loss = loss_fn.forward(&logits, &target).unwrap();
    loss.backward().unwrap();

    let grad = logits.grad();
    assert!(grad.is_some(), "Logits should have gradients");

    let grad_data = grad.unwrap().to_vec_f32();
    // The gradient for the correct class should be negative (wants to increase logit)
    // The gradient for wrong classes should be positive (wants to decrease logits)
    // Specifically: grad = (softmax(x) - target) / batch_size
    assert!(
        grad_data[0] < 0.0,
        "Gradient for correct class should be negative, got {}",
        grad_data[0]
    );
    assert!(
        grad_data[1] > 0.0,
        "Gradient for wrong class should be positive, got {}",
        grad_data[1]
    );
}

#[test]
fn test_cross_entropy_gradient_numerical() {
    let eps = 1e-3;
    let logits_data = vec![1.0f32, 2.0, 0.5];
    let target_data = vec![0.0f32, 1.0, 0.0];

    let logits = Variable::new(Tensor::from_vec(logits_data.clone(), &[1, 3]), true);
    let target = Variable::new(Tensor::from_vec(target_data.clone(), &[1, 3]), false);
    let loss_fn = CrossEntropyLoss::new();
    let loss = loss_fn.forward(&logits, &target).unwrap();
    loss.backward().unwrap();
    let analytical = logits.grad().unwrap().to_vec_f32();

    for i in 0..logits_data.len() {
        let mut dp = logits_data.clone();
        dp[i] += eps;
        let lp_logits = Variable::new(Tensor::from_vec(dp, &[1, 3]), false);
        let lp_target = Variable::new(Tensor::from_vec(target_data.clone(), &[1, 3]), false);
        let lp = loss_fn.forward(&lp_logits, &lp_target).unwrap();

        let mut dm = logits_data.clone();
        dm[i] -= eps;
        let lm_logits = Variable::new(Tensor::from_vec(dm, &[1, 3]), false);
        let lm_target = Variable::new(Tensor::from_vec(target_data.clone(), &[1, 3]), false);
        let lm = loss_fn.forward(&lm_logits, &lm_target).unwrap();

        let numerical = (lp.tensor().to_vec_f32()[0] - lm.tensor().to_vec_f32()[0]) / (2.0 * eps);
        assert!(
            (analytical[i] - numerical).abs() < 0.01,
            "CE grad mismatch at {}: analytical={}, numerical={}",
            i,
            analytical[i],
            numerical
        );
    }
}

// ---- Multi-class classification training test ----

#[test]
fn test_multiclass_classification_training() {
    // Train a 3-class classifier on synthetic linearly separable data.
    // Class 0: x > 0, y > 0  (quadrant I)
    // Class 1: x < 0          (left half)
    // Class 2: x > 0, y < 0  (quadrant IV)

    let model = Sequential::new(vec![
        Box::new(Linear::new(2, 8)),
        Box::new(ReLU::new()),
        Box::new(Linear::new(8, 3)), // 3 classes, raw logits (no softmax)
    ]);

    let loss_fn = CrossEntropyLoss::new();
    let mut optimizer = Adam::new(model.parameters(), 0.05);

    // Training data: 3 points per class
    let inputs = vec![
        (vec![1.0f32, 1.0], vec![1.0, 0.0, 0.0]), // class 0
        (vec![0.5, 2.0], vec![1.0, 0.0, 0.0]),    // class 0
        (vec![2.0, 0.5], vec![1.0, 0.0, 0.0]),    // class 0
        (vec![-1.0, 0.5], vec![0.0, 1.0, 0.0]),   // class 1
        (vec![-2.0, -1.0], vec![0.0, 1.0, 0.0]),  // class 1
        (vec![-0.5, 1.5], vec![0.0, 1.0, 0.0]),   // class 1
        (vec![1.0, -1.0], vec![0.0, 0.0, 1.0]),   // class 2
        (vec![2.0, -2.0], vec![0.0, 0.0, 1.0]),   // class 2
        (vec![0.5, -1.5], vec![0.0, 0.0, 1.0]),   // class 2
    ];

    let mut last_loss = f32::MAX;

    for _epoch in 0..300 {
        let mut epoch_loss = 0.0;

        for (inp, tgt) in &inputs {
            optimizer.zero_grad();

            let x = Variable::new(Tensor::from_vec(inp.clone(), &[1, 2]), false);
            let y = Variable::new(Tensor::from_vec(tgt.clone(), &[1, 3]), false);

            let logits = model.forward(&x).unwrap();
            let loss = loss_fn.forward(&logits, &y).unwrap();
            epoch_loss += loss.tensor().to_vec_f32()[0];

            loss.backward().unwrap();
            optimizer.step().unwrap();
        }

        last_loss = epoch_loss / inputs.len() as f32;
    }

    assert!(
        last_loss < 0.1,
        "3-class classifier should converge, but final loss = {}",
        last_loss
    );

    // Verify predictions (argmax of logits should match target class)
    let mut correct = 0;
    for (inp, tgt) in &inputs {
        let x = Variable::new(Tensor::from_vec(inp.clone(), &[1, 2]), false);
        let logits = model.forward(&x).unwrap().tensor().to_vec_f32();

        let pred_class = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let true_class = tgt
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;

        if pred_class == true_class {
            correct += 1;
        }
    }

    assert!(
        correct >= 7,
        "Should classify at least 7/9 correctly, got {}/9",
        correct
    );
}
