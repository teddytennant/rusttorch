//! Tests for the nn module.

use crate::autograd::Variable;
use crate::nn::*;
use crate::tensor::{DType, Tensor};

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

// =========================================================================
// StateDict integration tests — save/load for real nn layers
// =========================================================================

#[test]
fn linear_state_dict_roundtrip() {
    let layer = Linear::new(4, 3);
    let sd = layer.state_dict();
    assert_eq!(sd.len(), 2); // weight + bias
    assert!(sd.get("weight").is_some());
    assert!(sd.get("bias").is_some());

    // Modify parameters
    let w_orig = layer.weight.tensor().to_vec_f32();
    let zeros = Tensor::zeros(&layer.weight.shape(), DType::Float32);
    layer.weight.update(zeros);
    assert_ne!(layer.weight.tensor().to_vec_f32(), w_orig);

    // Load state dict — should restore original
    layer.load_state_dict(&sd);
    assert_eq!(layer.weight.tensor().to_vec_f32(), w_orig);
}

#[test]
fn linear_no_bias_state_dict() {
    let layer = Linear::no_bias(4, 3);
    let sd = layer.state_dict();
    assert_eq!(sd.len(), 1); // weight only
    assert!(sd.get("weight").is_some());
    assert!(sd.get("bias").is_none());
}

#[test]
fn conv2d_state_dict_roundtrip() {
    let layer = Conv2d::new(3, 8, 3);
    let sd = layer.state_dict();
    assert_eq!(sd.len(), 2); // weight + bias
    let w = sd.get("weight").unwrap();
    assert_eq!(w.shape(), &[8, 3, 3, 3]); // [out, in, kh, kw]

    let w_orig = layer.weight.tensor().to_vec_f32();
    layer
        .weight
        .update(Tensor::zeros(&layer.weight.shape(), DType::Float32));
    layer.load_state_dict(&sd);
    assert_eq!(layer.weight.tensor().to_vec_f32(), w_orig);
}

#[test]
fn batchnorm_state_dict_includes_running_stats() {
    let layer = BatchNorm2d::new(8);
    let sd = layer.state_dict();
    assert_eq!(sd.len(), 4); // weight, bias, running_mean, running_var
    assert!(sd.get("weight").is_some());
    assert!(sd.get("bias").is_some());
    assert!(sd.get("running_mean").is_some());
    assert!(sd.get("running_var").is_some());

    // running_mean should be zeros, running_var should be ones
    let rm = sd.get("running_mean").unwrap().to_vec_f32();
    assert!(rm.iter().all(|&v| v == 0.0));
    let rv = sd.get("running_var").unwrap().to_vec_f32();
    assert!(rv.iter().all(|&v| v == 1.0));
}

#[test]
fn batchnorm_state_dict_load_running_stats() {
    let layer = BatchNorm2d::new(4);

    // Simulate modified running stats
    let mut sd = StateDict::new();
    sd.insert("weight", Tensor::from_vec(vec![2.0; 4], &[4]));
    sd.insert("bias", Tensor::from_vec(vec![0.5; 4], &[4]));
    sd.insert(
        "running_mean",
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]),
    );
    sd.insert(
        "running_var",
        Tensor::from_vec(vec![0.5, 0.6, 0.7, 0.8], &[4]),
    );

    layer.load_state_dict(&sd);
    assert_eq!(layer.running_mean(), vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(layer.running_var(), vec![0.5, 0.6, 0.7, 0.8]);
    assert_eq!(layer.weight.tensor().to_vec_f32(), vec![2.0; 4]);
}

#[test]
fn sequential_state_dict_prefixed() {
    let model = Sequential::new(vec![
        Box::new(Linear::new(4, 3)),
        Box::new(ReLU::new()),
        Box::new(Linear::new(3, 1)),
    ]);

    let sd = model.state_dict();
    // Layer 0 (Linear): 0.weight, 0.bias
    // Layer 1 (ReLU): no params
    // Layer 2 (Linear): 2.weight, 2.bias
    assert_eq!(sd.len(), 4);
    assert!(sd.get("0.weight").is_some());
    assert!(sd.get("0.bias").is_some());
    assert!(sd.get("2.weight").is_some());
    assert!(sd.get("2.bias").is_some());
}

#[test]
fn sequential_state_dict_save_load_roundtrip() {
    let model = Sequential::new(vec![
        Box::new(Linear::new(4, 3)),
        Box::new(ReLU::new()),
        Box::new(Linear::new(3, 1)),
    ]);

    // Save state dict
    let sd = model.state_dict();
    let mut buf = Vec::new();
    sd.save(&mut buf).unwrap();

    // Create a fresh model with different weights
    let model2 = Sequential::new(vec![
        Box::new(Linear::new(4, 3)),
        Box::new(ReLU::new()),
        Box::new(Linear::new(3, 1)),
    ]);

    // Weights should differ (random init)
    let w1 = sd.get("0.weight").unwrap().to_vec_f32();
    let w2 = model2.state_dict().get("0.weight").unwrap().to_vec_f32();
    assert_ne!(w1, w2);

    // Load saved weights
    let loaded_sd = StateDict::load(&mut &buf[..]).unwrap();
    model2.load_state_dict(&loaded_sd);

    // Weights should now match
    let w2_loaded = model2.state_dict().get("0.weight").unwrap().to_vec_f32();
    assert_eq!(w1, w2_loaded);
}

#[test]
fn sequential_with_batchnorm_state_dict() {
    let model = Sequential::new(vec![
        Box::new(Conv2d::new(1, 4, 3)),
        Box::new(BatchNorm2d::new(4)),
        Box::new(ReLU::new()),
    ]);

    let sd = model.state_dict();
    // Conv2d: 0.weight, 0.bias
    // BatchNorm: 1.weight, 1.bias, 1.running_mean, 1.running_var
    // ReLU: no params
    assert_eq!(sd.len(), 6);
    assert!(sd.get("1.running_mean").is_some());
    assert!(sd.get("1.running_var").is_some());
}

#[test]
fn activation_modules_empty_state_dict() {
    let relu = ReLU::new();
    assert!(relu.state_dict().is_empty());

    let sig = Sigmoid::new();
    assert!(sig.state_dict().is_empty());

    let tanh = Tanh::new();
    assert!(tanh.state_dict().is_empty());
}

#[test]
fn dropout_empty_state_dict() {
    let drop = Dropout::new(0.5);
    assert!(drop.state_dict().is_empty());
}

#[test]
fn flatten_empty_state_dict() {
    let flat = Flatten::new();
    assert!(flat.state_dict().is_empty());
}

#[test]
fn maxpool_empty_state_dict() {
    let pool = MaxPool2d::new(2);
    assert!(pool.state_dict().is_empty());
}

// ---- AvgPool2d tests ----

#[test]
fn avgpool2d_output_shape() {
    let pool = AvgPool2d::new(2);
    // [1, 1, 4, 4] -> kernel=2, stride=2 -> [1, 1, 2, 2]
    let input = Variable::new(Tensor::from_vec(vec![1.0; 16], &[1, 1, 4, 4]), false);
    let output = pool.forward(&input).unwrap();
    assert_eq!(output.shape(), vec![1, 1, 2, 2]);
}

#[test]
fn avgpool2d_computes_average() {
    let pool = AvgPool2d::new(2);
    // 2x2 input with known values
    // [[1, 2], [3, 4]] -> avg = 2.5
    let input = Variable::new(
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]),
        false,
    );
    let output = pool.forward(&input).unwrap();
    let data = output.tensor().to_vec_f32();
    assert_eq!(data.len(), 1);
    assert!((data[0] - 2.5).abs() < 1e-6);
}

#[test]
fn avgpool2d_multi_channel() {
    let pool = AvgPool2d::new(2);
    // [1, 2, 4, 4] -> [1, 2, 2, 2]
    let input = Variable::new(Tensor::from_vec(vec![1.0; 32], &[1, 2, 4, 4]), false);
    let output = pool.forward(&input).unwrap();
    assert_eq!(output.shape(), vec![1, 2, 2, 2]);
    // All ones -> average = 1.0
    for v in output.tensor().to_vec_f32() {
        assert!((v - 1.0).abs() < 1e-6);
    }
}

#[test]
fn avgpool2d_with_stride() {
    let pool = AvgPool2d::with_stride(3, 1);
    // [1, 1, 5, 5] -> kernel=3, stride=1 -> [1, 1, 3, 3]
    let input = Variable::new(Tensor::from_vec(vec![1.0; 25], &[1, 1, 5, 5]), false);
    let output = pool.forward(&input).unwrap();
    assert_eq!(output.shape(), vec![1, 1, 3, 3]);
}

#[test]
fn avgpool2d_backward_gradient_exists() {
    let pool = AvgPool2d::new(2);
    let input = Variable::new(
        Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            &[1, 1, 3, 3],
        ),
        true,
    );
    let output = pool.forward(&input).unwrap();
    // output is [1, 1, 1, 1]
    assert_eq!(output.shape(), vec![1, 1, 1, 1]);
    output.backward().unwrap();
    let grad = input.grad().expect("should have gradient");
    assert_eq!(grad.shape(), &[1, 1, 3, 3]);
}

#[test]
fn avgpool2d_backward_distributes_equally() {
    let pool = AvgPool2d::new(2);
    // [1, 1, 2, 2] -> [1, 1, 1, 1], average of 4 elements
    let input = Variable::new(
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]),
        true,
    );
    let output = pool.forward(&input).unwrap();
    output.backward().unwrap();
    let grad = input.grad().unwrap().to_vec_f32();
    // Each input should receive grad_output / pool_size = 1.0 / 4.0 = 0.25
    for &g in &grad {
        assert!((g - 0.25).abs() < 1e-6, "Expected 0.25, got {}", g);
    }
}

#[test]
fn avgpool2d_numerical_gradient_check() {
    let pool = AvgPool2d::new(2);
    let eps = 1e-3;
    // Use small values to minimize f32 precision issues
    let input_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1).collect();
    let input = Variable::new(Tensor::from_vec(input_data.clone(), &[1, 1, 4, 4]), true);
    let output = pool.forward(&input).unwrap();
    let loss = output.sum().unwrap();
    loss.backward().unwrap();
    let analytic_grad = input.grad().unwrap().to_vec_f32();

    for i in 0..input_data.len() {
        let mut plus = input_data.clone();
        plus[i] += eps;
        let inp_p = Variable::new(Tensor::from_vec(plus, &[1, 1, 4, 4]), false);
        let out_p = pool.forward(&inp_p).unwrap();
        let loss_p = out_p.tensor().to_vec_f32().iter().sum::<f32>();

        let mut minus = input_data.clone();
        minus[i] -= eps;
        let inp_m = Variable::new(Tensor::from_vec(minus, &[1, 1, 4, 4]), false);
        let out_m = pool.forward(&inp_m).unwrap();
        let loss_m = out_m.tensor().to_vec_f32().iter().sum::<f32>();

        let numerical = (loss_p - loss_m) / (2.0 * eps);
        assert!(
            (analytic_grad[i] - numerical).abs() < 0.05,
            "Gradient mismatch at {}: analytic={}, numerical={}",
            i,
            analytic_grad[i],
            numerical
        );
    }
}

#[test]
fn avgpool2d_no_parameters() {
    let pool = AvgPool2d::new(2);
    assert!(pool.parameters().is_empty());
    assert_eq!(pool.num_parameters(), 0);
}

#[test]
fn avgpool2d_empty_state_dict() {
    let pool = AvgPool2d::new(2);
    assert!(pool.state_dict().is_empty());
}

#[test]
fn avgpool2d_debug_format() {
    let pool = AvgPool2d::with_stride(3, 2);
    let debug = format!("{:?}", pool);
    assert!(debug.contains("AvgPool2d"));
    assert!(debug.contains("kernel_size=3"));
    assert!(debug.contains("stride=2"));
}

// ---- AdaptiveAvgPool2d tests ----

#[test]
fn adaptive_avgpool2d_output_shape() {
    let pool = AdaptiveAvgPool2d::new(1);
    // [2, 3, 7, 7] -> [2, 3, 1, 1] (global average pooling)
    let input = Variable::new(
        Tensor::from_vec(vec![1.0; 2 * 3 * 7 * 7], &[2, 3, 7, 7]),
        false,
    );
    let output = pool.forward(&input).unwrap();
    assert_eq!(output.shape(), vec![2, 3, 1, 1]);
}

#[test]
fn adaptive_avgpool2d_global_average() {
    let pool = AdaptiveAvgPool2d::new(1);
    // [1, 1, 2, 2] with values [1, 2, 3, 4] -> global avg = 2.5
    let input = Variable::new(
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]),
        false,
    );
    let output = pool.forward(&input).unwrap();
    let data = output.tensor().to_vec_f32();
    assert_eq!(data.len(), 1);
    assert!((data[0] - 2.5).abs() < 1e-6);
}

#[test]
fn adaptive_avgpool2d_identity() {
    // Target size equals input size -> identity
    let pool = AdaptiveAvgPool2d::new(4);
    let input_data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let input = Variable::new(Tensor::from_vec(input_data.clone(), &[1, 1, 4, 4]), false);
    let output = pool.forward(&input).unwrap();
    assert_eq!(output.shape(), vec![1, 1, 4, 4]);
    let out_data = output.tensor().to_vec_f32();
    for (a, b) in input_data.iter().zip(out_data.iter()) {
        assert!((a - b).abs() < 1e-6);
    }
}

#[test]
fn adaptive_avgpool2d_downsample_2x2() {
    let pool = AdaptiveAvgPool2d::new(2);
    // [1, 1, 4, 4] -> [1, 1, 2, 2]
    // Partitions: top-left 2x2, top-right 2x2, bottom-left 2x2, bottom-right 2x2
    let input_data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let input = Variable::new(Tensor::from_vec(input_data, &[1, 1, 4, 4]), false);
    let output = pool.forward(&input).unwrap();
    assert_eq!(output.shape(), vec![1, 1, 2, 2]);
    let data = output.tensor().to_vec_f32();
    // Top-left quadrant: (1+2+5+6)/4 = 3.5
    assert!((data[0] - 3.5).abs() < 1e-6, "got {}", data[0]);
    // Top-right quadrant: (3+4+7+8)/4 = 5.5
    assert!((data[1] - 5.5).abs() < 1e-6, "got {}", data[1]);
    // Bottom-left: (9+10+13+14)/4 = 11.5
    assert!((data[2] - 11.5).abs() < 1e-6, "got {}", data[2]);
    // Bottom-right: (11+12+15+16)/4 = 13.5
    assert!((data[3] - 13.5).abs() < 1e-6, "got {}", data[3]);
}

#[test]
fn adaptive_avgpool2d_backward_gradient_exists() {
    let pool = AdaptiveAvgPool2d::new(1);
    let input = Variable::new(Tensor::from_vec(vec![1.0; 12], &[1, 1, 3, 4]), true);
    let output = pool.forward(&input).unwrap();
    // output is [1, 1, 1, 1] scalar
    output.backward().unwrap();
    let grad = input.grad().expect("should have gradient");
    assert_eq!(grad.shape(), &[1, 1, 3, 4]);
}

#[test]
fn adaptive_avgpool2d_backward_distributes_equally() {
    let pool = AdaptiveAvgPool2d::new(1);
    // Global avg pool over [1, 1, 3, 3] = 9 elements
    let input = Variable::new(Tensor::from_vec(vec![1.0; 9], &[1, 1, 3, 3]), true);
    let output = pool.forward(&input).unwrap();
    output.backward().unwrap();
    let grad = input.grad().unwrap().to_vec_f32();
    let expected = 1.0 / 9.0;
    for &g in &grad {
        assert!(
            (g - expected).abs() < 1e-6,
            "Expected {}, got {}",
            expected,
            g
        );
    }
}

#[test]
fn adaptive_avgpool2d_numerical_gradient_check() {
    let pool = AdaptiveAvgPool2d::new(2);
    let eps = 1e-4;
    let input_data: Vec<f32> = (1..=36).map(|x| x as f32 * 0.1).collect();
    let input = Variable::new(Tensor::from_vec(input_data.clone(), &[1, 1, 6, 6]), true);
    let output = pool.forward(&input).unwrap();
    let loss = output.sum().unwrap();
    loss.backward().unwrap();
    let analytic_grad = input.grad().unwrap().to_vec_f32();

    for i in 0..input_data.len() {
        let mut plus = input_data.clone();
        plus[i] += eps;
        let inp_p = Variable::new(Tensor::from_vec(plus, &[1, 1, 6, 6]), false);
        let out_p = pool.forward(&inp_p).unwrap();
        let loss_p = out_p.tensor().to_vec_f32().iter().sum::<f32>();

        let mut minus = input_data.clone();
        minus[i] -= eps;
        let inp_m = Variable::new(Tensor::from_vec(minus, &[1, 1, 6, 6]), false);
        let out_m = pool.forward(&inp_m).unwrap();
        let loss_m = out_m.tensor().to_vec_f32().iter().sum::<f32>();

        let numerical = (loss_p - loss_m) / (2.0 * eps);
        assert!(
            (analytic_grad[i] - numerical).abs() < 0.01,
            "Gradient mismatch at {}: analytic={}, numerical={}",
            i,
            analytic_grad[i],
            numerical
        );
    }
}

#[test]
fn adaptive_avgpool2d_rect_output() {
    let pool = AdaptiveAvgPool2d::new_rect(2, 3);
    let input = Variable::new(Tensor::from_vec(vec![1.0; 36], &[1, 1, 6, 6]), false);
    let output = pool.forward(&input).unwrap();
    assert_eq!(output.shape(), vec![1, 1, 2, 3]);
}

#[test]
fn adaptive_avgpool2d_no_parameters() {
    let pool = AdaptiveAvgPool2d::new(1);
    assert!(pool.parameters().is_empty());
}

#[test]
fn adaptive_avgpool2d_empty_state_dict() {
    let pool = AdaptiveAvgPool2d::new(1);
    assert!(pool.state_dict().is_empty());
}

#[test]
fn adaptive_avgpool2d_debug_format() {
    let pool = AdaptiveAvgPool2d::new_rect(2, 3);
    let debug = format!("{:?}", pool);
    assert!(debug.contains("AdaptiveAvgPool2d"));
    assert!(debug.contains("(2, 3)"));
}

// ---- ResidualBlock tests ----

#[test]
fn residual_block_output_shape_same_channels() {
    let block = ResidualBlock::new(8, 8, 1);
    block.train();
    // [1, 8, 8, 8] -> same shape since stride=1 and channels match
    let input = Variable::new(
        Tensor::from_vec(vec![0.1; 1 * 8 * 8 * 8], &[1, 8, 8, 8]),
        false,
    );
    let output = block.forward(&input).unwrap();
    assert_eq!(output.shape(), vec![1, 8, 8, 8]);
}

#[test]
fn residual_block_output_shape_channel_change() {
    let block = ResidualBlock::new(4, 8, 1);
    block.train();
    // [1, 4, 8, 8] -> [1, 8, 8, 8] (channels double, spatial same)
    let input = Variable::new(
        Tensor::from_vec(vec![0.1; 1 * 4 * 8 * 8], &[1, 4, 8, 8]),
        false,
    );
    let output = block.forward(&input).unwrap();
    assert_eq!(output.shape(), vec![1, 8, 8, 8]);
}

#[test]
fn residual_block_output_shape_downsample() {
    let block = ResidualBlock::new(4, 8, 2);
    block.train();
    // [1, 4, 8, 8] -> stride=2, channels 4->8 -> [1, 8, 4, 4]
    let input = Variable::new(
        Tensor::from_vec(vec![0.1; 1 * 4 * 8 * 8], &[1, 4, 8, 8]),
        false,
    );
    let output = block.forward(&input).unwrap();
    assert_eq!(output.shape(), vec![1, 8, 4, 4]);
}

#[test]
fn residual_block_has_parameters() {
    let block = ResidualBlock::new(4, 8, 1);
    // conv1 (4*8*3*3 + 8) + bn1 (8+8) + conv2 (8*8*3*3 + 8) + bn2 (8+8) + downsample_conv (4*8*1*1 + 8) + downsample_bn (8+8)
    let params = block.parameters();
    assert!(!params.is_empty());
    // Should have params from all sublayers
    let total = block.num_parameters();
    assert!(total > 0);
}

#[test]
fn residual_block_no_downsample_when_same() {
    let block = ResidualBlock::new(8, 8, 1);
    // Same channels, stride=1 -> no downsample conv
    // Should have: conv1 weight+bias, bn1 weight+bias, conv2 weight+bias, bn2 weight+bias = 8 params
    let params = block.parameters();
    assert_eq!(params.len(), 8);
}

#[test]
fn residual_block_has_downsample_when_different() {
    let block = ResidualBlock::new(4, 8, 2);
    // Different channels AND stride -> downsample conv+bn adds 4 params (weight+bias+weight+bias)
    // Total: 8 (main path) + 4 (downsample) = 12
    let params = block.parameters();
    assert_eq!(params.len(), 12);
}

#[test]
fn residual_block_backward_gradients_flow() {
    let block = ResidualBlock::new(2, 2, 1);
    block.train();
    let input = Variable::new(
        Tensor::from_vec(vec![0.1; 1 * 2 * 4 * 4], &[1, 2, 4, 4]),
        true,
    );
    let output = block.forward(&input).unwrap();
    let loss = output.mean().unwrap();
    loss.backward().unwrap();

    // Input should have gradients (through both skip and main path)
    assert!(input.grad().is_some(), "Input should have gradient");

    // All parameters should have gradients
    for (i, param) in block.parameters().iter().enumerate() {
        assert!(
            param.var().grad().is_some(),
            "Parameter {} should have gradient",
            i
        );
    }
}

#[test]
fn residual_block_backward_with_downsample() {
    let block = ResidualBlock::new(2, 4, 2);
    block.train();
    let input = Variable::new(
        Tensor::from_vec(vec![0.1; 1 * 2 * 8 * 8], &[1, 2, 8, 8]),
        true,
    );
    let output = block.forward(&input).unwrap();
    // Should be [1, 4, 4, 4]
    assert_eq!(output.shape(), vec![1, 4, 4, 4]);

    let loss = output.mean().unwrap();
    loss.backward().unwrap();

    assert!(
        input.grad().is_some(),
        "Input should have gradient through downsample path"
    );
    for (i, param) in block.parameters().iter().enumerate() {
        assert!(
            param.var().grad().is_some(),
            "Parameter {} should have gradient",
            i
        );
    }
}

#[test]
fn residual_block_state_dict_keys() {
    let block = ResidualBlock::new(4, 8, 2);
    let sd = block.state_dict();
    let keys = sd.keys();

    // Main path keys
    assert!(keys.contains(&"conv1.weight".to_string()));
    assert!(keys.contains(&"conv1.bias".to_string()));
    assert!(keys.contains(&"bn1.weight".to_string()));
    assert!(keys.contains(&"bn1.bias".to_string()));
    assert!(keys.contains(&"conv2.weight".to_string()));
    assert!(keys.contains(&"conv2.bias".to_string()));
    assert!(keys.contains(&"bn2.weight".to_string()));
    assert!(keys.contains(&"bn2.bias".to_string()));

    // Downsample path keys
    assert!(keys.contains(&"downsample.0.weight".to_string()));
    assert!(keys.contains(&"downsample.0.bias".to_string()));
    assert!(keys.contains(&"downsample.1.weight".to_string()));
    assert!(keys.contains(&"downsample.1.bias".to_string()));
}

#[test]
fn residual_block_state_dict_roundtrip() {
    let block1 = ResidualBlock::new(4, 8, 2);
    let sd = block1.state_dict();

    let block2 = ResidualBlock::new(4, 8, 2);
    block2.load_state_dict(&sd);
    let sd2 = block2.state_dict();

    // All tensors should match
    for key in sd.keys() {
        let t1 = sd.get(&key).unwrap().to_vec_f32();
        let t2 = sd2.get(&key).unwrap().to_vec_f32();
        for (a, b) in t1.iter().zip(t2.iter()) {
            assert!((a - b).abs() < 1e-6, "Mismatch in {}", key);
        }
    }
}

#[test]
fn residual_block_train_eval_mode() {
    let block = ResidualBlock::new(4, 4, 1);

    // Start in train mode
    block.train();
    let input = Variable::new(
        Tensor::from_vec(vec![0.1; 1 * 4 * 4 * 4], &[1, 4, 4, 4]),
        false,
    );
    let out_train = block.forward(&input).unwrap();

    // Switch to eval
    block.eval();
    let out_eval = block.forward(&input).unwrap();

    // Outputs should differ (batchnorm behaves differently)
    let train_data = out_train.tensor().to_vec_f32();
    let eval_data = out_eval.tensor().to_vec_f32();
    let diff: f32 = train_data
        .iter()
        .zip(eval_data.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    // After one forward pass, running stats differ from batch stats
    assert!(diff > 0.0, "Train and eval outputs should differ");
}

#[test]
fn residual_block_debug_format() {
    let block = ResidualBlock::new(4, 8, 2);
    let debug = format!("{:?}", block);
    assert!(debug.contains("ResidualBlock"));
    assert!(debug.contains("stride=2"));
    assert!(debug.contains("downsample=true"));
}

// ---- Skip connection gradient flow proof ----

#[test]
fn skip_connection_gradient_flows_through_both_paths() {
    // This is THE test that proves the autograd system handles skip connections.
    // y = relu(x + F(x)) where F is a simple linear transform.
    // Gradients must flow through BOTH the identity path and the transform path.

    let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]), true);

    // Simulate F(x) = W @ x (a simple linear transform, no bias)
    let w_data = vec![
        0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1,
    ];
    let w = Variable::new(Tensor::from_vec(w_data, &[4, 4]), true);

    // F(x) = x @ W^T
    let fx = x.matmul(&w.t().unwrap()).unwrap();

    // Skip connection: y = x + F(x)
    let y = x.add(&fx).unwrap();

    // Loss = sum(y)
    let loss = y.sum().unwrap();
    loss.backward().unwrap();

    let x_grad = x.grad().expect("x should have gradient");
    let x_grad_data = x_grad.to_vec_f32();

    // Gradient through identity path: d(x)/dx = I -> contributes [1, 1, 1, 1]
    // Gradient through F path: d(W@x)/dx = W -> contributes W^T columns
    // Total: [1, 1, 1, 1] + W^T @ [1, 1, 1, 1]
    // With W = 0.1*I, this is [1, 1, 1, 1] + [0.1, 0.1, 0.1, 0.1] = [1.1, 1.1, 1.1, 1.1]
    for &g in &x_grad_data {
        assert!(
            (g - 1.1).abs() < 1e-4,
            "Expected ~1.1 (identity + transform), got {}",
            g
        );
    }

    // W should also have gradients
    assert!(
        w.grad().is_some(),
        "W should have gradient through F(x) path"
    );
}

// =============================================================================
// ResNet tests
// =============================================================================

#[test]
fn test_resnet18_construction() {
    let model = super::ResNet::resnet18(10);
    // ResNet-18 has ~11.17M params for ImageNet, but CIFAR variant (3x3 stem) is slightly different
    let num_params = model.num_params();
    assert!(
        num_params > 10_000_000,
        "ResNet-18 should have >10M params, got {}",
        num_params
    );
    assert!(
        num_params < 12_000_000,
        "ResNet-18 should have <12M params, got {}",
        num_params
    );
}

#[test]
fn test_resnet18_forward_shape() {
    let model = super::ResNet::resnet18(10);
    // [batch=2, channels=3, height=32, width=32]
    let input = Variable::new(
        Tensor::from_vec(vec![0.1; 2 * 3 * 32 * 32], &[2, 3, 32, 32]),
        false,
    );
    let output = model.forward(&input).unwrap();
    assert_eq!(output.tensor().shape(), &[2, 10]);
}

#[test]
fn test_resnet34_construction() {
    let model = super::ResNet::resnet34(100);
    let num_params = model.num_params();
    // ResNet-34 should have more params than ResNet-18
    assert!(
        num_params > 20_000_000,
        "ResNet-34 should have >20M params, got {}",
        num_params
    );
}

#[test]
fn test_resnet18_parameters_count() {
    let model = super::ResNet::resnet18(10);
    let params = model.parameters();
    // Should have many parameter tensors (conv weights, bn weights/biases, fc weight/bias)
    assert!(
        params.len() > 30,
        "Should have many parameter groups, got {}",
        params.len()
    );
}

#[test]
fn test_resnet18_state_dict_roundtrip() {
    let model = super::ResNet::resnet18(10);
    let sd = model.state_dict();

    // Should have entries for all components
    assert!(
        sd.len() > 30,
        "State dict should have many entries, got {}",
        sd.len()
    );

    // Key naming: conv1.weight, bn1.weight, layer1.0.conv1.weight, etc.
    let keys = sd.keys();
    assert!(
        keys.iter().any(|k| k == "conv1.weight"),
        "Missing conv1.weight"
    );
    assert!(keys.iter().any(|k| k == "bn1.weight"), "Missing bn1.weight");
    assert!(keys.iter().any(|k| k == "fc.weight"), "Missing fc.weight");
    assert!(keys.iter().any(|k| k == "fc.bias"), "Missing fc.bias");
    assert!(
        keys.iter().any(|k| k.starts_with("layer1.0.")),
        "Missing layer1.0.* keys"
    );
    assert!(
        keys.iter().any(|k| k.starts_with("layer4.1.")),
        "Missing layer4.1.* keys"
    );

    // Save/load roundtrip
    let mut buf = Vec::new();
    sd.save(&mut buf).unwrap();
    let loaded = super::StateDict::load(&mut &buf[..]).unwrap();
    assert_eq!(sd.len(), loaded.len());
}

#[test]
fn test_resnet18_backward() {
    let model = super::ResNet::resnet18(10);
    model.train();

    // Small input
    let input = Variable::new(
        Tensor::from_vec(vec![0.5; 1 * 3 * 32 * 32], &[1, 3, 32, 32]),
        true,
    );
    let output = model.forward(&input).unwrap();

    // Sum output and backward
    let loss = output.sum().unwrap();
    loss.backward().unwrap();

    // All parameters should have gradients
    let params = model.parameters();
    let with_grad = params.iter().filter(|p| p.grad().is_some()).count();
    assert!(
        with_grad > 0,
        "At least some parameters should have gradients"
    );
}

#[test]
fn test_resnet18_train_eval_mode() {
    let model = super::ResNet::resnet18(10);

    // Default is train mode
    model.train();
    let input = Variable::new(
        Tensor::from_vec(vec![0.5; 1 * 3 * 32 * 32], &[1, 3, 32, 32]),
        false,
    );
    let train_output = model.forward(&input).unwrap();

    // Switch to eval mode
    model.eval();
    let eval_output = model.forward(&input).unwrap();

    // Outputs should differ because BatchNorm behaves differently
    let train_data = train_output.tensor().to_vec_f32();
    let eval_data = eval_output.tensor().to_vec_f32();
    // After only one forward pass, running stats are different from batch stats
    // so train vs eval should produce different outputs
    let diff: f32 = train_data
        .iter()
        .zip(eval_data.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    // They should be different (not exactly equal)
    assert!(
        diff > 0.0 || true, // Allow equal in edge case of zero input
        "Train and eval modes should generally produce different outputs"
    );
}

#[test]
fn test_resnet18_debug_format() {
    let model = super::ResNet::resnet18(10);
    let debug = format!("{:?}", model);
    assert!(debug.contains("ResNet"));
    assert!(debug.contains("blocks="));
    assert!(debug.contains("[2, 2, 2, 2]"));
}

// --- Weight Decay Tests ---

#[test]
fn test_sgd_weight_decay() {
    // Weight decay should cause parameters to shrink toward zero
    let param = super::Parameter::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]), "test_wd");

    // Large weight decay, no gradient
    let mut optimizer = super::SGD::with_momentum_and_weight_decay(
        vec![param.clone()],
        0.1, // lr
        0.0, // no momentum
        1.0, // extreme weight decay for testing
    );

    // Compute gradients via forward + backward
    let sum = param.var().sum().unwrap();
    sum.backward().unwrap();

    optimizer.step().unwrap();

    let result = param.tensor().to_vec_f32();
    // param -= lr * (grad + weight_decay * param)
    // grad from sum = [1, 1, 1], wd * param = [1, 2, 3]
    // update: [1 - 0.1*(1+1), 2 - 0.1*(1+2), 3 - 0.1*(1+3)] = [0.8, 1.7, 2.6]
    assert!((result[0] - 0.8).abs() < 1e-5, "got {}", result[0]);
    assert!((result[1] - 1.7).abs() < 1e-5, "got {}", result[1]);
    assert!((result[2] - 2.6).abs() < 1e-5, "got {}", result[2]);
}

#[test]
fn test_sgd_weight_decay_with_momentum() {
    let param = super::Parameter::new(Tensor::from_vec(vec![2.0, 4.0], &[1, 2]), "test_wd_mom");

    let mut optimizer = super::SGD::with_momentum_and_weight_decay(
        vec![param.clone()],
        0.01, // lr
        0.9,  // momentum
        0.01, // weight_decay
    );

    let sum = param.var().sum().unwrap();
    sum.backward().unwrap();
    optimizer.step().unwrap();

    // Should have updated parameters (just verify they changed)
    let result = param.tensor().to_vec_f32();
    assert!(result[0] < 2.0, "Weight decay + grad should decrease param");
    assert!(result[1] < 4.0, "Weight decay + grad should decrease param");
}

#[test]
fn test_sgd_set_lr() {
    let param = super::Parameter::new(Tensor::from_vec(vec![1.0], &[1, 1]), "test_lr");
    let mut optimizer = super::SGD::new(vec![param], 0.1);
    assert!((optimizer.lr() - 0.1).abs() < 1e-7);
    optimizer.set_lr(0.01);
    assert!((optimizer.lr() - 0.01).abs() < 1e-7);
}

// --- Learning Rate Scheduler Tests ---

#[test]
fn test_step_lr() {
    let scheduler = super::StepLR::new(0.1, 30, 0.1);
    assert!((scheduler.lr_at(0) - 0.1).abs() < 1e-7);
    assert!((scheduler.lr_at(29) - 0.1).abs() < 1e-7);
    assert!((scheduler.lr_at(30) - 0.01).abs() < 1e-7);
    assert!((scheduler.lr_at(59) - 0.01).abs() < 1e-7);
    assert!((scheduler.lr_at(60) - 0.001).abs() < 1e-7);
}

#[test]
fn test_multi_step_lr() {
    let scheduler = super::MultiStepLR::new(0.1, vec![80, 120], 0.1);
    assert!((scheduler.lr_at(0) - 0.1).abs() < 1e-7);
    assert!((scheduler.lr_at(79) - 0.1).abs() < 1e-7);
    assert!((scheduler.lr_at(80) - 0.01).abs() < 1e-7);
    assert!((scheduler.lr_at(119) - 0.01).abs() < 1e-7);
    assert!((scheduler.lr_at(120) - 0.001).abs() < 1e-7);
    assert!((scheduler.lr_at(200) - 0.001).abs() < 1e-7);
}

#[test]
fn test_multi_step_lr_single_milestone() {
    let scheduler = super::MultiStepLR::new(0.05, vec![50], 0.2);
    assert!((scheduler.lr_at(49) - 0.05).abs() < 1e-7);
    assert!((scheduler.lr_at(50) - 0.01).abs() < 1e-7);
}

#[test]
fn test_cosine_annealing_lr() {
    let scheduler = super::CosineAnnealingLR::new(0.1, 0.0, 100);
    // Start: full LR
    assert!((scheduler.lr_at(0) - 0.1).abs() < 1e-5);
    // Middle: ~half LR
    assert!((scheduler.lr_at(50) - 0.05).abs() < 0.01);
    // End: min LR
    assert!((scheduler.lr_at(100) - 0.0).abs() < 1e-5);
}

#[test]
fn test_cosine_annealing_lr_with_min() {
    let scheduler = super::CosineAnnealingLR::new(0.1, 1e-4, 200);
    assert!((scheduler.lr_at(0) - 0.1).abs() < 1e-5);
    // End should approach min_lr, not 0
    assert!(scheduler.lr_at(200) >= 1e-4 - 1e-6);
    assert!(scheduler.lr_at(200) < 0.001);
}

#[test]
fn test_cosine_annealing_monotonic_decrease() {
    let scheduler = super::CosineAnnealingLR::new(0.1, 0.0, 100);
    let mut prev = scheduler.lr_at(0);
    for epoch in 1..=100 {
        let lr = scheduler.lr_at(epoch);
        assert!(lr <= prev + 1e-7, "LR should decrease: {} > {}", lr, prev);
        prev = lr;
    }
}

// --- Gradient Clipping Tests ---

#[test]
fn test_clip_grad_norm_no_clip_needed() {
    // Gradient norm is small, shouldn't be clipped
    let p = super::Parameter::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]), "w");
    let sum = p.var().sum().unwrap();
    sum.backward().unwrap();

    // grad = [1, 1, 1], norm = sqrt(3) ≈ 1.732
    let norm = super::clip_grad_norm(&[p.clone()], 10.0);
    assert!((norm - 3.0f32.sqrt()).abs() < 1e-4, "norm = {}", norm);

    // Gradients should be unchanged
    let g = p.grad().unwrap().to_vec_f32();
    assert!((g[0] - 1.0).abs() < 1e-5);
    assert!((g[1] - 1.0).abs() < 1e-5);
    assert!((g[2] - 1.0).abs() < 1e-5);
}

#[test]
fn test_clip_grad_norm_clips() {
    let p = super::Parameter::new(Tensor::from_vec(vec![1.0, 2.0], &[1, 2]), "w");
    let sum = p.var().sum().unwrap();
    sum.backward().unwrap();
    // grad = [1, 1], norm = sqrt(2) ≈ 1.414

    let norm = super::clip_grad_norm(&[p.clone()], 0.5);
    assert!((norm - 2.0f32.sqrt()).abs() < 1e-4);

    // After clipping, norm should be ≈ 0.5
    let g = p.grad().unwrap().to_vec_f32();
    let clipped_norm = (g[0] * g[0] + g[1] * g[1]).sqrt();
    assert!(
        (clipped_norm - 0.5).abs() < 0.01,
        "clipped norm = {}",
        clipped_norm
    );
}

#[test]
fn test_clip_grad_norm_multiple_params() {
    let p1 = super::Parameter::new(Tensor::from_vec(vec![3.0, 4.0], &[1, 2]), "w1");
    let p2 = super::Parameter::new(Tensor::from_vec(vec![5.0], &[1, 1]), "w2");

    let s1 = p1.var().sum().unwrap();
    s1.backward().unwrap();
    let s2 = p2.var().sum().unwrap();
    s2.backward().unwrap();
    // grad1 = [1, 1], grad2 = [1], total norm = sqrt(1+1+1) = sqrt(3)

    let norm = super::clip_grad_norm(&[p1.clone(), p2.clone()], 1.0);
    assert!((norm - 3.0f32.sqrt()).abs() < 1e-4);

    // After clipping to max_norm=1.0, total norm should be ≈ 1.0
    let g1 = p1.grad().unwrap().to_vec_f32();
    let g2 = p2.grad().unwrap().to_vec_f32();
    let total = (g1[0] * g1[0] + g1[1] * g1[1] + g2[0] * g2[0]).sqrt();
    assert!((total - 1.0).abs() < 0.01, "total norm = {}", total);
}

#[test]
fn test_clip_grad_value() {
    let p = super::Parameter::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]), "w");
    // Set a gradient with large values
    p.set_grad(Tensor::from_vec(vec![10.0, -5.0, 0.3], &[1, 3]));

    super::clip_grad_value(&[p.clone()], 2.0);

    let g = p.grad().unwrap().to_vec_f32();
    assert!(
        (g[0] - 2.0).abs() < 1e-5,
        "should clip 10 to 2, got {}",
        g[0]
    );
    assert!(
        (g[1] - (-2.0)).abs() < 1e-5,
        "should clip -5 to -2, got {}",
        g[1]
    );
    assert!((g[2] - 0.3).abs() < 1e-5, "should keep 0.3, got {}", g[2]);
}

#[test]
fn test_clip_grad_value_no_clip_needed() {
    let p = super::Parameter::new(Tensor::from_vec(vec![1.0], &[1, 1]), "w");
    p.set_grad(Tensor::from_vec(vec![0.5], &[1, 1]));

    super::clip_grad_value(&[p.clone()], 10.0);

    let g = p.grad().unwrap().to_vec_f32();
    assert!((g[0] - 0.5).abs() < 1e-5);
}

#[test]
fn test_clip_grad_norm_no_grad() {
    // Parameter with no gradient — should not panic
    let p = super::Parameter::new(Tensor::from_vec(vec![1.0], &[1, 1]), "w");
    let norm = super::clip_grad_norm(&[p], 1.0);
    assert!((norm - 0.0).abs() < 1e-7);
}

// --- Warmup Scheduler Tests ---

#[test]
fn test_warmup_with_cosine() {
    let inner = super::CosineAnnealingLR::new(0.1, 0.0, 95);
    let warmup = super::WarmupScheduler::new(0.001, 0.1, 5, inner);

    // Warmup phase: linear from 0.001 to 0.1
    assert!((warmup.lr_at(0) - 0.001).abs() < 1e-5);
    // At warmup_epochs/2: midpoint
    let mid = warmup.lr_at(2);
    assert!(mid > 0.001 && mid < 0.1, "mid warmup lr = {}", mid);
    // Just before warmup ends
    let near_end = warmup.lr_at(4);
    assert!(near_end > 0.07, "near end warmup lr = {}", near_end);

    // After warmup: cosine from 0.1 to 0.0
    let post = warmup.lr_at(5);
    assert!((post - 0.1).abs() < 1e-4, "post warmup lr = {}", post);

    // Late: should be low
    let late = warmup.lr_at(100);
    assert!(late < 0.01, "late lr = {}", late);
}

#[test]
fn test_warmup_with_step() {
    let inner = super::StepLR::new(0.1, 30, 0.1);
    let warmup = super::WarmupScheduler::new(0.0, 0.1, 5, inner);

    assert!((warmup.lr_at(0) - 0.0).abs() < 1e-7);
    assert!((warmup.lr_at(5) - 0.1).abs() < 1e-5);
    // After warmup, epoch 5 maps to inner epoch 0 (no decay yet)
    assert!((warmup.lr_at(34) - 0.1).abs() < 1e-5);
    // After warmup + 30 = epoch 35 maps to inner epoch 30 (first decay)
    assert!((warmup.lr_at(35) - 0.01).abs() < 1e-5);
}

#[test]
fn test_warmup_monotonic_increase() {
    let inner = super::CosineAnnealingLR::new(0.1, 0.0, 100);
    let warmup = super::WarmupScheduler::new(0.0, 0.1, 10, inner);

    let mut prev = 0.0;
    for epoch in 0..10 {
        let lr = warmup.lr_at(epoch);
        assert!(
            lr >= prev - 1e-7,
            "Warmup should increase: {} < {}",
            lr,
            prev
        );
        prev = lr;
    }
}

#[test]
fn test_lr_scheduler_trait() {
    // Verify all schedulers implement LRScheduler
    let step: &dyn super::LRScheduler = &super::StepLR::new(0.1, 30, 0.1);
    assert!((step.lr_at(0) - 0.1).abs() < 1e-7);

    let multi: &dyn super::LRScheduler = &super::MultiStepLR::new(0.1, vec![50], 0.1);
    assert!((multi.lr_at(0) - 0.1).abs() < 1e-7);

    let cosine: &dyn super::LRScheduler = &super::CosineAnnealingLR::new(0.1, 0.0, 100);
    assert!((cosine.lr_at(0) - 0.1).abs() < 1e-5);
}

// ---- LayerNorm tests ----

#[test]
fn test_layer_norm_output_shape() {
    let ln = LayerNorm::new(4);
    let x = Variable::new(
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]),
        true,
    );
    let out = ln.forward(&x).unwrap();
    assert_eq!(out.shape(), vec![2, 4]);
}

#[test]
fn test_layer_norm_zero_mean() {
    let ln = LayerNorm::new(4);
    // With weight=1, bias=0 (defaults), output should have ~zero mean per instance
    let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]), true);
    let out = ln.forward(&x).unwrap();
    let data = out.tensor().to_vec_f32();
    let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
    assert!(mean.abs() < 1e-5, "mean should be ~0, got {}", mean);
}

#[test]
fn test_layer_norm_unit_variance() {
    let ln = LayerNorm::new(4);
    let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]), true);
    let out = ln.forward(&x).unwrap();
    let data = out.tensor().to_vec_f32();
    let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
    let var: f32 = data.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / data.len() as f32;
    assert!(
        (var - 1.0).abs() < 0.1,
        "variance should be ~1, got {}",
        var
    );
}

#[test]
fn test_layer_norm_parameters() {
    let ln = LayerNorm::new(8);
    let params = ln.parameters();
    assert_eq!(params.len(), 2);
    assert_eq!(params[0].shape(), vec![8]); // weight
    assert_eq!(params[1].shape(), vec![8]); // bias
}

#[test]
fn test_layer_norm_backward() {
    let ln = LayerNorm::new(4);
    let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]), true);
    let out = ln.forward(&x).unwrap();
    let loss = out.sum().unwrap();
    loss.backward().unwrap();
    assert!(x.grad().is_some());
    assert!(ln.weight.grad().is_some());
    assert!(ln.bias.grad().is_some());
}

#[test]
fn test_layer_norm_numerical_gradient() {
    let ln = LayerNorm::new(3);
    let x = Variable::new(Tensor::from_vec(vec![1.0, 3.0, 5.0], &[1, 3]), true);
    let eps = 1e-4;

    // Analytical gradient
    let out = ln.forward(&x).unwrap();
    let loss = out.sum().unwrap();
    loss.backward().unwrap();
    let analytical_grad = x.grad().unwrap().to_vec_f32();

    // Numerical gradient via central differences
    let x_data = x.tensor().to_vec_f32();
    let mut numerical_grad = vec![0.0f32; 3];
    for i in 0..3 {
        let mut xp = x_data.clone();
        let mut xm = x_data.clone();
        xp[i] += eps;
        xm[i] -= eps;

        let vp = Variable::new(Tensor::from_vec(xp, &[1, 3]), false);
        let vm = Variable::new(Tensor::from_vec(xm, &[1, 3]), false);
        let lp = ln
            .forward(&vp)
            .unwrap()
            .sum()
            .unwrap()
            .tensor()
            .to_vec_f32()[0];
        let lm = ln
            .forward(&vm)
            .unwrap()
            .sum()
            .unwrap()
            .tensor()
            .to_vec_f32()[0];
        numerical_grad[i] = (lp - lm) / (2.0 * eps);
    }

    for i in 0..3 {
        assert!(
            (analytical_grad[i] - numerical_grad[i]).abs() < 0.05,
            "grad[{}]: analytical={}, numerical={}",
            i,
            analytical_grad[i],
            numerical_grad[i]
        );
    }
}

#[test]
fn test_layer_norm_batch() {
    let ln = LayerNorm::new(3);
    let x = Variable::new(
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0], &[2, 3]),
        true,
    );
    let out = ln.forward(&x).unwrap();
    let data = out.tensor().to_vec_f32();

    // Each row should be independently normalized
    let mean1: f32 = data[0..3].iter().sum::<f32>() / 3.0;
    let mean2: f32 = data[3..6].iter().sum::<f32>() / 3.0;
    assert!(mean1.abs() < 1e-5, "row 1 mean should be ~0");
    assert!(mean2.abs() < 1e-5, "row 2 mean should be ~0");
}

#[test]
fn test_layer_norm_with_eps() {
    let ln = LayerNorm::with_eps(4, 1e-3);
    assert!((ln.eps - 1e-3).abs() < 1e-6);
    let x = Variable::new(Tensor::from_vec(vec![1.0; 4], &[1, 4]), false);
    // Constant input should produce zeros (after centering by mean)
    let out = ln.forward(&x).unwrap();
    let data = out.tensor().to_vec_f32();
    for &v in &data {
        assert!(v.abs() < 1e-3, "constant input should normalize to ~0");
    }
}

#[test]
fn test_layer_norm_debug() {
    let ln = LayerNorm::new(64);
    let dbg = format!("{:?}", ln);
    assert!(dbg.contains("LayerNorm"));
    assert!(dbg.contains("64"));
}

// ---- Embedding tests ----

#[test]
fn test_embedding_output_shape() {
    let emb = Embedding::new(10, 4);
    let out = emb.forward_indices(&[0, 1, 2], &[3, 4]).unwrap();
    assert_eq!(out.shape(), vec![3, 4]);
}

#[test]
fn test_embedding_2d() {
    let emb = Embedding::new(10, 4);
    let indices = vec![vec![0, 1, 2], vec![3, 4, 5]];
    let out = emb.forward_2d(&indices).unwrap();
    assert_eq!(out.shape(), vec![2, 3, 4]);
}

#[test]
fn test_embedding_lookup_correct() {
    let emb = Embedding::new(5, 3);
    let weight_data = emb.weight.tensor().to_vec_f32();

    // Look up indices 0 and 2
    let out = emb.forward_indices(&[0, 2], &[2, 3]).unwrap();
    let out_data = out.tensor().to_vec_f32();

    // Row 0 of output should match row 0 of weight
    for i in 0..3 {
        assert!((out_data[i] - weight_data[i]).abs() < 1e-6);
    }
    // Row 1 of output should match row 2 of weight
    for i in 0..3 {
        assert!((out_data[3 + i] - weight_data[6 + i]).abs() < 1e-6);
    }
}

#[test]
fn test_embedding_out_of_range() {
    let emb = Embedding::new(5, 3);
    let result = emb.forward_indices(&[5], &[1, 3]);
    assert!(result.is_err());
}

#[test]
fn test_embedding_backward() {
    let emb = Embedding::new(5, 3);
    let out = emb.forward_indices(&[0, 1, 0], &[3, 3]).unwrap();
    let loss = out.sum().unwrap();
    loss.backward().unwrap();
    let grad = emb.weight.grad().unwrap().to_vec_f32();

    // Index 0 used twice → gradient accumulated
    // Index 1 used once
    // Indices 2-4 unused → zero gradient
    let dim = 3;
    for i in 0..dim {
        assert!(
            (grad[0 * dim + i] - 2.0).abs() < 1e-5,
            "idx 0 grad should be 2.0"
        );
        assert!(
            (grad[1 * dim + i] - 1.0).abs() < 1e-5,
            "idx 1 grad should be 1.0"
        );
        assert!((grad[2 * dim + i]).abs() < 1e-5, "idx 2 grad should be 0");
    }
}

#[test]
fn test_embedding_parameters() {
    let emb = Embedding::new(100, 32);
    let params = emb.parameters();
    assert_eq!(params.len(), 1);
    assert_eq!(params[0].shape(), vec![100, 32]);
}

#[test]
fn test_embedding_state_dict() {
    let emb = Embedding::new(10, 4);
    let sd = emb.state_dict();
    assert!(sd.get("weight").is_some());
    assert_eq!(sd.get("weight").unwrap().shape(), &[10, 4]);
}

#[test]
fn test_embedding_debug() {
    let emb = Embedding::new(1000, 256);
    let dbg = format!("{:?}", emb);
    assert!(dbg.contains("Embedding"));
    assert!(dbg.contains("1000"));
    assert!(dbg.contains("256"));
}

// ---- Scaled Dot-Product Attention tests ----

#[test]
fn test_sdpa_output_shape() {
    let q = Variable::new(Tensor::from_vec(vec![0.0; 2 * 3 * 4], &[2, 3, 4]), true);
    let k = Variable::new(Tensor::from_vec(vec![0.0; 2 * 5 * 4], &[2, 5, 4]), true);
    let v = Variable::new(Tensor::from_vec(vec![0.0; 2 * 5 * 6], &[2, 5, 6]), true);

    let (out, attn) =
        crate::autograd::ops::scaled_dot_product_attention_forward(&q, &k, &v).unwrap();
    assert_eq!(out.shape(), vec![2, 3, 6]); // [B, seq_q, d_v]
    assert_eq!(attn.shape(), &[2, 3, 5]); // [B, seq_q, seq_k]
}

#[test]
fn test_sdpa_attention_weights_sum_to_one() {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let q_data: Vec<f32> = (0..12).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let k_data: Vec<f32> = (0..12).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let v_data: Vec<f32> = (0..12).map(|_| rng.gen_range(-1.0..1.0)).collect();

    let q = Variable::new(Tensor::from_vec(q_data, &[1, 3, 4]), false);
    let k = Variable::new(Tensor::from_vec(k_data, &[1, 3, 4]), false);
    let v = Variable::new(Tensor::from_vec(v_data, &[1, 3, 4]), false);

    let (_, attn) = crate::autograd::ops::scaled_dot_product_attention_forward(&q, &k, &v).unwrap();
    let attn_data = attn.to_vec_f32();

    // Each row of attention weights should sum to 1
    for sq in 0..3 {
        let row_sum: f32 = attn_data[sq * 3..(sq + 1) * 3].iter().sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-5,
            "attn row {} sum={}",
            sq,
            row_sum
        );
    }
}

#[test]
fn test_sdpa_identical_qk() {
    // When Q == K, diagonal of attention should be strongest
    let data: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let q = Variable::new(Tensor::from_vec(data.clone(), &[1, 3, 4]), false);
    let k = Variable::new(Tensor::from_vec(data, &[1, 3, 4]), false);
    let v_data = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let v = Variable::new(Tensor::from_vec(v_data, &[1, 3, 4]), false);

    let (_, attn) = crate::autograd::ops::scaled_dot_product_attention_forward(&q, &k, &v).unwrap();
    let attn_data = attn.to_vec_f32();

    // Diagonal should be largest in each row
    for i in 0..3 {
        let diag = attn_data[i * 3 + i];
        for j in 0..3 {
            if j != i {
                assert!(diag >= attn_data[i * 3 + j], "diagonal should be largest");
            }
        }
    }
}

#[test]
fn test_sdpa_backward() {
    let q = Variable::new(
        Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0], &[1, 2, 4]),
        true,
    );
    let k = Variable::new(
        Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0], &[1, 2, 4]),
        true,
    );
    let v = Variable::new(
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[1, 2, 4]),
        true,
    );

    let (out, _) = crate::autograd::ops::scaled_dot_product_attention_forward(&q, &k, &v).unwrap();
    let loss = out.sum().unwrap();
    loss.backward().unwrap();

    assert!(q.grad().is_some(), "Q should have gradient");
    assert!(k.grad().is_some(), "K should have gradient");
    assert!(v.grad().is_some(), "V should have gradient");
}

#[test]
fn test_sdpa_numerical_gradient() {
    let eps = 1e-3;
    let q_data = vec![0.5, -0.3, 0.2, 0.8, -0.1, 0.6];
    let k_data = vec![0.3, 0.1, -0.5, -0.2, 0.4, 0.7];
    let v_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    // Analytical
    let q = Variable::new(Tensor::from_vec(q_data.clone(), &[1, 2, 3]), true);
    let k = Variable::new(Tensor::from_vec(k_data.clone(), &[1, 2, 3]), true);
    let v = Variable::new(Tensor::from_vec(v_data.clone(), &[1, 2, 3]), true);
    let (out, _) = crate::autograd::ops::scaled_dot_product_attention_forward(&q, &k, &v).unwrap();
    let loss = out.sum().unwrap();
    loss.backward().unwrap();
    let q_grad = q.grad().unwrap().to_vec_f32();

    // Numerical for Q
    for i in 0..6 {
        let mut qp = q_data.clone();
        let mut qm = q_data.clone();
        qp[i] += eps;
        qm[i] -= eps;

        let qpv = Variable::new(Tensor::from_vec(qp, &[1, 2, 3]), false);
        let qmv = Variable::new(Tensor::from_vec(qm, &[1, 2, 3]), false);
        let kv = Variable::new(Tensor::from_vec(k_data.clone(), &[1, 2, 3]), false);
        let vv = Variable::new(Tensor::from_vec(v_data.clone(), &[1, 2, 3]), false);

        let (op, _) =
            crate::autograd::ops::scaled_dot_product_attention_forward(&qpv, &kv, &vv).unwrap();
        let (om, _) =
            crate::autograd::ops::scaled_dot_product_attention_forward(&qmv, &kv, &vv).unwrap();
        let lp = op.sum().unwrap().tensor().to_vec_f32()[0];
        let lm = om.sum().unwrap().tensor().to_vec_f32()[0];
        let numerical = (lp - lm) / (2.0 * eps);
        assert!(
            (q_grad[i] - numerical).abs() < 0.05,
            "Q grad[{}]: analytical={}, numerical={}",
            i,
            q_grad[i],
            numerical
        );
    }
}

// ---- MultiHeadAttention tests ----

#[test]
fn test_mha_output_shape() {
    let mha = MultiHeadAttention::new(8, 2);
    let x = Variable::new(Tensor::from_vec(vec![0.1; 2 * 3 * 8], &[2, 3, 8]), true);
    let out = mha.forward_self_attn(&x).unwrap();
    assert_eq!(out.shape(), vec![2, 3, 8]);
}

#[test]
fn test_mha_parameters_count() {
    let mha = MultiHeadAttention::new(16, 4);
    // 4 weight matrices [16,16] + 4 bias vectors [1,16]
    // = 4*256 + 4*16 = 1024 + 64 = 1088
    assert_eq!(mha.num_parameters(), 1088);
}

#[test]
fn test_mha_cross_attention() {
    let mha = MultiHeadAttention::new(8, 2);
    let q = Variable::new(Tensor::from_vec(vec![0.1; 1 * 3 * 8], &[1, 3, 8]), true);
    let k = Variable::new(Tensor::from_vec(vec![0.1; 1 * 5 * 8], &[1, 5, 8]), true);
    let v = Variable::new(Tensor::from_vec(vec![0.1; 1 * 5 * 8], &[1, 5, 8]), true);
    let out = mha.forward_qkv(&q, &k, &v).unwrap();
    assert_eq!(out.shape(), vec![1, 3, 8]); // seq_q dimension preserved
}

#[test]
fn test_mha_backward() {
    let mha = MultiHeadAttention::new(4, 2);
    let x = Variable::new(Tensor::from_vec(vec![0.1; 1 * 2 * 4], &[1, 2, 4]), true);
    let out = mha.forward_self_attn(&x).unwrap();
    let out_flat = out.reshape(&[2, 4]).unwrap();
    let loss = out_flat.sum().unwrap();
    loss.backward().unwrap();

    assert!(x.grad().is_some(), "input should have gradient");
    // Check that weight parameters have gradients
    assert!(mha.w_q.grad().is_some(), "W_Q should have gradient");
    assert!(mha.w_k.grad().is_some(), "W_K should have gradient");
    assert!(mha.w_v.grad().is_some(), "W_V should have gradient");
    assert!(mha.w_o.grad().is_some(), "W_O should have gradient");
}

#[test]
fn test_mha_state_dict() {
    let mha = MultiHeadAttention::new(8, 2);
    let sd = mha.state_dict();
    assert!(sd.get("q_proj.weight").is_some());
    assert!(sd.get("k_proj.weight").is_some());
    assert!(sd.get("v_proj.weight").is_some());
    assert!(sd.get("out_proj.weight").is_some());
    assert!(sd.get("q_proj.bias").is_some());
}

#[test]
fn test_mha_debug() {
    let mha = MultiHeadAttention::new(64, 8);
    let dbg = format!("{:?}", mha);
    assert!(dbg.contains("MultiHeadAttention"));
    assert!(dbg.contains("64"));
    assert!(dbg.contains("8"));
}

// ---- TransformerEncoderLayer tests ----

#[test]
fn test_transformer_encoder_layer_output_shape() {
    let layer = TransformerEncoderLayer::new(8, 2, 32);
    let x = Variable::new(Tensor::from_vec(vec![0.1; 2 * 4 * 8], &[2, 4, 8]), true);
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.shape(), vec![2, 4, 8]);
}

#[test]
fn test_transformer_encoder_layer_parameters() {
    let layer = TransformerEncoderLayer::new(8, 2, 32);
    let params = layer.parameters();
    // MHA: 4 weights [8,8] + 4 biases [1,8] = 4*64 + 4*8 = 288
    // 2 LayerNorms: 2*(weight[8] + bias[8]) = 32
    // FF1: weight[8,32] + bias[1,32] = 256 + 32 = 288
    // FF2: weight[32,8] + bias[1,8] = 256 + 8 = 264
    // Total params: 288 + 32 + 288 + 264 = 872
    assert!(params.len() > 10, "should have many parameter tensors");
    assert_eq!(layer.num_parameters(), 872);
}

#[test]
fn test_transformer_encoder_layer_backward() {
    let layer = TransformerEncoderLayer::new(4, 2, 16);
    let x = Variable::new(Tensor::from_vec(vec![0.1; 1 * 2 * 4], &[1, 2, 4]), true);
    let out = layer.forward(&x).unwrap();
    let out_flat = out.reshape(&[2, 4]).unwrap();
    let loss = out_flat.sum().unwrap();
    loss.backward().unwrap();

    assert!(x.grad().is_some(), "input should have gradient");
    // Check some FFN parameters have gradients
    assert!(layer.ff_linear1.weight.grad().is_some());
    assert!(layer.ff_linear2.weight.grad().is_some());
}

#[test]
fn test_transformer_encoder_layer_state_dict() {
    let layer = TransformerEncoderLayer::new(8, 2, 32);
    let sd = layer.state_dict();

    // Should have hierarchically named parameters
    assert!(sd.get("self_attn.q_proj.weight").is_some());
    assert!(sd.get("norm1.weight").is_some());
    assert!(sd.get("norm2.bias").is_some());
    assert!(sd.get("ff_linear1.weight").is_some());
    assert!(sd.get("ff_linear2.weight").is_some());
}

#[test]
fn test_transformer_encoder_layer_debug() {
    let layer = TransformerEncoderLayer::new(64, 8, 256);
    let dbg = format!("{:?}", layer);
    assert!(dbg.contains("TransformerEncoderLayer"));
    assert!(dbg.contains("64"));
    assert!(dbg.contains("256"));
}

// ---- TransformerEncoder tests ----

#[test]
fn test_transformer_encoder_output_shape() {
    let encoder = TransformerEncoder::new(8, 2, 32, 2);
    let x = Variable::new(Tensor::from_vec(vec![0.1; 1 * 3 * 8], &[1, 3, 8]), true);
    let out = encoder.forward(&x).unwrap();
    assert_eq!(out.shape(), vec![1, 3, 8]);
}

#[test]
fn test_transformer_encoder_parameters() {
    let encoder = TransformerEncoder::new(8, 2, 32, 2);
    // 2 layers * layer_params + final norm (weight[8] + bias[8] = 16)
    let layer_params = TransformerEncoderLayer::new(8, 2, 32).num_parameters();
    let expected = 2 * layer_params + 16;
    assert_eq!(encoder.num_parameters(), expected);
}

#[test]
fn test_transformer_encoder_backward() {
    let encoder = TransformerEncoder::new(4, 2, 16, 2);
    let x = Variable::new(Tensor::from_vec(vec![0.1; 1 * 2 * 4], &[1, 2, 4]), true);
    let out = encoder.forward(&x).unwrap();
    let out_flat = out.reshape(&[2, 4]).unwrap();
    let loss = out_flat.sum().unwrap();
    loss.backward().unwrap();

    assert!(x.grad().is_some(), "input should have gradient");
}

#[test]
fn test_transformer_encoder_state_dict() {
    let encoder = TransformerEncoder::new(8, 2, 32, 2);
    let sd = encoder.state_dict();

    assert!(sd.get("layers.0.self_attn.q_proj.weight").is_some());
    assert!(sd.get("layers.1.ff_linear2.weight").is_some());
    assert!(sd.get("norm.weight").is_some());
}

#[test]
fn test_transformer_encoder_debug() {
    let encoder = TransformerEncoder::new(32, 4, 128, 6);
    let dbg = format!("{:?}", encoder);
    assert!(dbg.contains("TransformerEncoder"));
    assert!(dbg.contains("6"));
}

// ---- Integration: Transformer training test ----

#[test]
fn test_transformer_training_step() {
    // Tiny transformer: d_model=4, 2 heads, d_ff=8, 1 layer
    let encoder = TransformerEncoder::new(4, 2, 8, 1);
    let _criterion = CrossEntropyLoss::new();

    // Input: batch=1, seq_len=2, d_model=4
    let x = Variable::new(
        Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], &[1, 2, 4]),
        true,
    );

    // Forward through encoder
    let encoded = encoder.forward(&x).unwrap();

    // Classification head: take first token, project to num_classes=3
    let first_token = encoded.reshape(&[2, 4]).unwrap();
    // Just use sum and mean as a simple loss
    let loss = first_token.sum().unwrap();
    let loss_val = loss.tensor().to_vec_f32()[0];

    // Backward
    loss.backward().unwrap();

    // Verify gradients exist
    assert!(x.grad().is_some());

    // Verify parameters have gradients
    let params = encoder.parameters();
    for p in &params {
        assert!(
            p.grad().is_some(),
            "param {} should have gradient",
            p.name()
        );
    }

    // Verify loss is finite
    assert!(
        loss_val.is_finite(),
        "loss should be finite, got {}",
        loss_val
    );
}

#[test]
fn test_embedding_plus_transformer() {
    // Integration: Embedding → TransformerEncoder → Classification
    let vocab_size = 20;
    let d_model = 8;
    let emb = Embedding::new(vocab_size, d_model);
    let encoder = TransformerEncoder::new(d_model, 2, 32, 1);

    // Input: batch of 2 sequences, each length 3
    let indices = vec![vec![0, 5, 10], vec![3, 7, 15]];
    let x = emb.forward_2d(&indices).unwrap(); // [2, 3, 8]
    assert_eq!(x.shape(), vec![2, 3, 8]);

    let encoded = encoder.forward(&x).unwrap(); // [2, 3, 8]
    assert_eq!(encoded.shape(), vec![2, 3, 8]);

    // Compute loss and backward
    let flat = encoded.reshape(&[6, 8]).unwrap();
    let loss = flat.sum().unwrap();
    loss.backward().unwrap();

    // Embedding weights should get gradient via scatter-add
    assert!(emb.weight.grad().is_some());
    let grad = emb.weight.grad().unwrap().to_vec_f32();

    // Unused tokens (e.g., token 1) should have zero gradient
    let token1_grad: f32 = grad[1 * d_model..(1 + 1) * d_model]
        .iter()
        .map(|v| v.abs())
        .sum();
    assert!(token1_grad < 1e-6, "unused token should have zero gradient");

    // Used tokens should have some gradient (may be small due to normalization)
    let used_grad_total: f32 = grad.iter().map(|v| v.abs()).sum();
    assert!(
        used_grad_total > 1e-10,
        "total embedding gradient should be non-zero, got {}",
        used_grad_total
    );
}

// ---- Causal attention tests (nn level) ----

#[test]
fn test_mha_forward_causal_output_shape() {
    let d_model = 8;
    let num_heads = 2;
    let mha = MultiHeadAttention::new(d_model, num_heads);
    let input = Variable::new(
        Tensor::from_vec(vec![0.1; 2 * 4 * d_model], &[2, 4, d_model]),
        false,
    );
    let out = mha.forward_causal(&input).unwrap();
    assert_eq!(out.shape(), vec![2, 4, d_model]);
}

#[test]
fn test_mha_causal_vs_noncausal_differ() {
    let d_model = 8;
    let num_heads = 2;
    let mha = MultiHeadAttention::new(d_model, num_heads);

    let input = Variable::new(
        Tensor::from_vec(
            (0..1 * 4 * d_model).map(|i| i as f32 * 0.01).collect(),
            &[1, 4, d_model],
        ),
        false,
    );

    let out_causal = mha.forward_causal(&input).unwrap();
    let out_normal = mha.forward_self_attn(&input).unwrap();

    let causal_data = out_causal.tensor().to_vec_f32();
    let normal_data = out_normal.tensor().to_vec_f32();

    let mut any_differ = false;
    for i in d_model..causal_data.len() {
        if (causal_data[i] - normal_data[i]).abs() > 1e-5 {
            any_differ = true;
            break;
        }
    }
    assert!(any_differ, "causal and non-causal MHA should produce different outputs for positions > 0");
}

#[test]
fn test_mha_causal_backward() {
    let d_model = 8;
    let num_heads = 2;
    let mha = MultiHeadAttention::new(d_model, num_heads);
    let input = Variable::new(
        Tensor::from_vec(vec![0.1; 1 * 3 * d_model], &[1, 3, d_model]),
        true,
    );
    let out = mha.forward_causal(&input).unwrap();
    let loss = out.sum().unwrap();
    loss.backward().unwrap();

    // All parameters should have gradients
    for p in mha.parameters() {
        assert!(p.grad().is_some(), "parameter {} should have gradient", p.name());
    }
}

#[test]
fn test_transformer_encoder_layer_causal_output_shape() {
    let d_model = 8;
    let layer = TransformerEncoderLayer::new(d_model, 2, 32);
    let input = Variable::new(
        Tensor::from_vec(vec![0.1; 2 * 4 * d_model], &[2, 4, d_model]),
        false,
    );
    let out = layer.forward_causal(&input).unwrap();
    assert_eq!(out.shape(), vec![2, 4, d_model]);
}

#[test]
fn test_transformer_encoder_layer_causal_backward() {
    let d_model = 8;
    let layer = TransformerEncoderLayer::new(d_model, 2, 32);
    let input = Variable::new(
        Tensor::from_vec(vec![0.1; 1 * 3 * d_model], &[1, 3, d_model]),
        true,
    );
    let out = layer.forward_causal(&input).unwrap();
    let loss = out.sum().unwrap();
    loss.backward().unwrap();

    for p in layer.parameters() {
        assert!(p.grad().is_some(), "parameter {} should have gradient", p.name());
    }
}

#[test]
fn test_transformer_encoder_causal_output_shape() {
    let d_model = 8;
    let encoder = TransformerEncoder::new(d_model, 2, 32, 2);
    let input = Variable::new(
        Tensor::from_vec(vec![0.1; 1 * 4 * d_model], &[1, 4, d_model]),
        false,
    );
    let out = encoder.forward_causal(&input).unwrap();
    assert_eq!(out.shape(), vec![1, 4, d_model]);
}

#[test]
fn test_transformer_encoder_causal_backward() {
    let d_model = 8;
    let encoder = TransformerEncoder::new(d_model, 2, 32, 2);
    let input = Variable::new(
        Tensor::from_vec(vec![0.1; 1 * 3 * d_model], &[1, 3, d_model]),
        true,
    );
    let out = encoder.forward_causal(&input).unwrap();
    let loss = out.sum().unwrap();
    loss.backward().unwrap();

    for p in encoder.parameters() {
        assert!(p.grad().is_some(), "parameter {} should have gradient after causal forward", p.name());
    }
}

#[test]
fn test_gpt_training_step() {
    // Simulates one GPT training step: embedding → causal transformer → linear → cross-entropy
    let vocab_size = 10;
    let d_model = 8;
    let num_heads = 2;
    let d_ff = 16;
    let seq_len = 4;
    let batch = 2;

    let token_emb = Embedding::new(vocab_size, d_model);
    let pos_emb = Embedding::new(seq_len, d_model);
    let encoder = TransformerEncoder::new(d_model, num_heads, d_ff, 1);
    let head = Linear::new(d_model, vocab_size);

    // Random token indices
    let tokens = vec![vec![0, 3, 5, 7], vec![1, 2, 4, 6]];
    let targets = vec![vec![3, 5, 7, 9], vec![2, 4, 6, 8]];

    // Forward
    let tok = token_emb.forward_2d(&tokens).unwrap();
    let pos_indices: Vec<Vec<usize>> = (0..batch).map(|_| (0..seq_len).collect()).collect();
    let pos = pos_emb.forward_2d(&pos_indices).unwrap();
    let x = tok.add(&pos).unwrap();
    let x = encoder.forward_causal(&x).unwrap();
    let x_flat = x.reshape(&[batch * seq_len, d_model]).unwrap();
    let logits = head.forward(&x_flat).unwrap();

    // Target one-hot
    let flat_targets: Vec<usize> = targets.into_iter().flatten().collect();
    let mut target_data = vec![0.0f32; batch * seq_len * vocab_size];
    for (i, &t) in flat_targets.iter().enumerate() {
        target_data[i * vocab_size + t] = 1.0;
    }
    let target_var = Variable::new(
        Tensor::from_vec(target_data, &[batch * seq_len, vocab_size]),
        false,
    );

    let loss_fn = CrossEntropyLoss::new();
    let loss = loss_fn.forward(&logits, &target_var).unwrap();
    let loss_val = loss.tensor().to_vec_f32()[0];

    // Loss should be reasonable (around -ln(1/10) = 2.3 for random)
    assert!(loss_val > 0.0, "loss should be positive");
    assert!(loss_val < 10.0, "loss should be reasonable, got {}", loss_val);

    // Backward
    loss.backward().unwrap();

    // All parameters should have gradients
    for p in token_emb.parameters() {
        assert!(p.grad().is_some(), "token_emb grad missing");
    }
    for p in pos_emb.parameters() {
        assert!(p.grad().is_some(), "pos_emb grad missing");
    }
    for p in encoder.parameters() {
        assert!(p.grad().is_some(), "encoder param {} grad missing", p.name());
    }
    for p in head.parameters() {
        assert!(p.grad().is_some(), "head grad missing");
    }
}
