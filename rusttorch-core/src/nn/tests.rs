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
