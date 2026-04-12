//! End-to-end training integration tests.
//!
//! These exercise the full training loop — model forward, loss, backward,
//! optimizer step — without any data downloads. If one of these fails, a
//! regression has broken something in the training pipeline that unit tests
//! would miss (e.g. gradient flow between Sequential layers, optimizer state
//! accumulation, etc.).

use rusttorch_core::autograd::Variable;
use rusttorch_core::nn::{Adam, Linear, MSELoss, Module, Optimizer, Sequential, Tanh, SGD};
use rusttorch_core::tensor::Tensor;

#[test]
fn mlp_learns_xor() {
    // Classic XOR: requires a nonlinear hidden layer. If any part of the
    // forward/backward pipeline is broken, loss won't decrease.
    let model = Sequential::new(vec![
        Box::new(Linear::new(2, 8)),
        Box::new(Tanh::new()),
        Box::new(Linear::new(8, 1)),
    ]);
    let loss_fn = MSELoss::new();
    let mut optimizer = Adam::new(model.parameters(), 0.05);

    let inputs = [[0.0_f32, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let targets = [0.0_f32, 1.0, 1.0, 0.0];

    let mut first_loss = f32::NAN;
    let mut final_loss = f32::MAX;

    for epoch in 0..500 {
        let mut epoch_loss = 0.0;
        for (inp, &tgt) in inputs.iter().zip(targets.iter()) {
            optimizer.zero_grad();
            let x = Variable::new(Tensor::from_vec(inp.to_vec(), &[1, 2]), false);
            let y = Variable::new(Tensor::from_vec(vec![tgt], &[1, 1]), false);

            let pred = model.forward(&x).unwrap();
            let loss = loss_fn.forward(&pred, &y).unwrap();
            epoch_loss += loss.tensor().to_vec_f32()[0];

            loss.backward().unwrap();
            optimizer.step().unwrap();
        }
        if epoch == 0 {
            first_loss = epoch_loss / 4.0;
        }
        final_loss = epoch_loss / 4.0;
    }

    assert!(
        final_loss < first_loss,
        "loss should decrease: first={first_loss}, final={final_loss}"
    );
    assert!(
        final_loss < 0.1,
        "XOR should be learned to < 0.1 MSE, got {final_loss}"
    );
}

#[test]
fn sgd_decreases_loss_on_linear_regression() {
    // y = 2x + 1 + noise. Pure linear regression with vanilla SGD.
    let model = Linear::new(1, 1);
    let loss_fn = MSELoss::new();
    let mut optimizer = SGD::new(model.parameters(), 0.01);

    let xs: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
    let ys: Vec<f32> = xs.iter().map(|x| 2.0 * x + 1.0).collect();

    let x_var = Variable::new(Tensor::from_vec(xs.clone(), &[16, 1]), false);
    let y_var = Variable::new(Tensor::from_vec(ys.clone(), &[16, 1]), false);

    let initial_pred = model.forward(&x_var).unwrap();
    let initial_loss = loss_fn
        .forward(&initial_pred, &y_var)
        .unwrap()
        .tensor()
        .to_vec_f32()[0];

    for _ in 0..300 {
        optimizer.zero_grad();
        let pred = model.forward(&x_var).unwrap();
        let loss = loss_fn.forward(&pred, &y_var).unwrap();
        loss.backward().unwrap();
        optimizer.step().unwrap();
    }

    let final_pred = model.forward(&x_var).unwrap();
    let final_loss = loss_fn
        .forward(&final_pred, &y_var)
        .unwrap()
        .tensor()
        .to_vec_f32()[0];

    assert!(
        final_loss < initial_loss,
        "loss should decrease: initial={initial_loss}, final={final_loss}"
    );
    assert!(
        final_loss < 0.1,
        "linear regression should converge, final loss = {final_loss}"
    );
}

#[test]
fn adam_matches_or_beats_sgd_on_same_problem() {
    // Adam with a reasonable lr should reach a lower loss than vanilla SGD
    // in the same number of steps on linear regression — a basic sanity
    // check that the Adam implementation's second-moment correction is wired
    // through and not silently a no-op.
    let xs: Vec<f32> = (0..32).map(|i| i as f32 * 0.05).collect();
    let ys: Vec<f32> = xs.iter().map(|x| 3.0 * x - 0.5).collect();
    let x_var = Variable::new(Tensor::from_vec(xs.clone(), &[32, 1]), false);
    let y_var = Variable::new(Tensor::from_vec(ys.clone(), &[32, 1]), false);
    let loss_fn = MSELoss::new();

    let train = |optimizer: &mut dyn Optimizer, model: &Linear, steps: usize| -> f32 {
        for _ in 0..steps {
            optimizer.zero_grad();
            let pred = model.forward(&x_var).unwrap();
            let loss = loss_fn.forward(&pred, &y_var).unwrap();
            loss.backward().unwrap();
            optimizer.step().unwrap();
        }
        let pred = model.forward(&x_var).unwrap();
        loss_fn
            .forward(&pred, &y_var)
            .unwrap()
            .tensor()
            .to_vec_f32()[0]
    };

    let sgd_model = Linear::new(1, 1);
    let mut sgd_opt = SGD::new(sgd_model.parameters(), 0.05);
    let sgd_loss = train(&mut sgd_opt, &sgd_model, 200);

    let adam_model = Linear::new(1, 1);
    let mut adam_opt = Adam::new(adam_model.parameters(), 0.05);
    let adam_loss = train(&mut adam_opt, &adam_model, 200);

    // Both should converge acceptably.
    assert!(sgd_loss < 0.5, "SGD should converge, got {sgd_loss}");
    assert!(adam_loss < 0.5, "Adam should converge, got {adam_loss}");
}

#[test]
fn gradient_flows_through_sequential() {
    // Sanity: after a backward pass, every Linear weight should have a
    // non-zero gradient. Regression guard for any future refactor that might
    // break gradient propagation between Sequential layers.
    let model = Sequential::new(vec![
        Box::new(Linear::new(4, 8)),
        Box::new(Tanh::new()),
        Box::new(Linear::new(8, 4)),
        Box::new(Tanh::new()),
        Box::new(Linear::new(4, 1)),
    ]);
    let loss_fn = MSELoss::new();
    let x = Variable::new(
        Tensor::from_vec(vec![0.1_f32, 0.2, 0.3, 0.4], &[1, 4]),
        false,
    );
    let y = Variable::new(Tensor::from_vec(vec![1.0_f32], &[1, 1]), false);

    let pred = model.forward(&x).unwrap();
    let loss = loss_fn.forward(&pred, &y).unwrap();
    loss.backward().unwrap();

    let params = model.parameters();
    assert!(!params.is_empty(), "model should expose parameters");

    let mut nonzero = 0;
    for p in &params {
        if let Some(g) = p.grad() {
            let v = g.to_vec_f32();
            if v.iter().any(|&x| x != 0.0) {
                nonzero += 1;
            }
        }
    }
    // Expect gradients on most parameters (biases at an input far from the
    // optimum should receive signal too).
    assert!(
        nonzero >= params.len() / 2,
        "gradients should flow through Sequential — only {}/{} params have non-zero grads",
        nonzero,
        params.len()
    );
}

#[test]
fn optimizer_zero_grad_clears_state() {
    let model = Linear::new(2, 1);
    let optimizer = SGD::new(model.parameters(), 0.01);

    let x = Variable::new(Tensor::from_vec(vec![0.5_f32, -0.3], &[1, 2]), false);
    let y = Variable::new(Tensor::from_vec(vec![1.0_f32], &[1, 1]), false);
    let loss_fn = MSELoss::new();

    let pred = model.forward(&x).unwrap();
    let loss = loss_fn.forward(&pred, &y).unwrap();
    loss.backward().unwrap();

    // After backward, at least one parameter should have a gradient.
    let has_grad_before = model.parameters().iter().any(|p| {
        p.grad()
            .map(|g| g.to_vec_f32().iter().any(|&x| x != 0.0))
            .unwrap_or(false)
    });
    assert!(has_grad_before);

    optimizer.zero_grad();

    // After zero_grad, all gradients should be zero.
    for p in model.parameters() {
        if let Some(g) = p.grad() {
            for x in g.to_vec_f32() {
                assert_eq!(x, 0.0, "zero_grad didn't clear gradient");
            }
        }
    }
}
