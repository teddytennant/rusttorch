//! Finite-difference gradcheck for autograd primitives.
//!
//! For each differentiable op, we compare the analytical gradient computed by
//! `backward()` with a centered finite-difference approximation
//!   f'(x_i) ≈ (f(x + eps*e_i) - f(x - eps*e_i)) / (2 * eps).
//!
//! Running this from `tests/` (an integration test binary) also exercises the
//! crate's public API surface the way downstream users see it.

use rusttorch_core::autograd::Variable;
use rusttorch_core::tensor::Tensor;

const EPS: f32 = 1e-3;
const TOL: f32 = 1e-2;

/// Numerical gradient of a scalar-valued function `f: Vec<f32> -> f32`.
fn numerical_grad<F>(x: &[f32], mut f: F) -> Vec<f32>
where
    F: FnMut(&[f32]) -> f32,
{
    let mut grad = vec![0.0_f32; x.len()];
    let mut xp = x.to_vec();
    let mut xm = x.to_vec();
    for i in 0..x.len() {
        xp.copy_from_slice(x);
        xm.copy_from_slice(x);
        xp[i] += EPS;
        xm[i] -= EPS;
        grad[i] = (f(&xp) - f(&xm)) / (2.0 * EPS);
    }
    grad
}

fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol + tol * b.abs().max(a.abs())
}

fn assert_grads_close(analytical: &[f32], numerical: &[f32], op: &str) {
    assert_eq!(
        analytical.len(),
        numerical.len(),
        "{op}: length mismatch"
    );
    for (i, (&a, &n)) in analytical.iter().zip(numerical.iter()).enumerate() {
        assert!(
            approx_eq(a, n, TOL),
            "{op}: grad[{i}] mismatch — analytical={a}, numerical={n}"
        );
    }
}

// ---- Elementwise ops ----

#[test]
fn gradcheck_add() {
    let x_vals = [1.0_f32, -2.0, 3.0, 0.5];
    let y_vals = [0.5_f32, 1.5, -1.0, 2.0];
    let shape = [4];

    let x = Variable::new(Tensor::from_vec(x_vals.to_vec(), &shape), true);
    let y = Variable::new(Tensor::from_vec(y_vals.to_vec(), &shape), true);
    let loss = x.add(&y).unwrap().sum().unwrap();
    loss.backward().unwrap();

    let gx = x.grad().unwrap().to_vec_f32();
    let num_gx = numerical_grad(&x_vals, |xv| {
        xv.iter().zip(y_vals.iter()).map(|(a, b)| a + b).sum()
    });
    assert_grads_close(&gx, &num_gx, "add");
}

#[test]
fn gradcheck_mul() {
    let x_vals = [1.0_f32, -2.0, 3.0, 0.5];
    let y_vals = [0.5_f32, 1.5, -1.0, 2.0];
    let shape = [4];

    let x = Variable::new(Tensor::from_vec(x_vals.to_vec(), &shape), true);
    let y = Variable::new(Tensor::from_vec(y_vals.to_vec(), &shape), true);
    let loss = x.mul(&y).unwrap().sum().unwrap();
    loss.backward().unwrap();

    let gx = x.grad().unwrap().to_vec_f32();
    let num_gx = numerical_grad(&x_vals, |xv| {
        xv.iter().zip(y_vals.iter()).map(|(a, b)| a * b).sum()
    });
    assert_grads_close(&gx, &num_gx, "mul");
}

#[test]
fn gradcheck_sub() {
    let x_vals = [3.0_f32, -1.0, 2.5];
    let y_vals = [1.0_f32, 1.0, 0.5];
    let shape = [3];

    let x = Variable::new(Tensor::from_vec(x_vals.to_vec(), &shape), true);
    let y = Variable::new(Tensor::from_vec(y_vals.to_vec(), &shape), true);
    let loss = x.sub(&y).unwrap().sum().unwrap();
    loss.backward().unwrap();

    let gx = x.grad().unwrap().to_vec_f32();
    let num_gx = numerical_grad(&x_vals, |xv| {
        xv.iter().zip(y_vals.iter()).map(|(a, b)| a - b).sum()
    });
    assert_grads_close(&gx, &num_gx, "sub");
}

#[test]
fn gradcheck_div() {
    let x_vals = [2.0_f32, 4.0, 6.0];
    let y_vals = [1.0_f32, 2.0, 3.0];
    let shape = [3];

    let x = Variable::new(Tensor::from_vec(x_vals.to_vec(), &shape), true);
    let y = Variable::new(Tensor::from_vec(y_vals.to_vec(), &shape), true);
    let loss = x.div(&y).unwrap().sum().unwrap();
    loss.backward().unwrap();

    let gx = x.grad().unwrap().to_vec_f32();
    let num_gx = numerical_grad(&x_vals, |xv| {
        xv.iter().zip(y_vals.iter()).map(|(a, b)| a / b).sum()
    });
    assert_grads_close(&gx, &num_gx, "div");
}

// ---- Activations ----

#[test]
fn gradcheck_relu() {
    // Avoid testing at exactly zero (non-differentiable kink).
    let x_vals = [-2.0_f32, -0.5, 0.7, 1.3, 3.0];
    let shape = [5];

    let x = Variable::new(Tensor::from_vec(x_vals.to_vec(), &shape), true);
    let loss = x.relu().sum().unwrap();
    loss.backward().unwrap();

    let gx = x.grad().unwrap().to_vec_f32();
    let num_gx = numerical_grad(&x_vals, |xv| xv.iter().map(|v| v.max(0.0)).sum());
    assert_grads_close(&gx, &num_gx, "relu");
}

#[test]
fn gradcheck_sigmoid() {
    let x_vals = [-1.5_f32, -0.3, 0.0, 0.4, 2.0];
    let shape = [5];

    let x = Variable::new(Tensor::from_vec(x_vals.to_vec(), &shape), true);
    let loss = x.sigmoid().unwrap().sum().unwrap();
    loss.backward().unwrap();

    let gx = x.grad().unwrap().to_vec_f32();
    let num_gx = numerical_grad(&x_vals, |xv| {
        xv.iter().map(|v| 1.0 / (1.0 + (-v).exp())).sum()
    });
    assert_grads_close(&gx, &num_gx, "sigmoid");
}

#[test]
fn gradcheck_tanh() {
    let x_vals = [-1.0_f32, -0.25, 0.0, 0.5, 1.5];
    let shape = [5];

    let x = Variable::new(Tensor::from_vec(x_vals.to_vec(), &shape), true);
    let loss = x.tanh_act().unwrap().sum().unwrap();
    loss.backward().unwrap();

    let gx = x.grad().unwrap().to_vec_f32();
    let num_gx = numerical_grad(&x_vals, |xv| xv.iter().map(|v| v.tanh()).sum());
    assert_grads_close(&gx, &num_gx, "tanh");
}

#[test]
fn gradcheck_gelu() {
    // GELU is smooth, so gradcheck is meaningful everywhere.
    let x_vals = [-2.0_f32, -0.5, 0.0, 0.7, 2.0];
    let shape = [5];

    let x = Variable::new(Tensor::from_vec(x_vals.to_vec(), &shape), true);
    let loss = x.gelu().sum().unwrap();
    loss.backward().unwrap();

    let gx = x.grad().unwrap().to_vec_f32();

    // Reference (tanh-based GELU approximation, matching rusttorch's impl).
    let gelu = |v: f32| -> f32 {
        let c = (2.0_f32 / std::f32::consts::PI).sqrt();
        0.5 * v * (1.0 + (c * (v + 0.044715 * v * v * v)).tanh())
    };
    let num_gx = numerical_grad(&x_vals, |xv| xv.iter().map(|&v| gelu(v)).sum());

    // GELU's finite-difference estimate is noisier — relax tolerance slightly.
    for (i, (&a, &n)) in gx.iter().zip(num_gx.iter()).enumerate() {
        assert!(
            (a - n).abs() < 5e-2,
            "gelu: grad[{i}] analytical={a}, numerical={n}"
        );
    }
}

// ---- Reductions ----

#[test]
fn gradcheck_sum() {
    let x_vals = [1.0_f32, 2.0, 3.0, 4.0];
    let shape = [4];

    let x = Variable::new(Tensor::from_vec(x_vals.to_vec(), &shape), true);
    let loss = x.sum().unwrap();
    loss.backward().unwrap();

    let gx = x.grad().unwrap().to_vec_f32();
    // d(sum(x))/dx_i = 1 for all i.
    for (i, &g) in gx.iter().enumerate() {
        assert!(
            (g - 1.0).abs() < 1e-6,
            "sum: grad[{i}] should be 1, got {g}"
        );
    }
}

#[test]
fn gradcheck_mean() {
    let x_vals = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let shape = [5];
    let n = x_vals.len() as f32;

    let x = Variable::new(Tensor::from_vec(x_vals.to_vec(), &shape), true);
    let loss = x.mean().unwrap();
    loss.backward().unwrap();

    let gx = x.grad().unwrap().to_vec_f32();
    for (i, &g) in gx.iter().enumerate() {
        assert!(
            (g - 1.0 / n).abs() < 1e-6,
            "mean: grad[{i}] should be 1/n, got {g}"
        );
    }
}

// ---- Matrix ops ----

#[test]
fn gradcheck_matmul_wrt_x() {
    // x: [2, 3], y: [3, 2]; loss = sum(x @ y)
    let x_vals = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let y_vals = [0.5_f32, -1.0, 2.0, 1.5, -0.5, 1.0];

    let x = Variable::new(Tensor::from_vec(x_vals.to_vec(), &[2, 3]), true);
    let y = Variable::new(Tensor::from_vec(y_vals.to_vec(), &[3, 2]), true);
    let loss = x.matmul(&y).unwrap().sum().unwrap();
    loss.backward().unwrap();

    let gx = x.grad().unwrap().to_vec_f32();

    let num_gx = numerical_grad(&x_vals, |xv| {
        // Compute sum(xv @ y_vals) where xv is [2, 3] and y_vals is [3, 2]
        let mut total = 0.0;
        for i in 0..2 {
            for j in 0..2 {
                let mut acc = 0.0;
                for k in 0..3 {
                    acc += xv[i * 3 + k] * y_vals[k * 2 + j];
                }
                total += acc;
            }
        }
        total
    });
    assert_grads_close(&gx, &num_gx, "matmul_wrt_x");
}

// ---- SiLU / SwiGLU ----

#[test]
fn gradcheck_silu() {
    // silu(x) = x * sigmoid(x). Smooth everywhere.
    let x_vals = [-2.0_f32, -0.5, 0.3, 1.5, 3.0];
    let shape = [5];

    let x = Variable::new(Tensor::from_vec(x_vals.to_vec(), &shape), true);
    let loss = x.silu().unwrap().sum().unwrap();
    loss.backward().unwrap();

    let gx = x.grad().unwrap().to_vec_f32();
    let silu = |v: f32| -> f32 { v * (1.0 / (1.0 + (-v).exp())) };
    let num_gx = numerical_grad(&x_vals, |xv| xv.iter().map(|&v| silu(v)).sum());
    assert_grads_close(&gx, &num_gx, "silu");
}

#[test]
fn gradcheck_swiglu_wrt_gate() {
    // swiglu(gate, value) = silu(gate) * value
    use rusttorch_core::nn::swiglu;

    let gate_vals = [0.5_f32, -0.3, 1.2, -1.5];
    let value_vals = [2.0_f32, -1.0, 0.5, 1.5];
    let shape = [4];

    let gate = Variable::new(Tensor::from_vec(gate_vals.to_vec(), &shape), true);
    let value = Variable::new(Tensor::from_vec(value_vals.to_vec(), &shape), false);
    let loss = swiglu(&gate, &value).unwrap().sum().unwrap();
    loss.backward().unwrap();

    let gg = gate.grad().unwrap().to_vec_f32();
    let silu = |v: f32| -> f32 { v * (1.0 / (1.0 + (-v).exp())) };
    let num_gg = numerical_grad(&gate_vals, |gv| {
        gv.iter()
            .zip(value_vals.iter())
            .map(|(&g, &v)| silu(g) * v)
            .sum()
    });
    assert_grads_close(&gg, &num_gg, "swiglu_gate");
}

#[test]
fn gradcheck_swiglu_wrt_value() {
    use rusttorch_core::nn::swiglu;

    let gate_vals = [0.5_f32, -0.3, 1.2, -1.5];
    let value_vals = [2.0_f32, -1.0, 0.5, 1.5];
    let shape = [4];

    let gate = Variable::new(Tensor::from_vec(gate_vals.to_vec(), &shape), false);
    let value = Variable::new(Tensor::from_vec(value_vals.to_vec(), &shape), true);
    let loss = swiglu(&gate, &value).unwrap().sum().unwrap();
    loss.backward().unwrap();

    let gv = value.grad().unwrap().to_vec_f32();
    // dL/d(value_i) = silu(gate_i)
    let silu = |v: f32| -> f32 { v * (1.0 / (1.0 + (-v).exp())) };
    for (i, (&g, expected)) in gv
        .iter()
        .zip(gate_vals.iter().map(|&g| silu(g)))
        .enumerate()
    {
        assert!(
            (g - expected).abs() < 1e-5,
            "swiglu grad_value[{i}]: got {g}, want {expected}"
        );
    }
}

// ---- RMSNorm ----

#[test]
fn rms_norm_forward_matches_reference() {
    // Reference: y = x * (1/sqrt(mean(x^2) + eps)) * weight (no bias).
    let x_vals: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let w_vals: [f32; 3] = [0.5, 1.0, 2.0];
    let eps = 1e-6_f32;

    let x = Variable::new(Tensor::from_vec(x_vals.to_vec(), &[2, 3]), false);
    let w = Variable::new(Tensor::from_vec(w_vals.to_vec(), &[3]), false);
    let y = x.rms_norm(3, Some(&w), eps).unwrap();
    let got = y.tensor().to_vec_f32();

    // Row 0: mean(1 + 4 + 9)/3 = 14/3, inv_rms = 1/sqrt(14/3 + eps)
    // Row 1: mean(16 + 25 + 36)/3 = 77/3, inv_rms = 1/sqrt(77/3 + eps)
    for row in 0..2 {
        let mean_sq: f32 = (0..3)
            .map(|j| x_vals[row * 3 + j] * x_vals[row * 3 + j])
            .sum::<f32>()
            / 3.0;
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();
        for j in 0..3 {
            let expected = x_vals[row * 3 + j] * inv_rms * w_vals[j];
            assert!(
                (got[row * 3 + j] - expected).abs() < 1e-5,
                "rms_norm mismatch at [{row}, {j}]: got {} want {expected}",
                got[row * 3 + j]
            );
        }
    }
}

#[test]
fn gradcheck_rms_norm_wrt_input() {
    // Small 1-row test so numerical_grad is tractable.
    let x_vals = [0.3_f32, -1.2, 2.5, 0.8];
    let w_vals = [1.0_f32, 0.8, 1.2, 0.9];
    let eps = 1e-5_f32;

    let x = Variable::new(Tensor::from_vec(x_vals.to_vec(), &[1, 4]), true);
    let w = Variable::new(Tensor::from_vec(w_vals.to_vec(), &[4]), false);
    let loss = x.rms_norm(4, Some(&w), eps).unwrap().sum().unwrap();
    loss.backward().unwrap();

    let gx = x.grad().unwrap().to_vec_f32();

    // Reference: sum(rms_norm(x, w))
    let ref_rms_norm_sum = |xv: &[f32]| -> f32 {
        let mean_sq: f32 = xv.iter().map(|v| v * v).sum::<f32>() / xv.len() as f32;
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();
        xv.iter()
            .zip(w_vals.iter())
            .map(|(x, w)| x * inv_rms * w)
            .sum()
    };
    let num_gx = numerical_grad(&x_vals, ref_rms_norm_sum);
    // RMSNorm's Jacobian is sensitive so use a slightly looser tolerance.
    for (i, (&a, &n)) in gx.iter().zip(num_gx.iter()).enumerate() {
        assert!(
            (a - n).abs() < 5e-3,
            "rms_norm grad[{i}]: analytical={a}, numerical={n}"
        );
    }
}

#[test]
fn gradcheck_rms_norm_wrt_weight() {
    let x_vals = [0.5_f32, 1.5, -0.25, 2.0];
    let w_vals = [1.0_f32, 1.0, 1.0, 1.0];
    let eps = 1e-5_f32;

    let x = Variable::new(Tensor::from_vec(x_vals.to_vec(), &[1, 4]), false);
    let w = Variable::new(Tensor::from_vec(w_vals.to_vec(), &[4]), true);
    let loss = x.rms_norm(4, Some(&w), eps).unwrap().sum().unwrap();
    loss.backward().unwrap();

    let gw = w.grad().unwrap().to_vec_f32();

    // dy/dw_j = x_j * inv_rms, so grad_w_j (for loss = sum(y)) = x_j * inv_rms
    let mean_sq: f32 = x_vals.iter().map(|v| v * v).sum::<f32>() / x_vals.len() as f32;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();
    for j in 0..4 {
        let expected = x_vals[j] * inv_rms;
        assert!(
            (gw[j] - expected).abs() < 1e-5,
            "rms_norm grad_w[{j}]: got {} want {expected}",
            gw[j]
        );
    }
}

// ---- Scaled dot-product attention ----

#[test]
fn sdpa_causal_masks_future_positions() {
    use rusttorch_core::nn::scaled_dot_product_attention;

    // Build Q = K = V = identity-ish for a single batch, 3-token sequence,
    // 2 embedding dims. With causal masking, position 0 should only see
    // itself, position 1 sees positions 0 and 1, etc.
    let q = Variable::new(
        Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[1, 3, 2]),
        false,
    );
    let k = q.clone();
    let v = Variable::new(
        Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[1, 3, 2]),
        false,
    );

    let out = scaled_dot_product_attention(&q, &k, &v, true).unwrap();
    let out_data = out.tensor().to_vec_f32();
    assert_eq!(out.shape(), vec![1, 3, 2]);

    // First token can only attend to itself, so its output must equal v[0].
    assert!(
        (out_data[0] - 10.0).abs() < 1e-4,
        "pos 0 attends only to self: got {} want {}",
        out_data[0],
        10.0
    );
    assert!(
        (out_data[1] - 20.0).abs() < 1e-4,
        "pos 0 attends only to self: got {} want {}",
        out_data[1],
        20.0
    );
}

#[test]
fn sdpa_non_causal_sees_all_positions() {
    use rusttorch_core::nn::scaled_dot_product_attention;

    // With uniform Q = zeros, attention weights are uniform over K and
    // the output is the mean of V along the sequence axis.
    let q = Variable::new(Tensor::from_vec(vec![0.0; 6], &[1, 3, 2]), false);
    let k = Variable::new(Tensor::from_vec(vec![0.0; 6], &[1, 3, 2]), false);
    let v = Variable::new(
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 3, 2]),
        false,
    );

    let out = scaled_dot_product_attention(&q, &k, &v, false).unwrap();
    let out_data = out.tensor().to_vec_f32();
    // Each query position's output = (v[0] + v[1] + v[2]) / 3
    //                              = ([1,2] + [3,4] + [5,6]) / 3
    //                              = [3.0, 4.0]
    for pos in 0..3 {
        assert!((out_data[pos * 2] - 3.0).abs() < 1e-4);
        assert!((out_data[pos * 2 + 1] - 4.0).abs() < 1e-4);
    }
}

// ---- GroupNorm ----

#[test]
fn gradcheck_group_norm_wrt_input() {
    // Small [1, 4, 2] input, num_groups=2, no affine → plain per-group normalization.
    let x_vals: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.0, 1.5, -0.5];
    let shape = [1, 4, 2];
    let eps = 1e-5_f32;

    let x = Variable::new(Tensor::from_vec(x_vals.clone(), &shape), true);
    let loss = x.group_norm(2, None, None, eps).unwrap().sum().unwrap();
    loss.backward().unwrap();
    let gx = x.grad().unwrap().to_vec_f32();

    // Reference: sum(group_norm(x)) with 2 groups of 2 channels × 2 spatial.
    let ref_sum = |xv: &[f32]| -> f32 {
        let mut out = vec![0.0f32; xv.len()];
        let c = 4;
        let l = 2;
        let num_groups = 2;
        let k = c / num_groups;
        let m = (k * l) as f32;
        for g in 0..num_groups {
            // Collect M elements
            let mut buf = Vec::with_capacity(k * l);
            for kk in 0..k {
                let cc = g * k + kk;
                for s in 0..l {
                    buf.push(xv[cc * l + s]);
                }
            }
            let mean = buf.iter().sum::<f32>() / m;
            let var = buf.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / m;
            let inv_s = 1.0 / (var + eps).sqrt();
            for kk in 0..k {
                let cc = g * k + kk;
                for s in 0..l {
                    out[cc * l + s] = (xv[cc * l + s] - mean) * inv_s;
                }
            }
        }
        out.iter().sum()
    };
    let num_gx = numerical_grad(&x_vals, ref_sum);
    // GroupNorm backward is numerically sensitive; use a slightly relaxed tol.
    for (i, (&a, &n)) in gx.iter().zip(num_gx.iter()).enumerate() {
        assert!(
            (a - n).abs() < 5e-3,
            "group_norm grad[{i}]: analytical={a}, numerical={n}"
        );
    }
}

// ---- Composite ----

#[test]
fn gradcheck_mse_like_loss() {
    // loss = sum((x - t) * (x - t))
    let x_vals = [0.5_f32, -0.8, 1.2, 2.3];
    let t_vals = [0.0_f32, 0.0, 1.0, 2.0];
    let shape = [4];

    let x = Variable::new(Tensor::from_vec(x_vals.to_vec(), &shape), true);
    let t = Variable::new(Tensor::from_vec(t_vals.to_vec(), &shape), false);
    let diff = x.sub(&t).unwrap();
    let sq = diff.mul(&diff).unwrap();
    let loss = sq.sum().unwrap();
    loss.backward().unwrap();

    let gx = x.grad().unwrap().to_vec_f32();
    let num_gx = numerical_grad(&x_vals, |xv| {
        xv.iter()
            .zip(t_vals.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum()
    });
    assert_grads_close(&gx, &num_gx, "mse_like_loss");
}
