"""End-to-end smoke test: train an XOR MLP from Python.

If this passes, it exercises the complete Python -> Rust autograd ->
optimizer pipeline: PyVariable, PyLinear.forward, PyMSELoss, backward,
PyAdam.step. A regression in any one of those surfaces here.

Run with:
    maturin develop  (from rusttorch-py/)
    python -m pytest python/tests/test_autograd_smoke.py -q
or just:
    python python/tests/test_autograd_smoke.py
"""
from __future__ import annotations

import numpy as np

import rusttorch
from rusttorch import Adam, Linear, MSELoss, SGD, Variable


def test_variable_numpy_roundtrip() -> None:
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    v = Variable.from_numpy(x, requires_grad=False)
    assert v.shape() == [2, 3]
    assert v.numel() == 6
    out = v.to_numpy()
    assert out.shape == (2, 3)
    assert np.allclose(out, x)


def test_backward_simple() -> None:
    x = Variable([1.0, 2.0, 3.0], [3], requires_grad=True)
    y = Variable([0.5, 1.5, -1.0], [3], requires_grad=True)
    loss = x.mul(y).sum()
    loss.backward()
    # d(sum(x*y))/dx = y
    grad = x.grad()
    assert grad is not None
    assert np.allclose(grad, [0.5, 1.5, -1.0], atol=1e-6)


def test_linear_forward_shape() -> None:
    lin = Linear(3, 2)
    assert lin.in_features() == 3
    assert lin.out_features() == 2
    params = lin.parameters()
    assert len(params) == 2  # weight + bias
    x = Variable([0.1, 0.2, 0.3], [1, 3], requires_grad=False)
    y = lin.forward(x)
    assert y.shape() == [1, 2]


def test_xor_learns() -> None:
    """Train a 2-8-1 MLP on XOR from pure Python.

    Uses Adam with lr=0.05 and trains for 500 epochs. Regression guard
    for the entire autograd + optimizer + module pipeline through the
    Python bindings.
    """
    fc1 = Linear(2, 8)
    fc2 = Linear(8, 1)
    params = fc1.parameters() + fc2.parameters()
    opt = Adam(params, 0.05)
    loss_fn = MSELoss()

    inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    targets = [0.0, 1.0, 1.0, 0.0]

    first_loss = None
    final_loss = None
    for epoch in range(500):
        epoch_loss = 0.0
        for x_row, t in zip(inputs, targets):
            opt.zero_grad()
            x = Variable(x_row, [1, 2], requires_grad=False)
            y = Variable([t], [1, 1], requires_grad=False)
            h = fc1.forward(x).tanh()
            pred = fc2.forward(h)
            loss = loss_fn.forward(pred, y)
            epoch_loss += loss.to_list()[0]
            loss.backward()
            opt.step()
        if epoch == 0:
            first_loss = epoch_loss / 4.0
        final_loss = epoch_loss / 4.0

    assert first_loss is not None and final_loss is not None
    print(f"first_loss={first_loss:.4f}  final_loss={final_loss:.4f}")
    assert final_loss < first_loss, "loss should decrease"
    assert final_loss < 0.1, f"XOR should converge below 0.1 MSE, got {final_loss}"


def test_sgd_linear_regression() -> None:
    """y = 2x + 1, pure-Python SGD."""
    lin = Linear(1, 1)
    opt = SGD(lin.parameters(), 0.05)
    loss_fn = MSELoss()

    xs = np.arange(16, dtype=np.float32).reshape(-1, 1) * 0.1
    ys = 2.0 * xs + 1.0

    x_var = Variable.from_numpy(xs.astype(np.float32))
    y_var = Variable.from_numpy(ys.astype(np.float32))

    initial = loss_fn.forward(lin.forward(x_var), y_var).to_list()[0]
    for _ in range(300):
        opt.zero_grad()
        pred = lin.forward(x_var)
        loss = loss_fn.forward(pred, y_var)
        loss.backward()
        opt.step()
    final = loss_fn.forward(lin.forward(x_var), y_var).to_list()[0]
    print(f"regression initial={initial:.4f}  final={final:.4f}")
    assert final < initial
    assert final < 0.1


if __name__ == "__main__":
    test_variable_numpy_roundtrip()
    print("OK test_variable_numpy_roundtrip")
    test_backward_simple()
    print("OK test_backward_simple")
    test_linear_forward_shape()
    print("OK test_linear_forward_shape")
    test_xor_learns()
    print("OK test_xor_learns")
    test_sgd_linear_regression()
    print("OK test_sgd_linear_regression")
    print(f"\nrusttorch version: {rusttorch.__version__}")
