"""Verify Q6.5 SGD: step() updates parameters in-place (param -= lr * grad)."""
import sys
import numpy as np

sys.path.insert(0, ".")
from mytorch.nn.sequential import Sequential
from mytorch.nn.modules.linear import Linear
from mytorch.nn.modules.activation import ReLU
from mytorch.nn.modules.loss import SoftmaxCrossEntropy
from mytorch.nn.optim import SGD


def test_sgd_step_updates_params():
    """After step(gradients), parameters should decrease by lr * grad (for positive grad)."""
    model = Sequential(Linear(3, 2), ReLU(), Linear(2, 2))
    params_before = [p.copy() for p in model.get_parameters()]

    # One forward/backward to get gradients
    x = np.random.randn(4, 3).astype(np.float64) * 0.1
    targets = np.zeros((4, 2))
    targets[:, 0] = 1.0
    out = model.forward(x)
    loss_fn = SoftmaxCrossEntropy()
    L_before = loss_fn.forward(out, targets)
    grad_out = loss_fn.backward()
    model.backward(grad_out)
    grads = model.get_gradients()

    lr = 0.1
    opt = SGD(model.get_parameters(), lr=lr)
    opt.step(grads)

    params_after = model.get_parameters()
    for i, (p_bef, p_aft, g) in enumerate(zip(params_before, params_after, grads)):
        expected = p_bef - lr * g
        assert np.allclose(p_aft, expected), f"param {i}: step() should do param -= lr*grad"
    print("SGD step (param -= lr * grad): OK")


def test_sgd_reduces_loss():
    """A few steps should decrease loss (sanity check that training direction is correct)."""
    np.random.seed(42)
    model = Sequential(Linear(4, 3), ReLU(), Linear(3, 2))
    loss_fn = SoftmaxCrossEntropy()
    opt = SGD(model.get_parameters(), lr=0.5)

    x = np.random.randn(8, 4).astype(np.float64) * 0.3
    targets = np.zeros((8, 2))
    targets[np.arange(8), np.random.randint(0, 2, 8)] = 1.0

    losses = []
    for _ in range(5):
        out = model.forward(x)
        L = loss_fn.forward(out, targets)
        losses.append(L)
        grad_out = loss_fn.backward()
        model.backward(grad_out)
        opt.step(model.get_gradients())

    assert losses[-1] < losses[0], f"Loss should decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    print(f"SGD reduces loss over steps: {losses[0]:.4f} -> {losses[-1]:.4f}  OK")


if __name__ == "__main__":
    print("--- Q6.5 SGD verification ---\n")
    test_sgd_step_updates_params()
    test_sgd_reduces_loss()
    print("\nAll checks passed.")
