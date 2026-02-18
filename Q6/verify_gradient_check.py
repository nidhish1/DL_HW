"""Verify Q6.6 gradient_check: max relative error < 1e-5."""
import sys
import numpy as np

sys.path.insert(0, ".")
from mytorch.nn.sequential import Sequential
from mytorch.nn.modules.linear import Linear
from mytorch.nn.modules.activation import ReLU
from mytorch.nn.modules.loss import SoftmaxCrossEntropy
from mytorch.utils.gradient_check import gradient_check


def test_gradient_check_small_batch():
    """Run gradient_check on a small model and batch; expect max relative error < 1e-5."""
    np.random.seed(42)
    model = Sequential(
        Linear(8, 6),
        ReLU(),
        Linear(6, 4),
    )
    loss_fn = SoftmaxCrossEntropy()
    N, C = 5, 4
    x = np.random.randn(N, 8).astype(np.float64) * 0.1
    y = np.zeros((N, C))
    y[np.arange(N), np.random.randint(0, C, N)] = 1.0

    max_err = gradient_check(model, loss_fn, x, y, epsilon=1e-5)
    print(f"Gradient check: max relative error = {max_err:.2e}")
    assert max_err < 1e-5, f"Expected max relative error < 1e-5, got {max_err}"
    print("Q6.6 gradient_check: PASS (error < 1e-5)")


if __name__ == "__main__":
    print("--- Q6.6 Gradient check verification ---\n")
    test_gradient_check_small_batch()
    print("\nAll checks passed.")
