"""Verify Q6.3 SoftmaxCrossEntropy: forward (scalar loss, batch average) and backward (dL/dz = (y_hat - y)/N)."""
import sys
import numpy as np

sys.path.insert(0, ".")
from mytorch.nn.modules.loss import SoftmaxCrossEntropy


def test_forward_backward_shape():
    """Loss is scalar; backward returns (batch_size, num_classes)."""
    loss_fn = SoftmaxCrossEntropy()
    N, C = 4, 3
    logits = np.random.randn(N, C).astype(np.float64)
    targets = np.zeros((N, C))
    targets[np.arange(N), np.array([0, 1, 2, 0])] = 1.0

    L = loss_fn.forward(logits, targets)
    assert np.isscalar(L) or L.shape == (), f"Loss should be scalar, got shape {np.shape(L)}"
    grad = loss_fn.backward()
    assert grad.shape == (N, C), f"grad shape should be ({N}, {C}), got {grad.shape}"
    print("SoftmaxCrossEntropy forward (scalar) and backward shape: OK")


def test_known_gradient():
    """When y_hat = y (perfect prediction), gradient (y_hat - y)/N should be zero."""
    loss_fn = SoftmaxCrossEntropy()
    # Make logits such that softmax is [1,0,0] for first sample
    logits = np.array([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0]], dtype=np.float64)
    targets = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)

    L = loss_fn.forward(logits, targets)
    grad = loss_fn.backward()
    assert np.allclose(grad, 0.0), f"At perfect prediction grad should be ~0, got max |grad| = {np.max(np.abs(grad))}"
    print("SoftmaxCrossEntropy known gradient (perfect pred): OK")


def test_numerical_gradient():
    """Numerical gradient check for dL/dz."""
    loss_fn = SoftmaxCrossEntropy()
    N, C = 2, 3
    np.random.seed(42)
    logits = np.random.randn(N, C).astype(np.float64) * 0.5
    targets = np.zeros((N, C))
    targets[0, 1] = 1.0
    targets[1, 0] = 1.0

    def loss_at(logits_flat):
        z = logits_flat.reshape(N, C)
        return loss_fn.forward(z, targets)

    L0 = loss_at(logits)
    grad_analytical = loss_fn.backward()
    eps = 1e-6
    grad_numerical = np.zeros_like(logits)
    for i in range(N):
        for j in range(C):
            logits_plus = logits.copy()
            logits_plus[i, j] += eps
            L_plus = loss_at(logits_plus)
            logits_minus = logits.copy()
            logits_minus[i, j] -= eps
            L_minus = loss_at(logits_minus)
            grad_numerical[i, j] = (L_plus - L_minus) / (2 * eps)

    diff = np.abs(grad_analytical - grad_numerical)
    denom = np.abs(grad_analytical) + np.abs(grad_numerical) + 1e-12
    rel_err = np.max(diff / denom)
    print(f"SoftmaxCrossEntropy numerical grad check: max relative error = {rel_err:.2e}")
    assert rel_err < 1e-5, f"Gradient check failed: relative error {rel_err}"
    print("SoftmaxCrossEntropy numerical gradient: PASS")


if __name__ == "__main__":
    print("--- Q6.3 SoftmaxCrossEntropy verification ---\n")
    test_forward_backward_shape()
    test_known_gradient()
    test_numerical_gradient()
    print("\nAll checks passed.")
