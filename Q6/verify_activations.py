"""Verify Q6.2 ReLU and Sigmoid forward/backward and gradients via finite differences."""
import sys
import numpy as np

# Run from Q6 so mytorch is on path
sys.path.insert(0, ".")
from mytorch.nn.modules.activation import ReLU, Sigmoid
from mytorch.utils.gradient_check import check_gradient_simple


def test_relu_forward_backward():
    """Sanity check: shapes and known values."""
    relu = ReLU()
    x = np.array([[-1.0, 2.0], [0.0, -3.0]])
    y = relu.forward(x)
    assert np.allclose(y, [[0, 2], [0, 0]]), f"ReLU forward: got {y}"
    grad_out = np.ones_like(y)
    grad_in = relu.backward(grad_out)
    assert np.allclose(grad_in, [[0, 1], [0, 0]]), f"ReLU backward: got {grad_in}"
    print("ReLU forward/backward (shape + values): OK")


def test_sigmoid_forward_backward():
    """Sanity check: shapes and known value at 0."""
    sig = Sigmoid()
    x = np.array([[0.0, 1.0], [-1.0, 0.5]])
    y = sig.forward(x)
    assert np.allclose(y[0, 0], 0.5), f"sigmoid(0) should be 0.5, got {y[0,0]}"
    grad_out = np.ones_like(y)
    grad_in = sig.backward(grad_out)
    # d/dx sigmoid(x) = sigmoid(x)*(1-sigmoid(x))
    assert grad_in.shape == x.shape
    print("Sigmoid forward/backward (shape + sigmoid(0)=0.5): OK")


def test_relu_gradient():
    """Numerical gradient check for ReLU: L = sum(relu(x)), dL/dx = 1_{x>0}."""
    relu = ReLU()
    np.random.seed(42)
    x = np.random.randn(4, 5).astype(np.float64)

    def func(x_in):
        y = relu.forward(x_in)
        loss = np.sum(y)
        grad = relu.backward(np.ones_like(y))
        return loss, grad

    anal, num, err = check_gradient_simple(func, x, epsilon=1e-6)
    print(f"ReLU gradient check: max relative error = {err:.2e}")
    assert err < 1e-5, f"ReLU gradient check failed: relative error {err}"
    print("ReLU gradient check: PASS")


def test_sigmoid_gradient():
    """Numerical gradient check for Sigmoid: L = sum(sigmoid(x))."""
    sig = Sigmoid()
    np.random.seed(42)
    x = np.random.randn(4, 5).astype(np.float64) * 0.5

    def func(x_in):
        y = sig.forward(x_in)
        loss = np.sum(y)
        grad = sig.backward(np.ones_like(y))
        return loss, grad

    anal, num, err = check_gradient_simple(func, x, epsilon=1e-6)
    print(f"Sigmoid gradient check: max relative error = {err:.2e}")
    assert err < 1e-5, f"Sigmoid gradient check failed: relative error {err}"
    print("Sigmoid gradient check: PASS")


if __name__ == "__main__":
    print("--- Q6.2 Activation verification ---\n")
    test_relu_forward_backward()
    test_sigmoid_forward_backward()
    test_relu_gradient()
    test_sigmoid_gradient()
    print("\nAll checks passed.")
