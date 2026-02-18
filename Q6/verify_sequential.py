"""Verify Q6.4 Sequential: forward order, backward reverse order, get_parameters/get_gradients."""
import sys
import numpy as np

sys.path.insert(0, ".")
from mytorch.nn.sequential import Sequential
from mytorch.nn.modules.linear import Linear
from mytorch.nn.modules.activation import ReLU
from mytorch.nn.modules.loss import SoftmaxCrossEntropy


def test_forward_shape():
    """Sequential forward: input passes through modules in order; output shape is correct."""
    model = Sequential(
        Linear(4, 3),
        ReLU(),
        Linear(3, 2),
    )
    x = np.random.randn(5, 4).astype(np.float64)
    out = model.forward(x)
    assert out.shape == (5, 2), f"Expected (5, 2), got {out.shape}"
    print("Sequential forward (shape): OK")


def test_backward_and_gradients():
    """Full forward -> loss -> backward(loss) -> model.backward; params and grads match."""
    model = Sequential(
        Linear(4, 3),
        ReLU(),
        Linear(3, 2),
    )
    loss_fn = SoftmaxCrossEntropy()
    N, C = 5, 2
    x = np.random.randn(N, 4).astype(np.float64) * 0.1
    targets = np.zeros((N, C))
    targets[np.arange(N), np.random.randint(0, C, N)] = 1.0

    out = model.forward(x)
    L = loss_fn.forward(out, targets)
    grad_out = loss_fn.backward()
    grad_in = model.backward(grad_out)

    assert grad_in.shape == x.shape, f"grad_in shape {grad_in.shape} != x shape {x.shape}"
    params = model.get_parameters()
    grads = model.get_gradients()
    assert len(params) == len(grads), f"#params {len(params)} != #grads {len(grads)}"
    for p, g in zip(params, grads):
        assert p.shape == g.shape, f"param shape {p.shape} != grad shape {g.shape}"
    print("Sequential backward + get_parameters/get_gradients: OK")


def test_full_forward_backward_value():
    """Sanity: two Linear layers only, known weights -> known output and gradient."""
    np.random.seed(42)
    model = Sequential(Linear(2, 2), Linear(2, 2))
    # Overwrite with simple weights for predictable result
    model.modules[0].W[:] = np.array([[1.0, 0.0], [0.0, 1.0]])
    model.modules[0].b[:] = 0.0
    model.modules[1].W[:] = np.array([[1.0, 0.0], [0.0, 1.0]])
    model.modules[1].b[:] = 0.0

    x = np.array([[1.0, 2.0]], dtype=np.float64)
    out = model.forward(x)
    assert np.allclose(out, [[1.0, 2.0]]), f"out = {out}"

    grad_out = np.array([[1.0, 1.0]], dtype=np.float64)
    grad_in = model.backward(grad_out)
    assert grad_in.shape == (1, 2)
    # dL/dW2 = grad_out.T @ h1, dL/dh1 = grad_out @ W2.T
    assert np.allclose(model.modules[1].dW, [[1.0, 2.0], [1.0, 2.0]])
    # dL/dW1 = (grad to h1).T @ x = [[1],[1]] @ [[1,2]] = [[1,2],[1,2]]
    assert np.allclose(model.modules[0].dW, [[1.0, 2.0], [1.0, 2.0]])
    print("Sequential full forward/backward (two Linear): OK")


if __name__ == "__main__":
    print("--- Q6.4 Sequential verification ---\n")
    test_forward_shape()
    test_backward_and_gradients()
    test_full_forward_backward_value()
    print("\nAll checks passed.")
