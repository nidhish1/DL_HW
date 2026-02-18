"""Numerical Gradient Checking Utilities."""
import numpy as np


def gradient_check(model, loss_fn, x: np.ndarray, y: np.ndarray, epsilon: float = 1e-5) -> float:
    """
    Compare analytical gradients with numerical gradients.
    
    For each parameter p, compute:
        numerical_grad[i] = (L(p[i] + eps) - L(p[i] - eps)) / (2 * eps)
    
    Then compute relative error:
        relative_error = |analytical - numerical| / (|analytical| + |numerical|)
    
    Args:
        model: Sequential model with get_parameters() and get_gradients()
        loss_fn: Loss function with forward() and backward()
        x: Input batch, shape (batch_size, input_features)
        y: Target batch (one-hot), shape (batch_size, num_classes)
        epsilon: Small perturbation for numerical gradient
        
    Returns:
        Maximum relative error across all parameters
    """
    # TODO: Implement gradient checking
    # 
    # Steps:
    # 1. Do a forward and backward pass to get analytical gradients
    #    - output = model.forward(x)
    #    - loss = loss_fn.forward(output, y)
    #    - grad = loss_fn.backward()
    #    - model.backward(grad)
    #    - analytical_grads = model.get_gradients()
    #
    # 2. For each parameter array in model.get_parameters():
    #    - For each element in the parameter array:
    #      a. Save original value
    #      b. Add epsilon, compute forward pass and loss (loss_plus)
    #      c. Subtract 2*epsilon, compute forward pass and loss (loss_minus)
    #      d. Restore original value
    #      e. numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)
    #      f. Compare with corresponding analytical gradient
    #      g. Compute relative error for this element
    #
    # 3. Return the maximum relative error
    #
    # Hint: Use np.nditer with 'multi_index' flag to iterate over array elements
    
    raise NotImplementedError("TODO: Implement gradient_check()")


def check_gradient_simple(func, x: np.ndarray, epsilon: float = 1e-5):
    """
    Simple gradient check for a scalar function.
    
    Args:
        func: Function that takes x and returns (loss, grad)
        x: Input array
        epsilon: Perturbation size
        
    Returns:
        Tuple of (analytical_grad, numerical_grad, max_relative_error)
    """
    loss, analytical_grad = func(x)
    
    numerical_grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        original = x[idx]
        
        x[idx] = original + epsilon
        loss_plus, _ = func(x)
        
        x[idx] = original - epsilon
        loss_minus, _ = func(x)
        
        x[idx] = original
        
        numerical_grad[idx] = (loss_plus - loss_minus) / (2 * epsilon)
        it.iternext()
    
    diff = np.abs(analytical_grad - numerical_grad)
    denom = np.abs(analytical_grad) + np.abs(numerical_grad) + 1e-12
    relative_error = np.max(diff / denom)
    
    return analytical_grad, numerical_grad, relative_error
