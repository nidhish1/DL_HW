"""Optimizer Implementations."""
import numpy as np


class SGD:
    """
    Stochastic Gradient Descent optimizer.
    
    Update rule: param = param - lr * grad
    """
    
    def __init__(self, parameters: list, lr: float = 0.01):
        """
        Args:
            parameters: List of parameter arrays to optimize
            lr: Learning rate
        """
        self.parameters = parameters
        self.lr = lr
    
    def step(self, gradients: list):
        """
        Update parameters using gradients.
        
        Args:
            gradients: List of gradient arrays (same order as parameters)
        """
        # TODO: Implement parameter update
        # For each (param, grad) pair: param -= lr * grad
        # Note: Use in-place update (param -= ...) to modify the original arrays
        
        raise NotImplementedError("TODO: Implement SGD.step()")
    
    def zero_grad(self):
        """
        Zero out stored gradients (if applicable).
        For this simple implementation, gradients are computed fresh each backward pass.
        """
        pass
