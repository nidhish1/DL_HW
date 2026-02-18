"""Activation Function Implementations."""
import numpy as np


class ReLU:
    """
    Rectified Linear Unit activation: y = max(0, x)
    """
    
    def __init__(self):
        self._cache = {}
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: y = max(0, x)
        
        Args:
            x: Input tensor of any shape
            
        Returns:
            Output tensor of same shape as input
        """
        self._cache["mask"] = x > 0
        return np.maximum(0, x)
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass: dL/dx = dL/dy * 1_{x > 0}
        
        Args:
            grad_output: Gradient from upstream, same shape as forward output
            
        Returns:
            Gradient w.r.t. input, same shape as forward input
        """
        mask = self._cache["mask"]
        return grad_output * mask
    
    def get_parameters(self):
        """ReLU has no learnable parameters."""
        return []
    
    def get_gradients(self):
        """ReLU has no gradients for parameters."""
        return []


class Sigmoid:
    """
    Sigmoid activation: y = 1 / (1 + exp(-x))
    """
    
    def __init__(self):
        self._cache = {}
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: y = sigmoid(x) = 1 / (1 + exp(-x))
        
        Numerically stable: clip x to avoid overflow; for x >= 0 use 1/(1+exp(-x)),
        for x < 0 use exp(x)/(1+exp(x)).
        
        Args:
            x: Input tensor of any shape
            
        Returns:
            Output tensor of same shape as input
        """
        x = np.clip(x, -500.0, 500.0)
        y = np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))
        self._cache["y"] = y
        return y
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass: dL/dx = dL/dy * sigmoid(x) * (1 - sigmoid(x))
                             = dL/dy * y * (1 - y)
        
        Args:
            grad_output: Gradient from upstream
            
        Returns:
            Gradient w.r.t. input
        """
        y = self._cache["y"]
        return grad_output * y * (1.0 - y)
    
    def get_parameters(self):
        """Sigmoid has no learnable parameters."""
        return []
    
    def get_gradients(self):
        """Sigmoid has no gradients for parameters."""
        return []
