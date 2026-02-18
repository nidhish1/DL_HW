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
        # TODO: Implement ReLU forward pass
        # 1. Cache x (or the mask x > 0) for backward
        # 2. Compute y = max(0, x)
        # 3. Return y
        
        raise NotImplementedError("TODO: Implement ReLU.forward()")
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass: dL/dx = dL/dy * 1_{x > 0}
        
        Args:
            grad_output: Gradient from upstream, same shape as forward output
            
        Returns:
            Gradient w.r.t. input, same shape as forward input
        """
        # TODO: Implement ReLU backward pass
        # 1. Retrieve cached values
        # 2. Compute gradient: grad_output * (x > 0)
        # 3. Return gradient
        
        raise NotImplementedError("TODO: Implement ReLU.backward()")
    
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
        
        Use numerically stable implementation!
        
        Args:
            x: Input tensor of any shape
            
        Returns:
            Output tensor of same shape as input
        """
        # TODO: Implement Sigmoid forward pass
        # 1. Clip x to avoid overflow (e.g., np.clip(x, -500, 500))
        # 2. Use numerically stable sigmoid:
        #    - For x >= 0: 1 / (1 + exp(-x))
        #    - For x < 0: exp(x) / (1 + exp(x))
        # 3. Cache the output y for backward
        # 4. Return y
        
        raise NotImplementedError("TODO: Implement Sigmoid.forward()")
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass: dL/dx = dL/dy * sigmoid(x) * (1 - sigmoid(x))
                             = dL/dy * y * (1 - y)
        
        Args:
            grad_output: Gradient from upstream
            
        Returns:
            Gradient w.r.t. input
        """
        # TODO: Implement Sigmoid backward pass
        # 1. Retrieve cached y = sigmoid(x)
        # 2. Compute gradient: grad_output * y * (1 - y)
        # 3. Return gradient
        
        raise NotImplementedError("TODO: Implement Sigmoid.backward()")
    
    def get_parameters(self):
        """Sigmoid has no learnable parameters."""
        return []
    
    def get_gradients(self):
        """Sigmoid has no gradients for parameters."""
        return []
