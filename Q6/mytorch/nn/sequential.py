"""Sequential Container for Neural Network Layers."""
import numpy as np


class Sequential:
    """
    A sequential container that chains modules together.
    
    Forward pass: applies modules in order
    Backward pass: applies modules in reverse order
    """
    
    def __init__(self, *modules):
        """
        Args:
            *modules: Variable number of modules to chain together
        """
        self.modules = list(modules)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through all modules in sequence.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after passing through all modules
        """
        out = x
        for module in self.modules:
            out = module.forward(out)
        return out
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Backward pass through all modules in reverse order.
        
        Args:
            grad: Gradient from upstream (typically from loss)
            
        Returns:
            Gradient w.r.t. input
        """
        g = grad
        for module in reversed(self.modules):
            g = module.backward(g)
        return g
    
    def get_parameters(self):
        """
        Collect all parameters from all modules.
        
        Returns:
            List of all parameter arrays
        """
        params = []
        for module in self.modules:
            params.extend(module.get_parameters())
        return params
    
    def get_gradients(self):
        """
        Collect all gradients from all modules.
        
        Returns:
            List of all gradient arrays (same order as parameters)
        """
        grads = []
        for module in self.modules:
            grads.extend(module.get_gradients())
        return grads
