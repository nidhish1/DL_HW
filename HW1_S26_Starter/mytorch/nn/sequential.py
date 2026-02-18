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
        # TODO: Implement forward pass
        # Loop through modules and apply each one in order
        
        raise NotImplementedError("TODO: Implement Sequential.forward()")
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Backward pass through all modules in reverse order.
        
        Args:
            grad: Gradient from upstream (typically from loss)
            
        Returns:
            Gradient w.r.t. input
        """
        # TODO: Implement backward pass
        # Loop through modules in REVERSE order and apply backward
        
        raise NotImplementedError("TODO: Implement Sequential.backward()")
    
    def get_parameters(self):
        """
        Collect all parameters from all modules.
        
        Returns:
            List of all parameter arrays
        """
        # TODO: Implement parameter collection
        # Loop through modules and collect all parameters
        
        raise NotImplementedError("TODO: Implement Sequential.get_parameters()")
    
    def get_gradients(self):
        """
        Collect all gradients from all modules.
        
        Returns:
            List of all gradient arrays (same order as parameters)
        """
        # TODO: Implement gradient collection
        # Loop through modules and collect all gradients
        
        raise NotImplementedError("TODO: Implement Sequential.get_gradients()")
