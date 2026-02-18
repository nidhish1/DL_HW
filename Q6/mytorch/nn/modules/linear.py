"""Linear (Fully Connected) Layer Implementation."""
import numpy as np


class Linear:
    """
    Fully connected layer: Z = X @ W.T + b
    
    Args:
        in_features: Size of each input sample
        out_features: Size of each output sample
    
    Shapes:
        - Input X: (batch_size, in_features)
        - Weight W: (out_features, in_features)
        - Bias b: (out_features,)
        - Output Z: (batch_size, out_features)
    """
    
    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialization: W ~ N(0, 1/max(n_in, n_out)) per Problem 6 spec
        std = 1.0 / np.sqrt(max(in_features, out_features))
        self.W = np.random.randn(out_features, in_features).astype(np.float64) * std
        self.b = np.zeros(out_features, dtype=np.float64)
        
        # Gradients (computed during backward pass)
        self.dW = None
        self.db = None
        
        # Cache for backward pass
        self._cache = {}
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass: Z = X @ W.T + b
        
        Args:
            X: Input tensor of shape (batch_size, in_features)
            
        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        self._cache["X"] = X
        Z = X @ self.W.T + self.b
        return Z
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass: Compute gradients w.r.t. W, b, and X
        
        Given dL/dZ (grad_output), compute:
            dL/dW = (dL/dZ).T @ X
            dL/db = sum over batch of dL/dZ
            dL/dX = dL/dZ @ W
        
        Args:
            grad_output: Gradient from upstream, shape (batch_size, out_features)
            
        Returns:
            Gradient w.r.t. input X, shape (batch_size, in_features)
        """
        X = self._cache["X"]
        self.dW = grad_output.T @ X
        self.db = np.sum(grad_output, axis=0)
        grad_input = grad_output @ self.W
        return grad_input
    
    def get_parameters(self):
        """Return list of parameters [W, b]."""
        return [self.W, self.b]
    
    def get_gradients(self):
        """Return list of gradients [dW, db]."""
        return [self.dW, self.db]
