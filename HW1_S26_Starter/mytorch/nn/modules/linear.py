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
        
        # Xavier/Glorot initialization: W ~ N(0, 2/(fan_in + fan_out))
        # TODO: Initialize weights using Xavier initialization
        std = None  # TODO: Calculate std = sqrt(2 / (in_features + out_features))
        self.W = None  # TODO: Sample from N(0, std^2) with shape (out_features, in_features)
        self.b = None  # TODO: Initialize bias to zeros with shape (out_features,)
        
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
        # TODO: Implement forward pass
        # 1. Cache X for use in backward pass
        # 2. Compute Z = X @ W.T + b
        # 3. Return Z
        
        raise NotImplementedError("TODO: Implement Linear.forward()")
    
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
        # TODO: Implement backward pass
        # 1. Retrieve cached X
        # 2. Compute dL/dW = grad_output.T @ X
        # 3. Compute dL/db = sum of grad_output over batch dimension
        # 4. Compute dL/dX = grad_output @ W
        # 5. Store self.dW and self.db
        # 6. Return dL/dX
        
        raise NotImplementedError("TODO: Implement Linear.backward()")
    
    def get_parameters(self):
        """Return list of parameters [W, b]."""
        return [self.W, self.b]
    
    def get_gradients(self):
        """Return list of gradients [dW, db]."""
        return [self.dW, self.db]
