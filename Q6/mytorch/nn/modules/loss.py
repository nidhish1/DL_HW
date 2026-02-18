"""Loss Function Implementations."""
import numpy as np


class SoftmaxCrossEntropy:
    """
    Combined Softmax and Cross-Entropy Loss for numerical stability.
    
    Forward:
        softmax: y_hat = exp(z - max(z)) / sum(exp(z - max(z)))
        loss: L = -sum(y * log(y_hat)) averaged over batch
        
    Backward:
        dL/dz = (y_hat - y) / batch_size
    """
    
    def __init__(self):
        self._cache = {}
    
    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute softmax cross-entropy loss.
        
        Args:
            logits: Raw scores from the network, shape (batch_size, num_classes)
            targets: One-hot encoded true labels, shape (batch_size, num_classes)
            
        Returns:
            Scalar loss value (averaged over batch)
        """
        # Numerically stable softmax: subtract max per row
        z_stable = logits - np.max(logits, axis=1, keepdims=True)
        exp_z = np.exp(z_stable)
        y_hat = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        # Cross-entropy: L = -(1/N) * sum_n sum_i y_i^(n) log(y_hat_i^(n))
        eps = 1e-12
        N = logits.shape[0]
        L = -np.sum(targets * np.log(y_hat + eps)) / N

        self._cache["y_hat"] = y_hat
        self._cache["targets"] = targets
        self._cache["N"] = N
        return float(L)
    
    def backward(self) -> np.ndarray:
        """
        Compute gradient of loss w.r.t. logits.
        
        Returns:
            Gradient dL/dz of shape (batch_size, num_classes)
        """
        y_hat = self._cache["y_hat"]
        targets = self._cache["targets"]
        N = self._cache["N"]
        return (y_hat - targets) / N
    
    def get_parameters(self):
        """Loss has no learnable parameters."""
        return []
    
    def get_gradients(self):
        """Loss has no gradients for parameters."""
        return []
