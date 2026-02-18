"""
MyTorch: A Simple Neural Network Library from Scratch

This library implements core neural network components using only NumPy.
Built as part of CS-GY 6953 Deep Learning, Spring 2026.

Structure:
    mytorch/
    |-- nn/
    |   |-- modules/
    |   |   |-- linear.py      # Linear (fully-connected) layer
    |   |   |-- activation.py  # ReLU, Sigmoid activations
    |   |   `-- loss.py        # SoftmaxCrossEntropy loss
    |   |-- sequential.py      # Sequential container
    |   `-- optim.py           # SGD optimizer
    `-- utils/
        `-- gradient_check.py  # Numerical gradient verification
"""

from . import nn
from . import utils

__version__ = "0.1.0"
