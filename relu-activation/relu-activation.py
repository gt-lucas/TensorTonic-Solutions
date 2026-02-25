import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    # Write code here
    X = np.asanyarray(x)
    out = np.maximum(0, X)
    return out
    pass