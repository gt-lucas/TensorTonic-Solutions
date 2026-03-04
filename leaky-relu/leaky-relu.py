import numpy as np

def leaky_relu(x, alpha=0.01):
    """
    Vectorized Leaky ReLU implementation.
    """
    # Write code here
    x_arr = np.asanyarray(x, dtype = float)
    result = np.where(x_arr>0, x_arr, alpha*x_arr)
    return result