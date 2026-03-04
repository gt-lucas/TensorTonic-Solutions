import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """
    # Write code here
    x = np.asanyarray(x, dtype=float)
    gamma = np.asanyarray(gamma, dtype=float)
    beta = np.asanyarray(beta, dtype=float)
    axis = (0,) if x.ndim == 2 else (0,2,3)
    mu = np.mean(x, axis=axis, keepdims=True)
    var = np.var(x, axis=axis, keepdims=True)
    x_hat = (x - mu) / np.sqrt(var + eps)
    if x.ndim == 4:
        gamma = gamma.reshape(1, -1, 1, 1)
        beta = beta.reshape(1, -1, 1, 1)
    out = gamma * x_hat + beta
    return out