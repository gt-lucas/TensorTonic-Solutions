import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """
    # Write code here
    x_arr = np.asanyarray(x, dtype=float)
    if x_arr.ndim < 3:
        raise ValueError(f"")
    pooled = np.mean(x_arr, axis=(-2, -1))
    return pooled