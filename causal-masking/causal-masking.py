import numpy as np

def apply_causal_mask(scores, mask_value=-1e9):
    """
    scores: np.ndarray with shape (..., T, T)
    mask_value: float used to mask future positions (e.g., -1e9)
    Return: masked scores (same shape, dtype=float)
    """
    # Write code here
    scores_arr = np.asanyarray(scores, dtype=float).copy()
    T = scores_arr.shape[-1]
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    scores_arr[..., mask] = mask_value
    return scores_arr
    pass