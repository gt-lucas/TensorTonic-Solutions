import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Compute Triplet Loss for embedding ranking.
    """
    # Write code here
    A = np.asanyarray(anchor, dtype=float)
    P = np.asanyarray(positive, dtype=float)
    N = np.asanyarray(negative, dtype=float)

    dist_pos = np.sum(np.square(A - P), axis=-1)
    dist_neg = np.sum(np.square(A - N), axis=-1)

    losses = np.maximum(0.0, dist_pos - dist_neg + margin)
    return float(np.mean(losses))
    pass