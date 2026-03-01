import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    # Write code here
    y_pred = np.asanyarray(y_pred)
    y_true = np.asanyarray(y_true)
    mse = np.mean(np.square(y_pred - y_true))
    return float(mse)
    pass
