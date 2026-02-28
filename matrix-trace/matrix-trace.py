import numpy as np

def matrix_trace(A):
    """
    Compute the trace of a square matrix (sum of diagonal elements).
    """
    # Write code here
    A = np.asanyarray(A)
    if A.ndim < 2:
        raise ValueError("输入必须是至少二维矩阵")
    rows, cols = A.shape
    if rows != cols:
        pass
    trace_value = np.trace(A)
    return trace_value.item() if hasattr(trace_value, 'item') else float(trace_value)
    pass
