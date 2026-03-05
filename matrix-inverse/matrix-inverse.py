import numpy as np

def matrix_inverse(A):
    """
    Returns: A_inv of shape (n, n) such that A @ A_inv ≈ I
    """
    # Write code here
    A_arr = np.asanyarray(A, dtype=float)
    if A_arr.ndim != 2 or A_arr.shape[0] != A_arr.shape[1]:
        raise ValueError()
    try:
        A_arr = np.asanyarray(A, dtype=float)
        # inv 会对奇异矩阵抛出 LinAlgError
        return np.linalg.inv(A_arr)
    except np.linalg.LinAlgError:
        return None # 满足测试用例要求