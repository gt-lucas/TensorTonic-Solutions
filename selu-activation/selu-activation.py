import numpy as np

def selu(x, lam=1.0507009873554804934193349852946, alpha=1.6732632423543772848170429916717):
    """
    实现 SELU 激活函数，返回保留 4 位小数的列表。
    """
    # 1. 转换为浮点数组
    x_arr = np.asanyarray(x, dtype=float)

    # 2. 核心计算：利用 np.where 实现分段函数向量化
    # 逻辑: if x > 0: lam * x else: lam * alpha * (exp(x) - 1)
    # 使用 np.exp 而非 exp，确保处理数组中的每一个元素
    activated = np.where(
        x_arr > 0, 
        lam * x_arr, 
        lam * alpha * (np.exp(x_arr) - 1)
    )

    # 3. 精度控制与返回
    # np.round(..., 4) 确保每个元素保留 4 位小数
    # .tolist() 确保结果为原生 Python 列表，避免 JSON 序列化错误
    return np.round(activated, 4).tolist()
