import numpy as np

def cross_entropy_loss(y_true, y_pred):
    # 1. 类型转换与维度获取
    # y_pred 必须是 float，y_true 转为整数用于索引
    y_p = np.asanyarray(y_pred, dtype=float)
    y_t = np.asanyarray(y_true, dtype=int)
    m = y_p.shape[0]

    # 2. 数值稳定性处理
    # 防止 log(0) 导致结果变为 inf
    eps = 1e-15
    y_p = np.clip(y_p, eps, 1.0 - eps)

    # 3. 逻辑推理：根据 y_true 的形状选择计算方式
    # 如果 y_true 是 [0, 1] 这种 1D 数组，说明它是类别索引
    if y_t.ndim == 1:
        # 使用高级索引提取每行中正确类别的概率
        # y_p[np.arange(m), y_t] 等价于 [y_p[0, y_t[0]], y_p[1, y_t[1]], ...]
        correct_probs = y_p[np.arange(m), y_t]
        loss = -np.mean(np.log(correct_probs))
    else:
        # 如果 y_true 是 One-hot 矩阵，使用元素级乘法
        loss = -np.sum(y_t * np.log(y_p)) / m

    # 4. 返回原生 float 以通过 JSON 校验
    return float(loss)