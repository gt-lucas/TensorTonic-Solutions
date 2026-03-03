import numpy as np

def adamw_step(w, m, v, grad, lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.01, eps=1e-8):
    """
    执行一次 AdamW (Decoupled Weight Decay) 更新步骤。
    
    逻辑推演:
    1. 权重衰减不再加在梯度上，而是直接从当前权重中扣除。
    2. 使用移动平均更新一阶动量 (m) 和二阶动量 (v)。
    3. 利用自适应梯度缩放更新最终权重。
    """
    # 1. 预处理：确保输入为高性能 NumPy 数组并维持浮点精度
    w_arr = np.asanyarray(w, dtype=float)
    m_arr = np.asanyarray(m, dtype=float)
    v_arr = np.asanyarray(v, dtype=float)
    g_arr = np.asanyarray(grad, dtype=float)

    # 2. 核心逻辑 A：解耦权重衰减 (Decoupled Weight Decay)
    # 这一步是 AdamW 区别于普通 Adam 的关键：在更新梯度前先收缩权重
    w_arr = w_arr * (1.0 - lr * weight_decay)

    # 3. 核心逻辑 B：更新一阶和二阶动量
    # m = beta1 * m + (1 - beta1) * g
    # v = beta2 * v + (1 - beta2) * g^2
    new_m = beta1 * m_arr + (1.0 - beta1) * g_arr
    new_v = beta2 * v_arr + (1.0 - beta2) * np.square(g_arr)

    # 4. 核心逻辑 C：自适应步长更新
    # 注意：eps 必须在根号之外以保证数值稳定性
    denom = np.sqrt(new_v) + eps
    new_w = w_arr - lr * (new_m / denom)

    # 5. 格式化输出：返回满足 JSON 序列化要求的嵌套列表
    return new_w, new_m, new_v