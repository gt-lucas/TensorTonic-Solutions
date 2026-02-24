def warmup_decay_schedule(base_lr, warmup_steps, total_steps, current_step):
    """
    计算预热 + 线性衰减学习率。
    修正了整数除法可能导致的返回 0.0 的问题。
    """
    # 1. 强制类型转换，从源头避免整数除法陷阱
    base_lr = float(base_lr)
    current_step = float(current_step)
    warmup_steps = float(warmup_steps)
    total_steps = float(total_steps)

    # 2. 边界判定：超过总步数直接归零
    if current_step >= total_steps:
        return 0.0

    # 3. 阶段判定
    if warmup_steps > 0 and current_step < warmup_steps:
        # 预热阶段：从 0 线性增加到 base_lr
        lr = base_lr * (current_step / warmup_steps)
    else:
        # 衰减阶段：从 base_lr 线性减少到 0
        # 公式: base_lr * (剩余步数 / 衰减总区间)
        numerator = total_steps - current_step
        denominator = total_steps - warmup_steps
        
        # 增加防御：防止分母为 0 或产生异常
        if denominator <= 0:
            lr = 0.0
        else:
            lr = base_lr * (numerator / denominator)

    # 4. 最终结果：使用 float() 确保高精度返回
    # max(0.0, ...) 防止极小负数误差
    return float(max(0.0, lr))

# --- 验证你的 Case ---
# base_lr=0.1, warmup=10, total=100, current=55
# numerator = 100 - 55 = 45.0
# denominator = 100 - 10 = 90.0
# lr = 0.1 * (45.0 / 90.0) = 0.1 * 0.5 = 0.05 (完美匹配 Expected)