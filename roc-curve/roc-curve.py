import numpy as np

def roc_curve(y_true, y_score):
    y_true, y_score = np.asanyarray(y_true), np.asanyarray(y_score)
    desc_indices = np.argsort(y_score)[::-1]
    y_score, y_true = y_score[desc_indices], y_true[desc_indices]

    # 找到分数变化的位置
    distinct_indices = np.where(np.diff(y_score))[0]
    threshold_indices = np.r_[distinct_indices, y_true.size - 1]

    tps = np.cumsum(y_true)[threshold_indices]
    fps = np.cumsum(1 - y_true)[threshold_indices]

    fpr = np.r_[0, fps / fps[-1]]
    tpr = np.r_[0, tps / tps[-1]]
    # 严格遵循预期：第一个阈值必须是 inf
    thresholds = np.r_[np.inf, y_score[threshold_indices]]
    return fpr, tpr, thresholds