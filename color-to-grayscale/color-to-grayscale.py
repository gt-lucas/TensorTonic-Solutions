import numpy as np

def color_to_grayscale(image):
    """
    使用 NumPy 进行高性能计算，并返回 Python 原生嵌套列表。
    """
    # 1. 预处理：确保输入是 NumPy 数组
    # 如果输入已经是列表，这一步会将其转为数组以便矩阵运算
    img = np.asanyarray(image)

    # 2. 权重定义 (Rec. 601 标准)
    weights = np.array([0.299, 0.587, 0.114])

    # 3. 核心计算：点积运算 (Luminance Calculation)
    # img[..., :3] 自动处理 RGB 或 RGBA (忽略第 4 通道)
    # 运算结果是一个形状为 (H, W) 的 2D NumPy 数组
    grayscale_array = np.dot(img[..., :3], weights)

    # 4. 类型转换与格式输出
    # A. 首先四舍五入并转为整数（通常灰度图像素为 0-255）
    # B. 调用 .tolist() 将 2D 数组转为嵌套列表 [[...], [...]]
    return grayscale_array.tolist()