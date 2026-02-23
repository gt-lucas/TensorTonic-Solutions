def differencing(series, order):
    res = list(series)
    for _ in range(order):
        new_res = []
        for i in range(1, len(res)):
            new_res.append(res[i] - res[i-1])
        res = new_res
    return res