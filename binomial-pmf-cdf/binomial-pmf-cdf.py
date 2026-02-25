import numpy as np
from scipy.special import comb

def binomial_pmf_cdf(n, p, k):
    """
    Compute Binomial PMF and CDF.
    """
    # Write code here
    pmf = comb(n, k) * (p**k) * ((1-p) ** (n-k))
    cdf = 0.0
    for i in range(int(k)+1):
        prob_i = comb(n, i) * (p ** i) * ((1-p) ** (n-i))
        cdf += prob_i
    return [float(pmf), float(cdf)]
    pass