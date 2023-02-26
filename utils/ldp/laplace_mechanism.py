import numpy as np
from numpy import clip
from numpy.random import laplace

# for input bounded in [-1, 1], meaning that sensitivity=2
def apply_laplace(x, eps):
    y = x + laplace(scale=2/eps)
    # y = clip(y, -1, 1)
    return y
