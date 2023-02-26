import numpy as np
from numpy import exp, clip
from numpy.random import rand

def apply_pm_sub(x, eps):
    expeps = exp(eps)
    expeps3 = exp(eps/3)

    p_H = expeps/(expeps3 + expeps)

    CLR = (expeps+expeps3)/( expeps3*(expeps-1) )
    L = CLR*(x*expeps3 - 1)
    R = CLR*(x*expeps3 + 1)
    A = (expeps+expeps3)*(expeps3+1)/ (expeps3*(expeps-1) )

    u = rand()
    if u < p_H:
        y = rand()*(R-L) + L
    else:
        y = rand()*(L+A) - A if rand() < (L+A)/(2*A-R+L) else rand()*(A-R) + R

    #y = clip(y, -1, 1)
    #y = y/A
    return y




if __name__ == '__main__':

    eps = 2
    x = .4

    expeps = exp(eps)
    expeps3 = exp(eps/3)
    CLR = (expeps+expeps3)/( expeps3*(expeps-1) )
    L = CLR*(x*expeps3 - 1)
    R = CLR*(x*expeps3 + 1)
    A = (expeps+expeps3)*(expeps3+1)/ (expeps3*(expeps-1) )


    num = .5*expeps3*(expeps-1)
    den = (expeps3+expeps)**2
    d = num/den
    c = expeps*d

    #y_range = [apply_pm_opt(x, eps) for i in range(10000)]
    y_range = [apply_pm_sub(x, eps)/A for i in range(10000)]
    y_range = np.asarray(y_range)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.hist(y_range, bins=30, density=True)
    plt.yticks([c*A, d*A], ['c', 'd'])
    plt.xticks([-1, L/A, x, R/A, 1], [-1, 'L', 'x', 'R', 1])
    plt.show()


    pass
