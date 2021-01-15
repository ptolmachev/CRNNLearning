'''
description of sigma(h) state function which is a piecewise linear function with the slopes k and lambda'
'''

import numpy as np

def s(lmbd, k, h):
    bounds = (-1.0 / lmbd, 1.0 / lmbd)
    c = (1 - k / lmbd) #
    x = 1.0 * (h <= bounds[0])
    z = 1.0 * (h >= bounds[1])
    y = (np.ones_like(h) - x - z)
    return (k * h - c) * x + (lmbd * h) * y + (k * h + c) * z

def der_s(lmbd, k, h):
    bounds = (-1.0 / lmbd, 1.0 / lmbd)
    x = 1.0 * (h <= bounds[0])
    z = 1.0 * (h >= bounds[1])
    y = (np.ones_like(h) - x - z)
    return k * (x + z) + lmbd * y

def inv_s(lmbd, k, s):
    x = 1.0 * (s <= -1)
    z = 1.0 * (s >= 1)
    y = (np.ones_like(s) - x - z)
    c = (1 - k / lmbd)
    return ((s + c) / k) * x + (s / lmbd) * y + ((s - c) / k) * z

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    fig = plt.figure(figsize = (10, 3))
    x = np.linspace(-10,10, 100)
    plt.grid(True)
    plt.plot(x, s(0.5, 0.1, x))
    plt.xlabel("x")
    plt.ylabel("s(lmbd, k, x)")
    # plt.plot(x, inv_s(0.5, 0.1, s(0.5, 0.1, x)))
    plt.show()




