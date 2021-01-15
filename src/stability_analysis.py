'''
script which contains routine of finding equilibria and a Jacobian around these points,
to check the stability using the largest eigenvalue of the Jacobian
'''
import numpy as np
from scipy.optimize import fsolve
from scipy.linalg import eig
from src.state_function import *
import numdifftools as nd
import warnings
warnings.filterwarnings("ignore")

def calc_equilibria(lmbd, k, W, b):
    # you get a function which needs to be solved
    # h = W sigma(h) + b
    def func(h):
        return -h + W @ s(lmbd, k, h) + b
    fps = []
    fp_hashes = []
    #make sure you find all the fixed points
    for i in range(101):
        fp = fsolve(func, 100 * np.random.rand(len(b)), xtol=1e-18)
        fp_rounded = np.round(fp, 2)
        fp_hash = hash(np.sum(fp_rounded)**2)
        if fp_hash in fp_hashes:
            pass
        else:
            fp_hashes.append(fp_hash)
            fps.append(fp)
    return fps

# def calculate_jacobian(h_star, lmbd, W):
#     N = len(h_star)
#     return -np.identity(N) + W * der_s(lmbd, h_star)

def calculate_Jacobian(h_star, lmbd, k, W, b):

    def func(h):
        return -h + W @ s(lmbd, k, h) + b

    return nd.Jacobian(func)(h_star)

if __name__ == '__main__':
    N = 5
    lmbd = 0.5
    k = 0.1
    W = np.random.randn(N,N)
    np.fill_diagonal(W, 0)

    b = np.random.randn(N)
    fps = calc_equilibria(lmbd, k, W, b)
    print(f'fixed points: {fps}')
    for i in range(len(fps)):
        fp = fps[i]
        jac = calculate_Jacobian(fp, lmbd, k, W, b)
        res = eig(jac)
        print(f'Re(egs): {np.real(res[0])}')
