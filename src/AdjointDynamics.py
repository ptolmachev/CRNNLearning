import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy
from collections import deque
from src.state_function import *

class AdjointDynamics():
    def __init__(self, dt, lmbd, k, W, b):
        self.dt = dt
        self.N = len(b)
        self.W = W
        self.b = b
        self.k = k
        self.lmbd = lmbd
        self.p = np.zeros((self.N, self.N, self.N))
        self.r = np.zeros((self.N, self.N))
        self.t = 0

        #buffers
        self.p_buffer = deque(maxlen=10000)
        self.r_buffer = deque(maxlen=10000)
        # self.p_buffer.append(deepcopy(self.p))
        # self.r_buffer.append(deepcopy(self.r))

    def rhs_p(self, h):
        # dp^i_{jk} / dt = -p^i_{jk} + D_{ji}s_k + sum_l w_{il} s'_l p^l_{jk}
        RHS_P = - self.p \
                + np.einsum('ij,k->ijk', np.eye(self.N), s(self.lmbd, self.k, h)) \
                + np.einsum('ij,j,jkl->ikl', self.W, der_s(self.lmbd, self.k, h), self.p)
        return RHS_P

    def rhs_r(self, h):
        # dr^i_{j} / dt = -r^i_k + sum_j w_{ij} s'_j r^j_k + D_{ik}
        RHS_R = - self.r \
                + np.einsum('ij,j,jk->ik', self.W, der_s(self.lmbd, self.k, h), self.r) \
                + np.eye(self.N)
        return RHS_R

    def step(self, h):
        self.p += self.dt * self.rhs_p(h)
        self.r += self.dt * self.rhs_r(h)
        return None

    def update_history(self):
        self.p_buffer.append(deepcopy(self.p))
        self.r_buffer.append(deepcopy(self.r))
        self.t += self.dt
        return None

    def reset(self):
        self.p_buffer = deque(maxlen=10000)
        self.r_buffer = deque(maxlen=10000)
        self.p = np.zeros((self.N, self.N, self.N))
        self.r = np.zeros((self.N, self.N))
        return None


