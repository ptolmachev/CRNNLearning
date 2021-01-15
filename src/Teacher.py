import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy
from collections import deque
from src.state_function import *
from tqdm.auto import tqdm

class Teacher():
    def __init__(self, dt, rnn, ad, k, lmbd, W_0, b_0, targets, etha, learning_window, TF=True):
        self.targets = targets
        self.etha = etha #learning rate
        self.lmbd = lmbd # central slope of the activation function
        self.k = k # periferal slope of the activation function
        self.dt = dt
        self.learning_window = learning_window

        self.rnn = rnn
        self.ad = ad

        self.rnn.W = W_0
        self.rnn.b = b_0
        self.rnn.lmbd = lmbd
        self.rnn.k = k
        self.rnn.dt = dt

        self.ad.W = W_0
        self.ad.b = b_0
        self.ad.lmbd = lmbd
        self.ad.k = k
        self.ad.dt = dt

        self.error_buffer = deque(maxlen=10000)
        self.t = 0
        self.curr_target_index = 0

        self.TF = TF
        # self.weights_history = []

    def step(self):
        self.rnn.step()
        self.ad.step(self.rnn.h)
        return None

    def update_history(self):
        self.rnn.update_history()
        self.ad.update_history()

        error = (s(self.lmbd, self.k, self.rnn.h) - self.targets[self.curr_target_index])
        self.error_buffer.append(error)
        self.curr_target_index += 1
        self.t += self.dt
        return None

    def calculate_grads(self):
        e = np.array(self.error_buffer)[:, :]
        n = e.shape[0]
        p_out = np.array(self.ad.p_buffer)[:, :, :, :]  # (time, outputs, i, j)
        r_out = np.array(self.ad.r_buffer)[:, :, :]  # (time, outputs)
        h_out = np.array(self.rnn.h_history)[-n:,:]  # (time, outputs)
        grad_W = np.einsum("ij,ijkl->kl", e * der_s(self.lmbd, self.k, h_out), p_out)
        np.fill_diagonal(grad_W, 0)
        grad_b = np.einsum("ij,ijk->k", e * der_s(self.lmbd, self.k, h_out), r_out)
        return grad_W, grad_b

    def train(self):
        grad_W, grad_b = self.calculate_grads()
        self.rnn.W = deepcopy(self.rnn.W - self.etha * grad_W)
        self.rnn.b = deepcopy(self.rnn.b - self.etha * grad_b)

        self.ad.W = deepcopy(self.rnn.W)
        self.ad.b = deepcopy(self.rnn.b)
        # self.weights_history.append(deepcopy(self.rnn.W))
        self.reset()
        return None

    def reset(self):
        self.ad.reset()
        self.error_buffer = deque(maxlen=10000)
        if self.TF == True:
            self.rnn.h = deepcopy(inv_s(self.lmbd, self.k, self.targets[self.curr_target_index - 1]))
        return None

    def run_training(self, T):
        for i in tqdm(range(int(T/self.dt))):
            self.step()
            self.update_history()
            if ((i % int(self.learning_window / self.dt)) == 0) and (i != 0):
                self.train()
        return None




