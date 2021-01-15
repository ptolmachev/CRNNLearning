'''
A script containing a simple RNN description:
equations:
dh/dt = -h W sigma(h) + b

sigma(h) function described in 'state_function.py'
'''
from matplotlib import pyplot as plt
from copy import deepcopy
from collections import deque
from src.state_function import *

class RNN():
    def __init__(self, dt, lmbd, k, W, b, maxlen=10000):
        self.lmbd = lmbd
        self.k = k
        self.dt = dt
        self.W = W
        self.b = b
        #number of neurons-nodes
        self.N = len(self.b)
        self.h = 10 * np.random.randn(self.N)

        self.t = 0
        self.h_history = deque(maxlen=maxlen)

    #state function
    def state(self, h):
        return s(self.lmbd, self.k, h)

    def rhs(self):
        return -self.h + self.W @ self.state(self.h) + self.b

    def step(self):
        self.h = self.h + self.dt * self.rhs()
        return None

    def update_history(self):
        self.h_history.append(deepcopy(self.h))
        self.t += self.dt
        return None

    def run(self, T):
        N_steps = int(np.ceil(T/ self.dt))
        for i in (range(N_steps)):
            self.step()
            self.update_history()
        return None

    def get_history(self):
        h_array = np.array(self.h_history)
        return h_array

    def plot_history(self, N_last_inds=False):

        transients = 100
        fig, ax = plt.subplots(self.N, 1, figsize=(15, 5))
        if N_last_inds != False:
            h_array = self.get_history()[-N_last_inds:]
            t_array = np.arange(h_array.shape[0]) * self.dt
        else:
            h_array = self.get_history()[transients:]
            t_array = np.arange(h_array.shape[0]) * self.dt
        for i in range(self.N):
            ax[i].plot(t_array, h_array[:, i], linewidth=2, color='k')
            if (i == self.N//2):
                ax[i].set_ylabel(f'h', fontsize=24, rotation=0)
        ax[-1].set_xlabel('t', fontsize=24)
        plt.subplots_adjust(hspace=0)
        plt.suptitle(f"Trajectory of a neural network, N={self.N}, lmbd={self.lmbd}, k={self.k}", fontsize=24)
        return fig, ax

if __name__ == '__main__':
    N = 5
    W = 0.1 * (np.random.randn(N, N))
    np.fill_diagonal(W, 0)
    b = 0.1 * np.random.randn(N)
    lmbd = 0.5
    k = 0.1
    dt = 0.05
    rnn = RNN(dt, lmbd, k, W, b)
    T = 100
    rnn.run(T)
    fig, ax = rnn.plot_history()
    plt.show(block=True)







