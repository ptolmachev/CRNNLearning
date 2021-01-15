from matplotlib import pyplot as plt
import numpy as np

def plot_comparison(h_target, h_follower):
    N = h_target.shape[-1]
    fig, ax = plt.subplots(N, 1, figsize=(15, 5))
    for i in range(N):
        ax[i].plot(h_target[:, i], linewidth=2, color='r', linestyle ='--', alpha = 0.5)
        ax[i].plot(h_follower[:, i], linewidth=2, color='k', linestyle='-')
        if (i == N // 2):
            ax[i].set_ylabel(f'h', fontsize=24, rotation=0)
    ax[-1].set_xlabel('t', fontsize=24)
    plt.subplots_adjust(hspace=0)
    # plt.suptitle(f"Trajectory of a neural network, N={self.N}, lmbd={self.lmbd}, k={self.k}", fontsize=24)
    return fig, ax