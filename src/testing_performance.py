from src.RNN import RNN
from src.AdjointDynamics import AdjointDynamics
from src.Teacher import Teacher
import numpy as np
import pickle
import os
from matplotlib import pyplot as plt
from src.plot_comparision import plot_comparison

N = 7
lmbd = 0.2
k = 0.01
dt = 0.1
etha = 5e-3
learning_window = 100 * dt

#load target data
file_name = f"targets_{N}_{lmbd}_{k}_{dt}.pkl"
file_path = os.path.join("../", "data", file_name)
signal_dict = pickle.load(open(file_path, "rb+"))
W_true = signal_dict["params"]["W"]
b_true = signal_dict["params"]["b"]
h_0 = signal_dict["h"][0]
h_out = signal_dict["h"][1:]
s_out = signal_dict["s"][1:]
targets = s_out

W_0 = W_true * (1.0 + 0.3 * np.random.randn(N,N))
np.fill_diagonal(W_0, 0)
b_0 = b_true * (1.0 + 0.3 * np.random.randn(N))

#first see how different the trajectories are
#after the training we run the target rnn
rnn_target = RNN(dt, lmbd, k,  W_true, b_true)
# and the rnn with the learned weights
rnn_follower = RNN(dt, lmbd, k,  W_0, b_0)
# set them both at the start with the same initial condition, and see how they perform
rnn_target.h = h_0
rnn_follower.h = h_0
rnn_target.run(100)
rnn_follower.run(100)
h_target = rnn_target.get_history()
h_follower = rnn_follower.get_history()
fig, ax = plot_comparison(h_target, h_follower)
plt.show(block=True)

# LEARNING
rnn = RNN(dt, lmbd, k,  W_0, b_0)
rnn.h = h_0
ad = AdjointDynamics(dt, lmbd, k,  W_0, b_0)
teacher = Teacher(dt, rnn, ad, k, lmbd, W_0, b_0, targets, etha, learning_window, TF=True)
#run training for some time
teacher.run_training(200000)
teacher.rnn.plot_history(N_last_inds=1000)
plt.show()

#extract learned weights
W_learned = teacher.rnn.W
b_learned = teacher.rnn.b

print(f"The norm of the difference between weights before the training: {np.sum((W_0 - W_true)**2)}")
print(f"The norm of the difference between weights after the training: {np.sum((W_learned - W_true)**2)}")

print(f"The norm of the difference between biases before the training: {np.sum((b_0 - b_true)**2)}")
print(f"The norm of the difference between biases after the training: {np.sum((b_learned - b_true)**2)}")

# after the training we run the target rnn
rnn_target = RNN(dt, lmbd, k,  W_true, b_true)
# and the rnn with the learned weights
rnn_follower = RNN(dt, lmbd, k,  W_learned, b_learned)
#set them both at the start with the same initial condition, and see how they perform
rnn_target.h = h_0
rnn_follower.h = h_0
rnn_target.run(200)
rnn_follower.run(200)
h_target = rnn_target.get_history()
h_follower = rnn_follower.get_history()
fig, ax = plot_comparison(h_target, h_follower)
plt.show(block=True)




