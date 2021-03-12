import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from config import shuffled_csv
from NN import NN_model, ReLU, Sigmoid, MSE, L1_reg, L2_reg
from NN.utility import batch_train, batch_out, Model_Wrapper

from LevelMethod import LevelMethod, LevelMethod2d, TestFunction

import time

data = pd.read_csv(shuffled_csv, index_col=0).to_numpy()
data = data[:100,:]
n_samples = data.shape[0]
X_data = data[:, :10]
Y_data = data[:, 10:]

Y_scaler = StandardScaler()
Y_scaled = Y_scaler.fit_transform(Y_data)

np.random.seed(10)

model = NN_model([10, 5, 5, 2], ReLU, MSE)

reg_loss = L1_reg(1e-5)
f = Model_Wrapper(model, X_data, Y_scaled, reg_loss)

lambdas = [0.25, 0.5, 0.75]
qs = [0.5, 1, 10]
ns = [1, 3, 5]
ms = [500, 1000, 1500]
configs = np.asarray([[l,q,n,m] for l in lambdas for q in qs for n in ns for m in ms])
mean_mee = np.empty_like(configs)
duration = np.empty_like(configs)

for idx, configuration in enumerate(configs):
    t0 = time.time()

    lambda_ = configuration[0]
    q = configuration[1]
    n = configuration[2]
    m = configuration[3]
    print(
        "\nConfiguration:",
        f"""
        lambda = {lambda_}
        Q side length = {q}
        loops = {n}
        max_iter = {m}
        """
    )
    
    model.init_weights()
    for i in range(int(n)):
        solver = LevelMethod(bounds=q, lambda_=lambda_, epsilon=0.01, max_iter=m)
        x = model.Weights
        status = solver.solve(f,x)
        model.Weights = solver.x_upstar
        if status == -1:
            print(f"Terminato al loop {i+1}.")
            break

    print('')
    print(f'Exited with status: {status}')
    print('')

    Y_out = batch_out(model, X_data)
    Y_out = Y_scaler.inverse_transform(Y_out)
    #plt.scatter(Y_data[:,0], Y_data[:,1], s=0.1)
    #plt.scatter(Y_out[:,0], Y_out[:,1], s=0.1)

    print('MEE is:')
    mee = 0
    for y1, y2 in zip(Y_data, Y_out):
        mee += np.linalg.norm(y1 - y2)
    print(mee/n_samples)

    mean_mee[idx] = mee/n_samples
    duration[idx] = time.time() - t0
    #plt.show()

np.savetxt("configs.csv", configs, delimiter=",")
np.savetxt("mean_mee.csv", mean_mee, delimiter=",")
np.savetxt("duration.csv", duration, delimiter=",")