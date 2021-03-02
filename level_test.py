import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from config import shuffled_csv
from NN import NN_model, ReLU, Sigmoid, MSE, L1_reg, L2_reg
from NN.utility import batch_train, batch_out, Model_Wrapper

from LevelMethod import LevelMethod, LevelMethod2d, TestFunction

test_2d = False
test_smooth = False

if test_2d:
    print("Hello World!")
    # 2d version is outdated:
    #   - bound is not centred in x0
    #   - does not have memory option
    f = LevelMethod2d(bounds = 20)
    f.solve(TestFunction(), [-1,-3], plot=False)
else:
    data = pd.read_csv(shuffled_csv, index_col=0).to_numpy()
    data = data[:500,:]
    n_samples = data.shape[0]
    X_data = data[:, :10]
    Y_data = data[:, 10:]

    Y_scaler = StandardScaler()
    Y_scaled = Y_scaler.fit_transform(Y_data)

    np.random.seed(10)
    if test_smooth:
        model = NN_model([10, 5, 5, 2], Sigmoid, MSE)
    else:
        model = NN_model([10, 5, 5, 2], ReLU, MSE)

    model.init_weights()

    # set level to WARNING to avoid printing INFOs
    logging.basicConfig(level='INFO')

    f = Model_Wrapper(model, X_data, Y_scaled)

    if test_smooth:
        base_bound = 1
        bound_decay = 10
        max_iter = 200
        loops = 2
    else:
        base_bound = 0.5
        bound_decay = 1
        max_iter = 500
        loops = 2

    for i in range(loops):
        bound = base_bound / (bound_decay ** i)
        print(bound)
        solver = LevelMethod(bounds=bound, lambda_=0.29289, epsilon=0.01, max_iter=max_iter * (1+i), memory=None)
        x = model.Weights
        status = solver.solve(f,x)
        model.Weights = solver.x_upstar
    
    print('')
    print(f'Exited with status: {status}')
    print('')

    Y_out = batch_out(model, X_data)
    Y_out = Y_scaler.inverse_transform(Y_out)
    plt.scatter(Y_data[:,0], Y_data[:,1], s=0.1)
    plt.scatter(Y_out[:,0], Y_out[:,1], s=0.1)

    print('MEE is:')
    mee = 0
    for y1, y2 in zip(Y_data, Y_out):
        mee += np.linalg.norm(y1 - y2)
    print(mee/n_samples)

    plt.show()