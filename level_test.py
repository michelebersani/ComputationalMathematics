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
    data = data[:100,:]
    n_samples = data.shape[0]
    X_data = data[:, :10]
    Y_data = data[:, 10:]

    Y_scaler = StandardScaler()
    Y_scaled = Y_scaler.fit_transform(Y_data)

    np.random.seed()
    if test_smooth:
        model = NN_model([10, 5, 5, 2], Sigmoid, MSE)
    else:
        model = NN_model([10, 20, 20, 2], ReLU, MSE)

    # set level to WARNING to avoid printing INFOs
    logging.basicConfig(level='INFO')

    reg_loss = L1_reg(1e-4)
    f = Model_Wrapper(model, X_data, Y_scaled, reg_loss)

    if test_smooth:
        base_bound = 1
        bound_decay = 10
        max_iter = 200
        loops = 2
        max_iter = [max_iter]*loops
    else:
        base_bound = 0.5
        bound_decay = 1
        max_iter = [200]
        loops = len(max_iter)
    print(
        "\nConfiguration:",
        f"""
        base_bound = {base_bound}
        bound_decay = {bound_decay}
        max_iter = {max_iter}
        loops = {loops}
        """
    )
    
    for method in ["MOSEK"]:#, "CLP", "ECOS", "ECOS_BB", "GLPK"]:
        print(method)
        model.init_weights()
        for i in range(loops):
            bound = base_bound / (bound_decay ** i)
            solver = LevelMethod(bounds=bound, lambda_=0.5, epsilon=0.01, max_iter=max_iter[i], memory=None, LP_solver=method)
            x = model.Weights
            status = solver.solve(f,x)
            model.Weights = solver.x_upstar
            if status == -1:
                print(f"Terminato al loop {i+1}.")
                break
        
        times = solver.times
        plt.plot(times["step"][1:], label=f"Step duration")
        plt.plot(times["LP"][1:], label=f"LP duration - {method}")
        plt.plot(times["QP"][1:], label=f"QP duration - MOSEK")
    plt.legend(loc="upper left")
    plt.show()

    print('')
    print(f'Exited with status: {status}')
    print('')

    Y_out = batch_out(model, X_data)
    Y_out = Y_scaler.inverse_transform(Y_out)
    plt.scatter(Y_data[:,0], Y_data[:,1], s=1)
    plt.scatter(Y_out[:,0], Y_out[:,1], s=1)

    print('MEE is:')
    mee = 0
    for y1, y2 in zip(Y_data, Y_out):
        mee += np.linalg.norm(y1 - y2)
    print(mee/n_samples)

    plt.show()