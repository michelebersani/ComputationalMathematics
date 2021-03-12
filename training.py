import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from config import shuffled_csv
from NN import NN_model, Sigmoid, MSE, L2_reg, ReLU, L1_reg
from NN.utility import batch_train, batch_out, Model_Wrapper
from LBFGS import LBFGS


data = pd.read_csv(shuffled_csv, index_col=0).to_numpy()
data = data[:100, :]
n_samples = data.shape[0]
X_data = data[:, :10]
Y_data = data[:, 10:]

Y_scaler = StandardScaler()
Y_scaled = Y_scaler.fit_transform(Y_data)

np.random.seed(11)
model = NN_model([10, 20, 20, 2], Sigmoid, MSE)
model.init_weights()
reg_loss = L2_reg(0)

# set level to WARNING to avoid printing INFOs
logging.basicConfig(level="INFO")

solver = LBFGS(eps=1e-6,max_feval=5e4,M=20)
f = Model_Wrapper(model, X_data, Y_scaled, reg_loss)
x = model.Weights
status = solver.solve(f, x)
print("")
print(f"Exited with status: {status}")
print(f"f evaluations: {solver.feval}")
print(f"g norm: {np.linalg.norm(solver.g)}")
print(f"f value: {solver.f_value}")
print("")

Y_out = batch_out(model, X_data)
Y_out = Y_scaler.inverse_transform(Y_out)
plt.scatter(Y_data[:, 0], Y_data[:, 1],s=0.1)
plt.scatter(Y_out[:, 0], Y_out[:, 1],s=0.1)

print("MEE is:")
mee = 0
for y1, y2 in zip(Y_data, Y_out):
    mee += np.linalg.norm(y1 - y2)
print(mee / n_samples)