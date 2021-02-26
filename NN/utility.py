import numpy as np
from .NN import NN_model


class Model_Wrapper():
    def __init__(self, model: NN_model, X: np.ndarray, Y: np.ndarray):
        self.model = model
        self.X = X
        self.Y = Y

    def function(self, weights:np.ndarray):
        self.model.Weights = weights
        self.model.zero_grad()
        loss = batch_train(self.model, self.X, self.Y)
        gradient = self.model.get_grad()
        return loss, gradient

def batch_train(model: NN_model, X: np.ndarray, Y: np.ndarray):
    """Runs the input `model` over the dataset `(X,Y)`,
    and returns the total loss."""
    loss = 0
    for x, y in zip(X, Y):
        model(x)
        loss += model.loss(y)
        model.grad()
    return loss


def batch_out(model: NN_model, X: np.ndarray):
    out = [model(x) for x in X]
    return np.vstack(out)