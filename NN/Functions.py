import numpy as np

class ActivFunction():
    def __call__(self, x:np.ndarray)->np.ndarray:
        raise NotImplementedError

class LossFunction():
    def __call__(self, x:np.ndarray, y:np.ndarray)->float:
        raise NotImplementedError

class RegLossFunction():
    def __call__(self, weights:np.ndarray)->float:
        raise NotImplementedError


class Identity(ActivFunction):
    def __call__(self, x:np.ndarray)->np.ndarray:
        self.grad = np.ones_like(x)
        return x

class ReLU(ActivFunction):
    def __call__(self, x:np.ndarray)->np.ndarray:
        out = np.where(x > 0, x, 0.)
        self.grad = np.where(x > 0., 1., 0.)
        return out

class Sigmoid(ActivFunction):
    def __call__(self, x:np.ndarray)->np.ndarray:
        # x_clipped = np.clip(x,-100, 100)
        x_clipped = x
        out = 1/(1+np.exp(-x_clipped))
        self.grad = out*(1-out)
        return out-0.5

class MSE(LossFunction):
    def __call__(self, x:np.ndarray, y:np.ndarray)->float:
        self.grad = 2*(x-y)
        return ((y-x)**2).sum()

class L1_reg(RegLossFunction):
    def __init__(self, alpha:float):
        self.alpha = alpha

    def __call__(self, weights:np.ndarray)->float:
        self.grad = np.ones_like(weights)
        self.grad[weights < 0] = -1
        return self.alpha*np.linalg.norm(weights,ord=1)

class L2_reg(RegLossFunction):
    def __init__(self, alpha:float):
        self.alpha = alpha

    def __call__(self, weights:np.ndarray)->float:
        self.grad = 2*weights*self.alpha
        return self.alpha*np.linalg.norm(weights)**2