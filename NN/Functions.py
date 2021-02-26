import numpy as np

class ActivFunction():
    def __call__(self, x:np.ndarray)->np.ndarray:
        raise NotImplementedError

class LossFunction():
    def __call__(self, x:np.ndarray, y:np.ndarray)->np.ndarray:
        raise NotImplementedError


class Identity(ActivFunction):
    def __call__(self, x:np.ndarray)->np.ndarray:
        self.grad = np.ones_like(x)
        return x

class Sigmoid(ActivFunction):
    def __call__(self, x:np.ndarray)->np.ndarray:
        out = 1/(1+np.exp(-x))
        self.grad = out*(1-out)
        return out-0.5

class MSE(LossFunction):
    def __call__(self, x:np.ndarray, y:np.ndarray)->np.ndarray:
        self.grad = 2*(x-y)
        return ((y-x)**2).sum()