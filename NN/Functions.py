import numpy as np

class ActivFunction():
    def __call__(self, x:np.ndarray)->np.ndarray:
        raise NotImplementedError

    def gradient(self, x:np.ndarray)->np.ndarray:
        raise NotImplementedError

class LossFunction():
    def __call__(self, x:np.ndarray, y:np.ndarray)->np.ndarray:
        raise NotImplementedError

    def gradient(self, x:np.ndarray, y:np.ndarray)->np.ndarray:
        raise NotImplementedError


class Sigmoid(ActivFunction):
    def __call__(self, x:np.ndarray)->np.ndarray:
        return 1/(1+np.exp(-x))

    def gradient(self, x:np.ndarray)->np.ndarray:
        v = 1/(1+np.exp(-x))
        return v*(1-v)

class MSE():
    def __call__(self, x:np.ndarray, y:np.ndarray)->np.ndarray:
        return (y-x)**2

    def gradient(self, x:np.ndarray, y:np.ndarray)->np.ndarray:
        return 2*(x-y)