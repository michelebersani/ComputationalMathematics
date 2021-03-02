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

class ReLU(ActivFunction):
    def __call__(self, x:np.ndarray)->np.ndarray:
        out = np.where(x > 0, x, 0.)
        self.grad = np.where(x > 0., 1., np.where(x == 0, 0.5, 0.))
        return out

class Sigmoid(ActivFunction):
    def __call__(self, x:np.ndarray)->np.ndarray:
        out = 1/(1+np.exp(-x))
        self.grad = out*(1-out)
        return out-0.5

class MSE(LossFunction):
    def __call__(self, x:np.ndarray, y:np.ndarray)->np.ndarray:
        self.grad = 2*(x-y)
        return ((y-x)**2).sum()

# regularization functions
# they return directly the gradient component
# equivalent to adding a regularization term to the loss

def L1_reg(alpha:float, weights:np.ndarray)->np.ndarray:
    grad = np.ones_like(weights)
    grad[weights < 0] = -1
    return grad*alpha

def L2_reg(alpha:float, weights:np.ndarray)->np.ndarray:
    grad = weights*alpha
    return grad