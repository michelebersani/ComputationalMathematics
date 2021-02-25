import numpy as np
from .Functions import ActivFunction, LossFunction


class NN_model:
    def __init__(self, layers: list, activ_f: ActivFunction, loss_f: LossFunction):
        # input layer does not count
        n_layers = len(layers) - 1
        # for each layer, a matrix of weights
        self.weights = []
        for i in range(n_layers):
            fan_in = layers[i]
            fan_out = layers[i + 1]
            fan_in += 1  # for the layer bias
            shape = (fan_out, fan_in)
            w = np.zeros(shape)
            self.weights.append(w)

        self.activ_f = activ_f
        self.loss_f = loss_f
        # for each layer, keep track of the input
        self.ins = [None] * n_layers
        # for each layer, keep track of the output
        # remember: outs[i] = activ_f(nets[i])
        self.outs = [None] * n_layers
        # D_out[i] is the gradient of the Loss w.r.t. outs[i]
        self.D_outs = [None] * n_layers
        # D_weights[i] is the gradient of the Loss w.r.t. weights[i]
        self.D_weights = [None] * n_layers
        # keep track of the last input to the model
        self.input = None

    def init_weights(self):
        n_layers = len(self.weights)
        # glorot's initalization
        for i in range(n_layers):
            shape_i = self.weights[i].shape
            fan_in = shape_i[1]
            fan_out = shape_i[0]
            a = 6 / (fan_in + fan_out)
            a = np.sqrt(a)
            self.weights[i] = np.random.uniform(low=-a, high=a, size=shape_i)

    def get_weights(self):
        total_weights = [w.ravel() for w in self.weights]
        return np.concatenate(total_weights)

    def set_weights(self, ws):
        p = 0
        n_layers = len(self.weights)
        for i in range(n_layers):
            size = self.weights[i].size
            slice = ws[p : p + size]
            self.weights[i] = slice.reshape(self.weights[i].shape)
            p += size

    def __call__(self, input: np.ndarray) -> np.ndarray:
        self.input = np.append(input, [1.0])  # for bias
        n_layers = len(self.weights)

        # first layer
        self.ins[0] = np.dot(self.weights[0], self.input)
        self.outs[0] = self.activ_f(self.ins[0])
        self.outs[0] = np.append(self.outs[0], [1.0])  # bias
        # other layers
        for i in range(1, n_layers):
            self.ins[i] = np.dot(self.weights[i], self.outs[i - 1])
            self.outs[i] = self.activ_f(self.ins[i])
            if i < n_layers-1:
                self.outs[i] = np.append(self.outs[i], [1.0])  # bias
        # return output of last layer
        return self.outs[-1]

    def loss(self, target: np.ndarray):
        x = self.outs[-1]
        return self.loss_f(x, target)

    def grad(self, target: np.ndarray) -> np.ndarray:
        n_layers = len(self.weights)

        loss_grad = self.loss_f.gradient(self.outs[-1], target)
        self.D_outs[-1] = loss_grad

        # loop in reverse, up to first hidden layer (excluded)
        for i in reversed(range(1,n_layers)):
            g = self.D_outs[i]
            g = g * self.activ_f.gradient(self.ins[i])
            self.D_weights[i] = np.outer(g, self.outs[i - 1])
            wT = self.weights[i].transpose()
            self.D_outs[i - 1] = np.dot(wT, g)

        # first hidden layer:
        g = self.D_outs[0][:-1] #bias not needed
        g = g * self.activ_f.gradient(self.ins[0])
        self.D_weights[0] = np.outer(g, self.input)

        # return gradient on weights
        total_gradient = [Dw.flatten() for Dw in self.D_weights]
        total_gradient = np.concatenate(total_gradient)
        return total_gradient
