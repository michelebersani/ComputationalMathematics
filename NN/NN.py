import numpy as np
from .Functions import ActivFunction, LossFunction, Identity


class NN_model:
    def __init__(
        self,
        layers: list,
        activ_f: ActivFunction,
        loss_f: LossFunction,
        output_f: ActivFunction = Identity,
    ):
        """Feed-forward NN model.

        Args:
            layers (list): number of units per layer.
            activ_f (ActivFunction): activation function for the hidden layers.
            loss_f (LossFunction): loss function.
            output_f (ActivFunction): activation function for the output layer.
                Defaults to `Identity`.
    
        Examples
        --------
        The X and Y are lists of `numpy` vectors, with lenths of 4 and 2 respectively.
        >>> model = NN_model([4, 5, 3, 2], Sigmoid, MSE)
        >>> model = model.init_weights()
        >>> for i in range(len(X)):
        ...     model.zero_grad()
        ...     out = model(X[i])
        ...     loss = model.loss(Y[i])
        ...     print(loss)
        ...     g = model.grad().get_grad()
        ...     ws = model.get_weights()
        ...     ws = ws - 0.5*g
        ...     model.set_weights(ws)
        ...
        """
        # input layer does not count
        n_layers = len(layers) - 1
        # for each layer, a matrix of weights
        self.weights = []
        # D_weights[i] is the gradient of the Loss w.r.t. weights[i]
        self.D_weights = []
        for i in range(n_layers):
            fan_in = layers[i]
            fan_out = layers[i + 1]
            fan_in += 1  # for the layer bias
            shape = (fan_out, fan_in)
            w = np.zeros(shape)
            D_w = np.zeros(shape)
            self.weights.append(w)
            self.D_weights.append(D_w)

        self.activ_fs = [activ_f() for i in range(n_layers - 1)]
        # output layer with Identity activ function
        self.activ_fs.append(Identity())
        self.loss_f = loss_f()
        # for each layer, keep track of the output
        self.outs = [None] * n_layers
        # D_outs[i] is the gradient of the Loss w.r.t. outs[i]
        self.D_outs = [None] * n_layers
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
        ls_in = np.dot(self.weights[0], self.input)
        self.outs[0] = self.activ_fs[0](ls_in)
        self.outs[0] = np.append(self.outs[0], [1.0])  # bias
        # other layers
        for i in range(1, n_layers):
            ls_in = np.dot(self.weights[i], self.outs[i - 1])
            self.outs[i] = self.activ_fs[i](ls_in)
            if i < n_layers - 1:
                self.outs[i] = np.append(self.outs[i], [1.0])  # bias
        # return output of last layer
        return self.outs[-1]

    def loss(self, target: np.ndarray):
        x = self.outs[-1]
        return self.loss_f(x, target)

    def grad(self) -> np.ndarray:
        n_layers = len(self.weights)

        loss_grad = self.loss_f.grad
        self.D_outs[-1] = loss_grad

        # loop in reverse, up to first hidden layer (excluded)
        for i in reversed(range(1, n_layers)):
            g = self.D_outs[i]
            g = g * self.activ_fs[i].grad
            self.D_weights[i] += np.outer(g, self.outs[i - 1])
            # remove bias column and transpose
            wT = self.weights[i][:, :-1].transpose()
            self.D_outs[i - 1] = np.dot(wT, g)

        # first hidden layer uses self.input
        g = self.D_outs[0]
        g = g * self.activ_fs[0].grad
        self.D_weights[0] += np.outer(g, self.input)
        return self

    def get_grad(self):
        # return gradient on weights
        total_gradient = [Dw.flatten() for Dw in self.D_weights]
        total_gradient = np.concatenate(total_gradient)
        return total_gradient

    def zero_grad(self):
        # set the gradient to zero
        for D_w in self.D_weights:
            D_w.fill(0.0)
