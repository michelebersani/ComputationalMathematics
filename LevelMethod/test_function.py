import numpy as np

class TestFunction:

    def __init__(self):
        self.dim = 2
        self.rand_distortion = np.random.rand(self.dim)*5

    def get_values(self, x_in):
        x = np.add(x_in, [2,2])
        values = np.empty((0))
        values = np.append(values, -np.sum(np.square(x)))
        values = np.append(values, 10 - np.sum(np.abs(np.add(x, self.rand_distortion))))
        values = np.append(values, np.sum(np.exp(x))/10**8)

        return values

    def get_f(self, x):
        return np.max(self.get_values(x)) - 10

    def __call__(self, x):
        f = self.get_f(x)

        ministep = 1e-4
        g = np.zeros(len(x))
        for dim in range(len(x)):
            x_minus = [x[i] if i != dim else x[i] - ministep for i in range(len(x))]
            x_plus = [x[i] if i != dim else x[i] + ministep for i in range(len(x))]
            #print("X", x_minus, x_plus, x)
            y_minus = self.get_f(x_minus)
            y_plus = self.get_f(x_plus)
            #print(y_plus, y_minus)
            g[dim] = (y_plus-y_minus)/(2*ministep)

        return f, g

if __name__ == "__main__":
    tf = TestFunction()
    f,g = tf([1,2])
    print(f, g)
