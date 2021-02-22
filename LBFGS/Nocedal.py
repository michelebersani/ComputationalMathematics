from collections import deque
import numpy as np


class NocedalAlgorithm:
    def __init__(self, M: int, B_0):
        # memory
        self.M = M
        # initial approximation of B (diagonal)
        self.B_0 = B_0
        self.saved_s = deque([])
        self.saved_y = deque([])
        self.saved_rho = deque([])

    def inverse_H_product(self, q: np.ndarray) -> np.ndarray:
        """Returns the product `B*q`, where `B` is the
        current approximation of the inverse of the Hessian.

        Args:
            q (numpy.ndarray): input vector
        """
        current_memory = len(self.saved_rho)
        if current_memory == 0:
            return self.B_0 * q

        saved_alpha = deque([])

        for i in reversed(range(current_memory)):
            new_alpha = self.saved_rho[i] * np.dot(self.saved_s[i], q)
            saved_alpha.append(new_alpha)
            q = q - new_alpha * self.saved_y[i]

        k = np.dot(self.saved_s[-1], self.saved_y[-1])
        k = k / np.dot(self.saved_y[-1], self.saved_y[-1])

        new_d = k * self.B_0 * q

        for i in range(current_memory):
            beta = self.saved_rho[i] * np.dot(self.saved_y[i], new_d)
            new_d += (saved_alpha.pop() - beta) * self.saved_s[i]
        return new_d

    def save(self, new_s, new_y, new_rho):
        self.saved_s.append(new_s)
        self.saved_y.append(new_y)
        self.saved_rho.append(new_rho)

        current_memory = len(self.saved_rho)
        if current_memory > self.M:
            self.saved_s.popleft()
            self.saved_y.popleft()
            self.saved_rho.popleft()

    def damp_y(self, s: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Returns a damped version of `y`, given `s` and `y`.
        See Nocedal's Damped L-BGFS for further details.
        """
        B_s = self.inverse_H_product(s)
        s_B_s = np.dot(s, B_s)
        s_y = np.dot(s, y)

        if s_y >= 0.2 * s_B_s:
            theta = 1
        else:
            theta = 0.8 * s_B_s / (s_B_s - s_y)

        y = theta * y + (1 - theta) * B_s
        return y