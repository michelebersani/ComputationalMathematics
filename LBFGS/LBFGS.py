import numpy as np
import matplotlib.pyplot as plt
from .Nocedal import NocedalAlgorithm
from .checks import check_input
import logging


class LBFGS:
    def __init__(
        self,
        M=20,
        delta=1,
        eps=1e-10,
        max_feval=1000,
        m1=0.0001,
        m2=0.9,
        tau=0.9,
        mina=1e-16,
    ):
        self.M = M
        self.delta = delta
        self.eps = eps
        self.max_feval = max_feval
        self.feval = 0
        self.m1 = m1
        self.m2 = m2
        self.tau = tau
        self.mina = mina

        self.f = None
        self.x = None
        self.g = None
        self.f_value = None
        self.new_x = None
        self.new_g = None

    def solve(self, f, x):
        self.f = f
        self.x = x
        self.f_value = self.f.function(self.x)
        n = len(x)
        B_0 = np.repeat(self.delta, n)
        self.nocedal = NocedalAlgorithm(self.M, B_0)

        status = None
        while status is None:
            status = self.step()

        print("Exited with status:")
        print(status)
        print("Minimum function value found is:")
        print(self.f_value)

    def step(self):
        self.g = self.f.gradient(self.x)

        ng = np.linalg.norm(self.g)
        if ng <= self.eps:
            return "optimal"

        if self.feval > self.max_feval:
            return "max n. of f evaluations reached"
        
        d = -self.nocedal.inverse_H_product(self.g)
        
        phi0 = self.f_value
        phip0 = np.dot(self.g, d)
        if phip0 > 0:
            return "phip0 > 0"
        
        # find new_x and new_g using AW line search
        alpha, self.f_value, lsiter = self.AW_line_search(d, phi0, phip0)
        logging.info(f"AW line-search phase 0 steps: {lsiter[0]}")
        logging.info(f"AW line-search phase 1 steps: {lsiter[1]}")
        logging.info(f"AW line-search found {alpha:6.4f}")

        s = self.new_x - self.x
        y = self.new_g - self.g
        inv_rho = np.dot(y, s)
        if inv_rho < 1e-16:
            return "1/rho too small: y*s < 1e-16"
        
        rho = 1 / inv_rho
        self.nocedal.save(s, y, rho)

        self.x = self.new_x
        self.g = self.new_g


    def AW_line_search(self, d, phi0, phip0):
        lsiter = [0, 0]  # count iterations of phase 0 and 1
        phase = 0
        alpha = 1
        while self.feval < self.max_feval:
            self.feval += 1
            lsiter[phase] += 1
            # set new candidate point
            self.new_x = self.x + alpha * d
            self.new_g = self.f.gradient(self.new_x)
            # values for line search
            phia = self.f.function(self.new_x)
            phipa = np.dot(self.new_g, d)
            # AW conditions
            armijo = phia <= phi0 + self.m1 * alpha * phip0
            wolfe = abs(phipa) <= -self.m2 * phip0
            if armijo and wolfe:
                return alpha, phia, lsiter
            # update alpha
            alpha = alpha / self.tau if phase == 0 else alpha * self.tau
            
            # if derivative is positive, start phase 1
            if phase == 0 and phipa >= 0:
                phase = 1
                alpha = self.tau
            
            if alpha < self.mina:
                return None

        # No point found! D:
        return None
