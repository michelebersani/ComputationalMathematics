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
        alpha0=0.9,
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
        self.alpha0=1
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
        ### log infos header
        row = ['AW LS iters [0]', 'AW LS iters [1]', 'alpha', 'f value']
        _log_infos(row)
        ###
        while status is None:
            status = self.step()
        
        return status

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
        AW_result = self.AW_line_search(d, phi0, phip0)
        if AW_result is None:
            return "AW line-search could not find a point"
        alpha, self.f_value, lsiter = AW_result
        
        ### log infos
        row = [lsiter[0], lsiter[1]]
        row.append(f"{alpha:6.4f}")
        row.append(f"{self.f_value:1.3E}")
        _log_infos(row)
        ###

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
        alpha = self.alpha0
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
                alpha = self.alpha0*self.tau
            
            if alpha < self.mina:
                return None

        # No point found! D:
        return None


def _log_infos(row):
    string = "{: >15} {: >15} {: >15} {: >15}".format(*row) 
    logging.info(string)