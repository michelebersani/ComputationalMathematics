import numpy as np
import matplotlib.pyplot as plt
from .Nocedal import NocedalAlgorithm
from .checks import check_input
import logging


class LBFGS:
    def __init__(
        self,
        M: int = 20,
        delta: float = 1,
        eps: float = 1e-10,
        max_feval: int = 1000,
        m1: float = 0.0001,
        m2: float = 0.9,
        tau: float = 0.9,
        alpha0: float = 1,
        caution_thresh = 0.01,
        caution_alpha = 1,
        mina: float = 1e-16,
    ):
        """Limited-memory BFGS (quasi-Newton method).

        Args:
            M (int, optional): Memory amount. Defaults to 20.
            delta (float, optional): approximation of inverse H is initialized
                as `I*delta`. Defaults to 1.
            eps (float, optional): the algorithm stops when
                `||gradient|| < eps`. Defaults to 1e-10.
            max_feval (int, optional): Max number of f evaluations. Defaults to 1000.
            m1 (float, optional): parameter for Armijo's condition. Defaults to 0.0001.
            m2 (float, optional): parameter for Wolfe's condition. Defaults to 0.9.
            tau (float, optional): Exponential factor for line-search. Defaults to 0.9.
            alpha0 (float, optional): Initial factor for line-search. Defaults to 1.
            mina (float, optional): Min factor for line-search. Defaults to 1e-16.

        Examples
        --------
        `f` must be an object that implements a `f.function(x)` method returning `(v,g)`,
        where `v` and `g` are respectively the value and the gradient of `f` in `x`.
        The `x` and the gradient must be `numpy.ndarray`.
        >>> solver = LBFGS()
        >>> status = solver.solve(f,x)
        >>> print(status, "Solution found:", solver.x)
        >>> print("Value of f in solution:", solver.f_value)
        """
        self.M = M
        self.delta = delta
        self.eps = eps
        self.max_feval = max_feval
        self.feval = 0
        self.m1 = m1
        self.m2 = m2
        self.tau = tau
        self.alpha0 = alpha0
        self.caution_thresh = caution_thresh
        self.caution_alpha = caution_alpha
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
        n = len(x)
        B_0 = np.repeat(self.delta, n)
        self.nocedal = NocedalAlgorithm(self.M, B_0)

        status = None
        ### log infos header
        row = ["AW LS [0]", "AW LS [1]", "alpha", "f value", "g norm","s norm", "f_evals"]
        _log_infos(row)
        ###
        while status is None:
            status = self.step()

        return status

    def step(self):
        self.f_value, self.g = self.f.function(self.x)

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

        s = self.new_x - self.x
        ns = np.linalg.norm(s)
        y = self.new_g - self.g
        inv_rho = np.dot(y, s)
        if inv_rho < self.eps**2:
            return f"1/rho too small: y*s < {self.eps**2:1.3E}"

        ### log infos
        row = [lsiter[0], lsiter[1]]
        row.append(f"{alpha:6.4f}")
        row.append(f"{self.f_value:1.3E}")
        row.append(f"{ng:1.3E}")
        row.append(f"{ns:1.3E}")
        row.append(f"{self.feval}")
        _log_infos(row)
        ###

        rho = 1 / inv_rho

        #Cautious update of B imposes this check. If fails just skip and proceed with old B
        if inv_rho/np.dot(s,s) >= self.caution_thresh * (ng ** self.caution_alpha):
            self.nocedal.save(s, y, rho)
        else:
            logging.info("--- skipped B update!")

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
            # values for line search
            phia, self.new_g = self.f.function(self.new_x)
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
                alpha = self.alpha0 * self.tau

            if alpha < self.mina:
                return None

        # No point found! D:
        return None


def _log_infos(row):
    string = "{: >10} {: >10} {: >10} {: >10} {: >10} {: >10} {: >10}".format(*row)
    logging.info(string)