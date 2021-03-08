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
        eps: float = 1e-5,
        max_feval: int = 1000,
        m1: float = 1e-4,
        m2: float = 0.9,
        alpha0: float = 1,
        caution_thresh = 0.01,
        caution_alpha = 1,
    ):
        """Limited-memory BFGS (quasi-Newton method).

        Args:
            M (int, optional): Memory amount. Defaults to 20.
            delta (float, optional): approximation of inverse H is initialized
                as `I*delta`. Defaults to 1.
            eps (float, optional): the algorithm stops when (w.r.t. the first iteration)
                the norm of the gradient has diminished by a factor `eps`.
            max_feval (int, optional): Max number of f evaluations. Defaults to 1000.
            m1 (float, optional): parameter for Armijo's condition. Defaults to 0.0001.
            m2 (float, optional): parameter for Wolfe's condition. Defaults to 0.9.
            alpha0 (float, optional): Initial factor for line-search. Defaults to 1.

        Examples
        --------
        `f` must be callable such that `f(x)` returns `(v,g)`,
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
        self.eps_g = None
        self.max_feval = max_feval
        self.feval = 0
        self.m1 = m1
        self.m2 = m2
        self.alpha0 = alpha0
        self.caution_thresh = caution_thresh
        self.caution_alpha = caution_alpha

        self.f = None
        self.x = None
        self.g = None
        self.f_value = None
        self.new_x = None
        self.new_g = None

        self.status = None

    def solve(self, f, x):
        self.f = f
        self.x = x
        self.f_value, self.g = self.f(self.x)
        n = len(x)
        B_0 = np.repeat(self.delta, n)
        self.nocedal = NocedalAlgorithm(self.M, B_0)
        status = None
        ### log infos header
        row = ["step_size", "f value","delta f_v", "g norm", "f_evals"]
        _log_infos(row)
        ###
        while status is None:
            status = self.step()

        return status

    def step(self):
        ng = np.linalg.norm(self.g)

        if self.eps_g is None:
            self.eps_g = self.eps*ng
        if ng <= self.eps_g:
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
        alpha, new_f_value = AW_result

        ### log infos
        row = []
        row.append(f"{alpha:6.4f}")
        row.append(f"{self.f_value:1.3E}")
        row.append(f"{new_f_value-self.f_value:1.3E}")
        row.append(f"{ng:1.3E}")
        row.append(f"{self.feval}")
        _log_infos(row)
        ###
        
        # update B matrix
        s = self.new_x - self.x
        y = self.new_g - self.g
        inv_rho = np.dot(y, s)
        if inv_rho < self.eps**2*1e-2:
            return f"1/rho too small: y*s < {self.eps**2*1e-2:1.3E}"

        rho = 1 / inv_rho

        #Cautious update of B imposes this check. If fails just skip and proceed with old B
        if inv_rho/np.dot(s,s) >= self.caution_thresh * (ng ** self.caution_alpha):
            self.nocedal.save(s, y, rho)
        else:
            logging.info("--- skipped B update!")

        self.x = self.new_x
        self.g = self.new_g
        self.f_value = new_f_value

    def AW_line_search(self, d, phi0, phip0):
        alpha_max = 1e3
        alpha_i = self.alpha0
        old_alpha = 0
        old_phia = phi0

        while self.feval < self.max_feval:
            self.feval += 1
            # set new candidate point
            self.new_x = self.x + alpha_i * d
            # values for line search
            phia, self.new_g = self.f(self.new_x)
            phipa = np.dot(self.new_g, d)
            # AW conditions
            armijo = phia <= phi0 + self.m1 * alpha_i * phip0
            wolfe = abs(phipa) <= -self.m2 * phip0

            if armijo and wolfe:
                return alpha_i, phia

            if not armijo or phia > old_phia:
                return self.AW_zoom(d, phi0, phip0, old_alpha, alpha_i, old_phia)

            # at this point, alpha_i satisfies armijo but has phipa too big in abs
            if phipa > 0:
                return self.AW_zoom(d, phi0, phip0, alpha_i, old_alpha, phia)

            alpha_i = alpha_i*2

            if alpha_i > alpha_max:
                return None

            old_alpha = alpha_i
            old_phia = phia

        # No point found! D:
        return None

    def AW_zoom(self, d, phi0, phip0, alpha_low, alpha_high, phi_low):
        while self.feval < self.max_feval:
            self.feval += 1
            # set new candidate point
            alpha_j = (alpha_high + alpha_low) * 0.5
            self.new_x = self.x + alpha_j * d
            # values for line search
            phia, self.new_g = self.f(self.new_x)
            phipa = np.dot(self.new_g, d)
            # AW conditions
            armijo = phia <= phi0 + self.m1 * alpha_j * phip0
            wolfe = abs(phipa) <= -self.m2 * phip0

            if armijo and wolfe:
                return alpha_j, phia
            
            #print(phia - phi_low, alpha_j-alpha_low, phipa)

            if not armijo or phia > phi_low:
                alpha_high = alpha_j
            else:
                if phipa*(alpha_high-alpha_low) > 0:
                    alpha_high = alpha_low
                alpha_low = alpha_j
                phi_low = phia

        # max evals reached
        return None


def _log_infos(row):
    string = "{: >15} {: >15} {: >15} {: >15} {: >10}".format(*row)
    logging.info(string)