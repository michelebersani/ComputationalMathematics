import numpy as np
import logging

class adam_SGD:
    def __init__(
        self,
        alpha: float = 0.01,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        k: float = 1e-8,
        eps: float = 1e-6,
        max_feval: int = 10000
    ):

        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.k = k
        self.eps = eps
        self.max_feval = max_feval
        self.f = None
        self.x = None
        self.feval = 0
        self.g = None
        self.f_value = None
    
    def solve(self, f, x):
        self.f = f
        self.x = x
        status = None
        m_t = np.zeros_like(x)
        v_t = np.zeros_like(x)
        self.f_value, self.g = f(x)
        ng = np.linalg.norm(self.g)
        ng0 = 1  # un-scaled stopping criterion
        if self.eps < 0:
            ng0 = -ng

        row = ["f value","delta f_v", "g norm", "f_evals"]
        _log_infos(row)

        while(True): #till it gets converged
            self.feval += 1
            new_f_value, self.g = f(x)
            m_t = self.beta_1 * m_t + (1-self.beta_1)*self.g
            v_t = self.beta_2 * v_t + (1-self.beta_2)*np.multiply(self.g,self.g)
            m_hat = m_t/(1-self.beta_1**(self.feval+1))
            v_hat = v_t/(1-self.beta_2**(self.feval+1))
            x = x - self.alpha*np.multiply(m_hat,1/(np.sqrt(v_hat+self.k)))
            ng = np.linalg.norm(self.g)
            ### log infos
            row = []
            row.append(f"{self.f_value:1.3E}")
            row.append(f"{new_f_value-self.f_value:1.3E}")
            row.append(f"{ng:1.3E}")
            row.append(f"{self.feval}")
            _log_infos(row)
            ###
            #STOPPING CRITERIA
            if ng <= self.eps * ng0:
                status = "optimal"
                break
            if self.feval > self.max_feval:
                status = "stopped for reached max_feval"
                break

            #UPDATE ITERATES
            self.f_value = new_f_value

def _log_infos(row):
    string = "{: >15} {: >15} {: >15} {: >10}".format(*row)
    logging.info(string)