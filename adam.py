import numpy as np
import logging
import time
import matplotlib.pyplot as plt

class adam_SGD:
    def __init__(
        self,
        alpha: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        k: float = 1e-8,
        eps: float = 1e-6,
        max_feval: int = 10000,
        plot: bool = False,
        verbose:bool = False
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
        self.plot = plot
        self.verbose = verbose
        self.start_time = None
        self.time_evaluations = []


    def plot_fvalues(self):
        plt.plot(self.f_values)
        plt.yscale("log")
        plt.xlabel("f evaluations")

        plt.ylabel("Model Loss")
        plt.show()

    def solve(self, f, x):
        self.f = f
        self.x = x
        self.f_values = []
        status = None
        m_t = np.zeros_like(x)
        v_t = np.zeros_like(x)
        self.f_value, self.g = f(x)
        self.start_time = time.process_time()
        ng = np.linalg.norm(self.g)
        ng0 = ng
        
        row = ["f value","delta f_v", "g norm", "f_evals"]
        _log_infos(row)

        while(True): #till it gets converged
            self.feval += 1
            new_f_value, self.g = f(x)
            self.f_values.append(new_f_value)
            current_time = time.process_time()
            self.time_evaluations.append(current_time-self.start_time)
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
            
            if self.feval > self.max_feval:
                status = "optimal"
                #status = "stopped for reached max_feval"
                break

            #UPDATE ITERATES
            self.f_value = new_f_value

        if self.plot:
            self.plot_fvalues()

        if(self.verbose):
            print(f"f min reached: \t{min(self.f_values)}")
        return status


def _log_infos(row):
    string = "{: >15} {: >15} {: >15} {: >10}".format(*row)
    logging.info(string)