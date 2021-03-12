import numpy as np
import math
from .CuttingPlaneModel import CuttingPlaneModel, SolverFailedException

import time

class LevelMethodException(Exception):
    def __init__(self,message):
        self.message = message

class LevelMethodMaxIter(Exception):
    def __init__(self,message):
        self.message = message

class LevelMethod:
    def __init__(self, bounds=10, lambda_=0.29289, epsilon=0.001, max_iter=1000, memory=None, LP_solver="MOSEK"):
        self.bounds = bounds
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.memory = memory
        self.LP_solver = LP_solver

        self.function = None
        self.dim = None
        self.current_iter = None

        # Algorithm data
        self.f_upstar = None
        self.f_substar = None
        self.x_upstar = None
        self.x_substar = None
        self.x = None

        self.step_times = {
            "step": []
        }

    def solve(self, function, x, verbose=False):
        self.function = function
        self.dim = len(x)
        self.x = x

        # Build the cutting plane model
        self.model = CuttingPlaneModel(self.dim, self.bounds, np.ndarray.copy(x), memory=self.memory, LP_solver=self.LP_solver)

        self.f_upstar = math.inf
        self.x_upstar = None
        self.current_iter = 0
        gap = math.inf

        print(f"Iteration\tf*\t\tModel Min\t\tGap\t\tLevel\t")

        while gap > self.epsilon:
            try:
                t0 = time.time()
                gap = self.step(verbose=verbose)
                deltat = time.time()-t0
                self.times["step"].append(deltat)

            except (SolverFailedException, LevelMethodException) as e:
                print(type(e).__name__, e.message)
                return -1

            except LevelMethodMaxIter as e:
                print(type(e).__name__, e.message)
                return 0

        return 0
    
    def step(self, verbose=False):
        # Oracle computes f and g
        current_f, current_g = self.function(self.x)

        # Update model
        self.model.add_plane(current_f, current_g, self.x)

        # Compute f_substar, f_upstar, x_upstar
        self.x_substar, self.f_substar = self.model.solve()

        if self.f_upstar > current_f:
            self.f_upstar = current_f
            self.x_upstar = self.x

        # Compute target level and log iteration info
        gap = self.f_upstar - self.f_substar
        level = self.f_substar + self.lambda_ * gap

        print(f"{self.current_iter}\t\t{self.f_upstar:.6f}\t{self.f_substar:.6f}\t\t{gap:.6f}\t{level:.6f}")

        if gap < 0:
            raise LevelMethodException(f"Warning: Negative gap {gap}")

        # Project x on target level
        self.x = self.model.project_point(self.x, level)

        self.current_iter += 1
        if self.current_iter > self.max_iter:
            raise LevelMethodMaxIter("Warning: Maximum number of iterations reached.")

        return gap

    @property
    def times(self):
        merged_times = {**self.step_times, **self.model.times}
        return merged_times