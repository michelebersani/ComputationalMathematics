import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import linprog
from cvxpy import *

class CuttingPlaneModel:
    def __init__(self, dim, bounds, x0, memory=None):
        self.dim = dim
        self.bounds = bounds
        self.coefficients = np.empty((0,dim+1))
        self.x0 = x0

        self.memory = memory
        self.memory_index = 0

        if memory is not None and memory < dim:
            print(f"Warning: Memory is smaller than input dimension dim={dim}, memory={memory}")

    def __call__(self, x):#REMOVE
        y = [np.sum(np.multiply(coefficients_i, np.hstack((1,x)))) for coefficients_i in self.coefficients]
        return np.max(y), 0

    def get_constraints(self):
        A_ub_x = np.asarray([[c[i] for i in range(1, len(c))] for c in self.coefficients])
        A_ub_y = np.asarray([-1 for i in range(self.coefficients.shape[0])])
        b_ub = np.asarray([-c[0] for c in self.coefficients])
        return A_ub_x, A_ub_y, b_ub, -self.bounds, self.bounds, self.x0

    def add_plane(self, f, g, x):
        c = f - np.sum(np.multiply(g,x))
        new_plane = np.append(c,g)

        # Implements cutting plane memory
        if self.memory is not None and self.coefficients.shape[0] >= self.memory:
            self.coefficients[self.memory_index] = new_plane
            self.memory_index = (self.memory_index + 1) % self.memory
        else:
            self.coefficients = np.append(self.coefficients, [new_plane], axis=0)

    def solve(self):
        A_ub_x, A_ub_y, b_ub, lb, ub, x0 = self.get_constraints()

        x = Variable(self.dim)
        y = Variable() #TODO: yp yn se non funziona per min negativo
        constraints = []
        constraints.append(A_ub_x @ x + A_ub_y * y <= b_ub)
        objective = Minimize(y)
        problem = Problem(objective, constraints)

        try:
            problem.solve(verbose=False, solver='ECOS_BB')
        except:
            problem.solve(verbose=True, solver='ECOS_BB')


        #print("Problem status: ", problem.status)
        on_border = problem.status in ['unbounded', 'infeasible']# TODO infeasible fix se possibile
        if problem.status == 'infeasible':
            print("Warning: Infeasible problem")

        if on_border:
            constraints.append(lb <= x - x0)
            constraints.append(x - x0 <= ub)
            problem = Problem(objective, constraints)
            problem.solve(verbose=False)
            #print("Problem status: ", problem.status)

        #print("MODEL min: ", x.value, y.value)

        return x.value, y.value, on_border

    def project_point(self, x0, level, max_distance_error=1e-2, verbose=False):
        n = len(x0)

        P = np.eye(n)
        q = np.multiply(x0, -2)

        x = cvxpy.Variable(n)
        y = cvxpy.Variable()

        A_ub_x, A_ub_y, b_ub, _, _, _ = self.get_constraints()

        objective = cvxpy.quad_form(x, P) + q.T @ x
        constraints = []
        constraints.append(A_ub_x @ x + A_ub_y * y <= b_ub)
        constraints.append(y == level)

        prob = cvxpy.Problem(cvxpy.Minimize(objective), constraints)
        prob.solve(verbose=verbose, max_iter=10**7, time_limit=5)

        #print("Solution = ", x.value, y.value)

        if np.abs(level-y.value) > max_distance_error:
            print("Warning, projection error above threshold: ", np.abs(level-y.value))
        return x.value

class LevelMethod:
    def __init__(self, bounds=10, lambda_=0.29289, epsilon=0.001, max_iter=1000, memory=None):
        self.bounds = bounds
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.memory = memory

        self.function = None
        self.dim = None
        self.function_points = None
        self.current_iter = None

        # Algorithm data
        self.f_upstar = None
        self.f_substar = None
        self.x_upstar = None
        self.x_substar = None
        self.x = None

    def solve(self, function, x, verbose=False):
        self.function = function
        self.dim = len(x)
        self.x = x

        # Build the cutting plane model
        self.model = CuttingPlaneModel(self.dim, self.bounds, np.ndarray.copy(x), memory=self.memory)

        self.f_upstar = math.inf
        self.x_upstar = None
        gap = math.inf
        self.current_iter = 0

        print(f"Iteration\tf*\t\tModel Min\t\tGap\t\tLevel\t         Is on boder?")
        while gap > self.epsilon:
            # Oracle computes f and g
            current_f, current_g = function.function(self.x)

            # Update model
            self.model.add_plane(current_f, current_g, self.x)

            # Compute f_substar, f_upstar, x_upstar
            self.x_substar, self.f_substar, is_on_border = self.model.solve()

            if self.f_upstar > current_f:
                self.f_upstar = current_f
                self.x_upstar = self.x

            # Project x onto level set
            gap = self.f_upstar - self.f_substar
            level = self.f_substar + self.lambda_ * gap
            if gap < -0.1:
                print("Warning: Negative gap ", gap)
                print(f"{self.current_iter}\t\t{self.f_upstar:.6f}\t{self.f_substar:.6f}\t\t{gap:.6f}\t{level:.6f}      {is_on_border}")
                break

            if is_on_border: # Project x on the border, as target level is infinite.
                self.x = self.x_substar
            else: # Project x on the target level
                self.x = self.model.project_point(self.x, level, verbose=verbose)

            print(f"{self.current_iter}\t\t{self.f_upstar:.6f}\t{self.f_substar:.6f}\t\t{gap:.6f}\t{level:.6f}      {is_on_border}")

            self.current_iter += 1
            if self.current_iter > self.max_iter:
                print("Warning: Maximum number of iterations reached.")
                break