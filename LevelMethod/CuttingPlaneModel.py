import numpy as np
from cvxpy import *

class SolverFailedException(Exception):
    def __init__(self, message):
        self.message = message

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

        assert dim == len(x0)

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

    def solve(self, verbose=False):

        # Builds the LP
        A_ub_x, A_ub_y, b_ub, lb, ub, x0 = self.get_constraints()

        x = Variable(self.dim)
        y = Variable()
        constraints = []
        constraints.append(A_ub_x @ x + A_ub_y * y <= b_ub)
        objective = Minimize(y)
        problem = Problem(objective, constraints)

        try:
            problem.solve(verbose=verbose)
        except:
            raise SolverFailedException("LP failed")

        # If the problem is unbounded or infeasible we solve the bounded problem to get the projection
        on_border = problem.status in ['unbounded', 'infeasible']

        if on_border:
            constraints.append(lb <= x - x0)
            constraints.append(x - x0 <= ub)
            problem = Problem(objective, constraints)
            
            try:
                problem.solve(verbose=verbose)
            except:
                raise SolverFailedException("Bounded LP failed")

        return x.value, y.value, on_border

    def project_point(self, x0, level, max_distance_error=1e-2, verbose=False):

        # Builds the QP 
        P = np.eye(self.dim)
        q = np.multiply(x0, -2)

        x = cvxpy.Variable(n)
        y = cvxpy.Variable()

        A_ub_x, A_ub_y, b_ub, lb, ub, x0 = self.get_constraints()

        objective = cvxpy.quad_form(x, P) + q.T @ x
        constraints = []
        constraints.append(A_ub_x @ x + A_ub_y * y <= b_ub)
        constraints.append(y == level)
        constraints.append(lb <= x - x0)
        constraints.append(x - x0 <= ub)

        prob = cvxpy.Problem(cvxpy.Minimize(objective), constraints)

        try:
            prob.solve(verbose=verbose, max_iter=10**7, time_limit=5)
        except:
            raise SolverFailedException("QP failed")

        # Checks the quality of the projection
        if np.abs(level-y.value) > max_distance_error:
            print("Warning, projection error above threshold: ", np.abs(level-y.value))

        return x.value
