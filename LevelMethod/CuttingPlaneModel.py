import numpy as np

from cvxpy import *
from cvxopt import matrix, solvers
from mosek import iparam
solvers.options['mosek'] = {iparam.log: 0}

import time

class SolverFailedException(Exception):
    def __init__(self, message):
        self.message = message

class CuttingPlaneModel:
    def __init__(self, dim, bounds, x0, memory=None, LP_solver="mosek", QP_solver="mosek"):
        self.dim = dim
        self.bounds = bounds
        self.coefficients = np.empty((0,dim+1))
        self.x0 = x0
        self.LP_solver = LP_solver
        self.QP_solver = QP_solver

        self.memory = memory
        self.memory_index = 0

        self.times = {
            "LP": [],
            "QP": []
        }

        if memory is not None and memory < dim:
            print(f"Warning: Memory is smaller than input dimension dim={dim}, memory={memory}")

        assert dim == len(x0)

    def __call__(self, x):#REMOVE
        y = [np.sum(np.multiply(coefficients_i, np.hstack((1,x)))) for coefficients_i in self.coefficients]
        return np.max(y), 0

    def add_plane(self, f, g, x):
        c = f - np.sum(np.multiply(g,x))
        new_plane = np.append(c,g)

        # Implements cutting plane memory
        if self.memory is not None and self.coefficients.shape[0] >= self.memory:
            self.coefficients[self.memory_index] = new_plane
            self.memory_index = (self.memory_index + 1) % self.memory
        else:
            self.coefficients = np.append(self.coefficients, [new_plane], axis=0)

    def get_LP_constraints(self):
        A = np.asarray([[c[i] for i in range(1, len(c))] + [-1] for c in self.coefficients])
        A_ub = np.hstack((np.eye(self.dim),np.zeros((self.dim,1))))
        A_lb = -np.hstack((np.eye(self.dim),np.zeros((self.dim,1))))
        
        A = np.vstack((A, A_ub, A_lb))

        b = np.asarray([-c[0] for c in self.coefficients])
        b_ub = self.x0 + self.bounds
        b_lb = -self.x0 + self.bounds
        
        b = np.concatenate((b, b_ub, b_lb))
        
        return A, b

    def get_QP_constraints(self, level):
        A = np.asarray([[c[i] for i in range(1, len(c))] for c in self.coefficients])
        A_ub = np.eye(self.dim)
        A_lb = -np.eye(self.dim)
        
        A = np.vstack((A, A_ub, A_lb))

        b = np.asarray([-c[0] + level for c in self.coefficients])
        b_ub = self.x0 + self.bounds
        b_lb = -self.x0 + self.bounds
        
        b = np.concatenate((b, b_ub, b_lb))
        
        return A, b

    def solve(self):
        # Builds the LP
        A, b = self.get_LP_constraints()
        
        c = np.zeros(self.dim + 1)
        c[-1] = 1
        
        c = matrix(c)
        A = matrix(A)
        b = matrix(b)

        try:
            t0 = time.time()
            sol = solvers.lp(c, A, b, solver=self.LP_solver)
            self.times["LP"].append(time.time() - t0)
        except Exception as e:
            print(e)
            raise SolverFailedException("LP failed")
        
        return np.array(sol['x'][:-1]).flatten(), sol['x'][-1]

    def project_point(self, x0, level):
        # Builds the QP 
        Q = 2*matrix(np.eye(self.dim))
        p = matrix(np.multiply(x0, -2))
        
        if False:
            A_ub_x, _, b_ub, lb, ub, x0 = self.get_constraints(level=level)

            A = matrix(A_ub_x)# Credo di dover trasporre
            b = matrix(b_ub)
        else:
            A, b = self.get_QP_constraints(level)
            A = matrix(A)
            b = matrix(b)

        try:
            t0 = time.time()
            sol = solvers.qp(Q, p, A, b, None, None, solver=self.QP_solver)
            self.times["QP"].append(time.time() - t0)
        except Exception as e:
            print(e)
            raise SolverFailedException("QP failed")
        #print(sol['status'])
        return np.array(sol['x']).flatten()

    # The following cvxpy methods are unused because cvxopt is faster
    def get_constraints_cvxpy(self, level=None):
        A_ub_x = np.asarray([[c[i] for i in range(1, len(c))] for c in self.coefficients])
        A_ub_y = np.asarray([-1 for i in range(self.coefficients.shape[0])])

        if level is None:
            b_ub = np.asarray([-c[0] for c in self.coefficients])
        else:
            b_ub = np.asarray([-c[0] + level for c in self.coefficients])

        return A_ub_x, A_ub_y, b_ub, -self.bounds, self.bounds, self.x0

    def solve_cvxpy(self, verbose=False):
        # Builds the LP
        A_ub_x, A_ub_y, b_ub, lb, ub, x0 = self.get_constraints_cvxpy()

        x = Variable(self.dim)
        y = Variable()
        constraints = []
        constraints.append(A_ub_x @ x + A_ub_y * y <= b_ub)
        # Note that bound constraints are 2*self.dim inequalities
        constraints.append(lb <= x - x0)
        constraints.append(x - x0 <= ub)
        objective = Minimize(y)
        problem = Problem(objective, constraints)
            
        try:
            t0 = time.time()
            problem.solve(verbose=verbose, solver=self.LP_solver)
            self.times["LP"].append(time.time() - t0)
        except:
            raise SolverFailedException("Bounded LP failed")
        
        return x.value, y.value

    def project_point_cvxpy(self, x0, level, verbose=False):
        # Builds the QP 
        P = np.eye(self.dim)
        q = np.multiply(x0, -2)

        x = cvxpy.Variable(self.dim)

        A_ub_x, _, b_ub, lb, ub, x0 = self.get_constraints_cvxpy(level=level)

        objective = cvxpy.quad_form(x, P) + q.T @ x
        constraints = []
        constraints.append(A_ub_x @ x <= b_ub)
        constraints.append(lb <= x - x0)
        constraints.append(x - x0 <= ub)

        prob = cvxpy.Problem(cvxpy.Minimize(objective), constraints)
            
        try:
            t0 = time.time()
            prob.solve(verbose=verbose, solver=self.QP_solver)
            self.times["QP"].append(time.time() - t0)
        except Exception as e:
            print(e)
            raise SolverFailedException("QP failed")

        return x.value
