import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import linprog
from cvxpy import *

class CuttingPlaneModel:
    def __init__(self, dim, bounds):
        self.dim = dim
        self.bounds = bounds
        self.coefficients = np.empty((0,dim+1))

    def __call__(self, x):#REMOVE
        y = [np.sum(np.multiply(coefficients_i, np.hstack((1,x)))) for coefficients_i in self.coefficients]
        return np.max(y), 0

    def get_constraints(self):
        A_ub_x = np.asarray([[c[1],c[2]] for c in self.coefficients])
        A_ub_y = np.asarray([-1 for i in range(self.coefficients.shape[0])])
        b_ub = np.asarray([-c[0] for c in self.coefficients])
        return A_ub_x, A_ub_y, b_ub

    def add_plane(self, f, g, x):
        c = f - np.sum(np.multiply(g,x))
        new_plane = np.append(c,g)
        self.coefficients = np.append(self.coefficients, [new_plane], axis=0)

    def solve(self, lb, ub):
        A_ub_x, A_ub_y, b_ub = self.get_constraints()

        x = Variable(self.dim)
        y = Variable() #TODO: yp yn se non funziona per min negativo
        constraints = []
        constraints.append(A_ub_x @ x + A_ub_y * y <= b_ub)
        objective = Minimize(y)
        problem = Problem(objective, constraints)
        problem.solve(verbose=False)

        #print("Problem status: ", problem.status)
        on_border = problem.status in ['unbounded', 'infeasible']# TODO infeasible fix se possibile
        if problem.status == 'infeasible':
            print("Warning: Infeasible problem")

        if on_border:
            lb_constraint = [lb]*self.dim # Rewrite as two variables
            ub_contraint = [ub]*self.dim
            constraints.append(lb_constraint <= x)
            constraints.append(x <= ub_contraint)
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

        A_ub_x, A_ub_y, b_ub = self.get_constraints()

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

    def plot(self, points=[], temp_points=[]):
        plt.figure()

        xaxis = np.linspace(-self.bounds, self.bounds, num=100)
        yaxis = np.linspace(-self.bounds, self.bounds, num=100)

        result = np.zeros((len(xaxis),len(yaxis)))
        for i, x in enumerate(xaxis):
            for j, y in enumerate(yaxis):
                result[j,i], _ = self.__call__([x,y])


        c = plt.contour(xaxis, yaxis, result, 50)
        plt.colorbar()

        for p in temp_points:
            plt.plot(p[0], p[1], 'o', color='red');
        for i, p in enumerate(points):
            plt.plot(p[0], p[1], 'o', color='black');
            plt.text(p[0], p[1], str(i))

        plt.show()

class LevelMethod2d:
    def __init__(self, bounds=10, lambda_=0.29289, epsilon=0.001, max_iter=1000):
        self.bounds = bounds
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.max_iter = 1000

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

    def cache_points(self, xaxis=None, yaxis=None):# Todo, for x of dim n
        if xaxis is None:
            xaxis = np.linspace(-self.bounds, self.bounds, num=100)
        if yaxis is None:
            yaxis = np.linspace(-self.bounds, self.bounds, num=100)

        result = np.zeros((len(xaxis),len(yaxis)))
        for i, x in enumerate(xaxis):
            for j, y in enumerate(yaxis):
                result[j,i], _ = self.function([x,y])

        self.function_points = result

    def plot(self, points=[], temp_points=[]):
            plt.figure()

            xaxis = np.linspace(-self.bounds, self.bounds, num=100)
            yaxis = np.linspace(-self.bounds, self.bounds, num=100)

            if self.function_points is None:
                self.cache_points(xaxis, yaxis)

            c = plt.contour(xaxis, yaxis, self.function_points, 50)
            plt.colorbar()

            for p in temp_points:
                plt.plot(p[0], p[1], 'o', color='red');
            for i, p in enumerate(points):
                plt.plot(p[0], p[1], 'o', color='black');
                plt.text(p[0], p[1], str(i))

            plt.show()

    def solve(self, function, x, verbose=False, plot=False):
        self.function = function
        self.dim = len(x)
        self.x = x

        # Build the cutting plane model
        self.model = CuttingPlaneModel(self.dim, self.bounds)

        plot_points = [x]
        self.f_upstar = math.inf
        self.x_upstar = None
        gap = math.inf
        self.current_iter = 0

        print(f"Iteration\tf*\t\tModel Min\t\tGap\t\tLevel\t         Is on boder?")
        while gap > self.epsilon:
            # Oracle computes f and g
            current_f, current_g = function(self.x)

            # Update model
            self.model.add_plane(current_f, current_g, self.x)

            # Compute f_substar, f_upstar, x_upstar
            self.x_substar, self.f_substar, is_on_border = self.model.solve(-self.bounds,self.bounds)

            if self.f_upstar > current_f:
                self.f_upstar = current_f
                self.x_upstar = self.x

            # Project x onto level set
            gap = self.f_upstar - self.f_substar
            level = self.f_substar + self.lambda_ * gap
            if gap < -0.1:
                print("Warning: Negative gap ", gap)
                break

            if is_on_border: # Project x on the border, as target level is infinite.
                self.x = self.x_substar
            else: # Project x on the target level
                self.x = self.model.project_point(self.x, level, verbose=verbose)

            print(f"{self.current_iter}\t\t{self.f_upstar:.6f}\t{self.f_substar:.6f}\t\t{gap:.6f}\t{level:.6f}      {is_on_border}")

            if plot:
                plot_points.append(self.x)
                self.model.plot(plot_points)
                self.plot(plot_points)

            self.current_iter += 1
            if self.current_iter > self.max_iter:
                print("Warning: Maximum number of iterations reached.")
                break


if __name__ == "__main__":
    from test_function import TestFunction

    f = LevelMethod2d(bounds = 20)
    f.solve(TestFunction(), [-1,-3], plot=False)
