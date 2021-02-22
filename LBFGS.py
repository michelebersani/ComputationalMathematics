import numpy as np
import ackley
import simple_f
import matplotlib.pyplot as plt
from collections import deque
from checks import check_input

class NocedalAlgorithm:
    def __init__(self, M: int, B_0):
        # memory
        self.M = M
        # initial approximation of B (diagonal)
        self.B_0 = B_0
        self.saved_s = deque([])
        self.saved_y = deque([])
        self.saved_rho = deque([])

    def inverse_H_product(self, q: np.ndarray) -> np.ndarray:
        """Returns the product `B*q`, where `B` is the
        current approximation of the inverse of the Hessian.

        Args:
            q (numpy.ndarray): input vector
        """
        current_memory = len(self.saved_rho)
        if current_memory == 0:
            return self.B_0 * q

        saved_alpha = deque([])

        for i in reversed(range(current_memory)):
            new_alpha = self.saved_rho[i] * np.dot(self.saved_s[i], q)
            saved_alpha.append(new_alpha)
            q = q - new_alpha * self.saved_y[i]

        k = np.dot(self.saved_s[-1], self.saved_y[-1])
        k = k / np.dot(self.saved_y[-1], self.saved_y[-1])

        new_d = k * self.B_0 * q

        for i in range(current_memory):
            beta = self.saved_rho[i] * np.dot(self.saved_y[i], new_d)
            new_d += (saved_alpha.pop() - beta) * self.saved_s[i]
        return new_d
    
    def save(self, new_s, new_y, new_rho):
        self.saved_s.append(new_s)
        self.saved_y.append(new_y)
        self.saved_rho.append(new_rho)

        current_memory = len(self.saved_rho)
        if current_memory > self.M:
            self.saved_s.popleft()
            self.saved_y.popleft()
            self.saved_rho.popleft()
    
    def damp_y(self, s:np.ndarray, y:np.ndarray)->np.ndarray:
        """Returns a damped version of `y`, given `s` and `y`.
        See Nocedal's Damped L-BGFS for further details.
        """
        B_s = self.inverse_H_product(s)
        s_B_s = np.dot(s,B_s)
        s_y = np.dot(s,y)
        
        if s_y >= 0.2*s_B_s:
            theta = 1
        else:
            theta = 0.8*s_B_s / (s_B_s - s_y)
        
        y = theta*y + (1-theta)*B_s
        return y


def LBFGS(
    f,
    x,
    M=20,
    delta=1,
    eps=1e-10,
    max_feval=1000,
    m1=0.0001,
    m2=0.9,
    tau=0.9,
    sfgrd=0.01,
    mina=1e-16,
):

    # setup plotting
    Plotf = True  # if f and the trajectory have to be plotted when n = 2
    fig, ax = plt.subplots()
    ax.scatter(x[0], x[1], color="r", marker=".")
    f.plot_general(fig)

    # checking input - - - - - - - - - - - - - - - - - - - - -
    check_input(f, x, delta, eps, max_feval, m1, m2, tau, sfgrd, mina)
    n = len(x)
    print("\nL-BFGS method starts")

    # initializations - - - - - - - - - - - - - - - - - - - - - - - -
    feval = 1
    print("\nfeval\t\tx\tf(x)\t\t|| g(x) ||\tls\talpha*\t y*s\n")

    v = f.function(x)
    g = f.gradient(x)
    ng = np.linalg.norm(g)
    ng0 = 1  # un-scaled stopping criterion
    if eps < 0:
        ng0 = -ng  # norm of first subgradient: why is there a "-"? -)

    B_0 = np.repeat(delta, n)
    nocedal = NocedalAlgorithm(M, B_0)

    # main loop - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    iteration = 0

    while True:
        # output stats
        print(f"{feval}\t{x[0]:4.2f} ; {x[1]:4.2f}\t{v:6.4f}\t\t{ng:6.6f}", end="")

        # stopping criteria - - - - - - - - - - - - - - - - - - - - - - - - - -
        if ng <= eps * ng0:
            status = "optimal"
            break
        if feval > max_feval:
            status = "stopped for reached max_feval"
            break

        # determine new descent direction - - - - - - - - - - - - - - - - - - -
        d = -nocedal.inverse_H_product(g)
        # compute step size - - - - - - - - - - - - - - - - - - - - - - - - - -
        # as in Newton's method, the default initial stepsize is 1

        phip0 = np.dot(g, d)
        if phip0 > 0:
            print(f"\n\nphip0 = {phip0}")
            status = "phip0 > 0"
            break
        alpha, v, feval, new_x, new_g = ArmijoWolfeLS(
            f, x, d, feval, v, phip0, 1, m1, m2, tau, sfgrd, max_feval, mina
        )
        # output statistics - - - - - - - - - - - - - - - - - - - - - - - - - -

        print(f"\t{alpha:6.4f}", end="")

        # - - plot new point - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        if n == 2 and Plotf:
            ax.scatter(new_x[0], new_x[1], color="r", marker=".")
            ax.quiver(
                x[0],
                x[1],
                new_x[0] - x[0],
                new_x[1] - x[1],
                scale_units="xy",
                angles="xy",
                scale=1,
                color="r",
                alpha=0.3,
            )
        # - - compute and store s,y and rho - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        s = new_x - x  # s^i = x^{i + 1} - x^i
        y = new_g - g  # y^i = \nabla f( x^{i + 1} ) - \nabla f( x^i )
        # y = nocedal.damp_y(s,y) # comment for non-damped L-BFGS
        inv_rho = np.dot(y, s)
        if inv_rho < 1e-16:
            print("\n\nError: y^i s^i = {inv_rho:6.4f}")
            status = "Rho < threshold"
            break
        print(f"\t{inv_rho:2.2E}")
        rho = 1 / inv_rho
        
        nocedal.save(s,y,rho)

        # - - update and iterate - - - - - - - - - - - - - - - - - - - - - - - - - - -
        x = new_x
        g = new_g

        ng = np.linalg.norm(g)
        iteration += 1

    print("\n\nEXITED WITH STATUS:\t", status)
    print(f"Last x was: {new_x} \t Last gradient was: {new_g}")
    plt.show()


def f2phi(f, alpha, x, d, feval):

    # phi( alpha ) = f( x + alpha * d )
    # phi'( alpha ) = < \nabla f( x + alpha * d ) , d >

    new_x = x + alpha * d
    phi = f.function(new_x)
    new_g = f.gradient(new_x)
    phip = np.dot(d, new_g)
    feval = feval + 1
    return phi, phip, new_x, new_g, feval


def ArmijoWolfeLS(
    f, x, d, feval, phi0, phip0, alpha_0, m1, m2, tau, sfgrd, max_feval, mina
):

    lsiter = 1  # count iterations of first phase
    alpha = alpha_0

    while feval <= max_feval:
        phia, phip_sup, new_x, new_g, feval = f2phi(f, alpha, x, d, feval)
        if (phia <= phi0 + m1 * alpha * phip0) and (abs(phip_sup) <= -m2 * phip0):
            print("\t(A)", lsiter, end="")
            return (
                alpha,
                phia,
                feval,
                new_x,
                new_g,
            )  # Armijo + strong Wolfe satisfied, we are done
        if phip_sup >= 0:  # derivative is positive, break
            break
        alpha = alpha / tau
        lsiter += 1

    lsiter = 1  # count iterations of second phase
    alpha = alpha_0

    while feval <= max_feval:
        phia, phip_sup, new_x, new_g, feval = f2phi(f, alpha, x, d, feval)
        if (phia <= phi0 + m1 * alpha * phip0) and (abs(phip_sup) <= -m2 * phip0):
            print("\t(A)", lsiter, end="")
            return (
                alpha,
                phia,
                feval,
                new_x,
                new_g,
            )  # Armijo + strong Wolfe satisfied, we are done
        alpha = alpha * tau
        if alpha < mina:
            break
        lsiter += 1

    print("WE STILL HAVE TO HANDLE NO POINT SATISFYING A-W")





if __name__ == "__main__":
    LBFGS(ackley, [4, 1])
