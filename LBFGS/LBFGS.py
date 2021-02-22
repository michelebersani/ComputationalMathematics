import numpy as np
import matplotlib.pyplot as plt
from .Nocedal import NocedalAlgorithm
from .checks import check_input
from .LineSearch import ArmijoWolfeLS


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
    mina=1e-16,
):

    # setup plotting
    Plotf = True  # if f and the trajectory have to be plotted when n = 2
    fig, ax = plt.subplots()
    ax.scatter(x[0], x[1], color="r", marker=".")
    f.plot_general(fig)

    # checking input - - - - - - - - - - - - - - - - - - - - -
    check_input(f, x, delta, eps, max_feval, m1, m2, tau, mina)
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
            f, x, d, feval, v, phip0, 1, m1, m2, tau, max_feval, mina
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

        nocedal.save(s, y, rho)

        # - - update and iterate - - - - - - - - - - - - - - - - - - - - - - - - - - -
        x = new_x
        g = new_g

        ng = np.linalg.norm(g)
        iteration += 1

    print("\n\nEXITED WITH STATUS:\t", status)
    print(f"Last x was: {new_x} \t Last gradient was: {new_g}")
    plt.show()