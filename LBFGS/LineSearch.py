import numpy as np


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
    f, x, d, feval, phi0, phip0, alpha_0, m1, m2, tau, max_feval, mina
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
            print("\t(B)", lsiter, end="")
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