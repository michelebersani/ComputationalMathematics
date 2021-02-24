import numpy as np


def f2phi(f, alpha, x, d):

    # phi( alpha ) = f( x + alpha * d )
    # phi'( alpha ) = < \nabla f( x + alpha * d ) , d >

    new_x = x + alpha * d
    phi = f.function(new_x)
    new_g = f.gradient(new_x)
    phip = np.dot(d, new_g)
    return phi, phip, new_x, new_g


def ArmijoWolfeLS(f, x, d, feval, phi0, phip0, alpha_0, m1, m2, tau, max_feval, mina):

    lsiter1 = 0  # count iterations of first phase
    lsiter2 = 0  # count iterations of second phase
    alpha = alpha_0
    while feval <= max_feval:
        phia, phip_sup, new_x, new_g = f2phi(f, alpha, x, d)
        feval += 1
        lsiter1 += 1
        if (phia <= phi0 + m1 * alpha * phip0) and (abs(phip_sup) <= -m2 * phip0):
            return (
                new_x,
                new_g,
                alpha,
                phia,
                feval,
                (lsiter1, lsiter2),
            )  # Armijo + strong Wolfe satisfied, we are done
        if phip_sup >= 0:  # derivative is positive, break
            break
        alpha = alpha / tau

    alpha = alpha_0 * tau
    while feval <= max_feval:
        phia, phip_sup, new_x, new_g = f2phi(f, alpha, x, d)
        feval = feval + 1
        lsiter2 += 1
        if (phia <= phi0 + m1 * alpha * phip0) and (abs(phip_sup) <= -m2 * phip0):
            return (
                new_x,
                new_g,
                alpha,
                phia,
                feval,
                (lsiter1, lsiter2),
            )  # Armijo + strong Wolfe satisfied, we are done
        alpha = alpha * tau
        if alpha < mina:
            break

    raise Exception("ArmijoWolfeLS could not find a point.")