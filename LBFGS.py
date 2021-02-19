import numpy as np
import ackley
import matplotlib.pyplot as plt
from collections import deque

def LBFGS (f, x, M = 20, delta = 1, eps = 1e-10, max_feval = 1000, m1 = 0.0001, m2 = 0.9, tau = 0.9, sfgrd = 0.01, m_inf = - np.Inf, mina = 1e-16):
    
    # setup plotting
    Plotf = True  # if f and the trajectory have to be plotted when n = 2
    fig, ax = plt.subplots()
    ax.scatter(x[0],x[1],color = 'r', marker = '.')
    ackley.plot_general(fig)

    # reading and checking input - - - - - - - - - - - - - - - - - - - - -
    if not check_input(f, x, delta, eps, max_feval, m1, m2, tau, sfgrd, m_inf, mina):
        return 

    n = len(x)

    print("\nL-BFGS method starts")

    # initializations - - - - - - - - - - - - - - - - - - - - - - - -
    
    new_x = np.zeros(n)  # last point visited in the line search
    new_g = np.zeros(n) # gradient of new_x
    saved_s = deque([])
    saved_y = deque([])
    saved_rho = deque([])

    feval = 1
    print('\nfeval\t\tx\tf(x)\t\t|| g(x) ||\tls\talpha*\t rho\n')
    
    v = ackley.function(x)
    g = ackley.gradient(x)
    ng = np.linalg.norm(g)
    ng0 = 1     # un-scaled stopping criterion
    if eps < 0:
        ng0 = - ng  # norm of first subgradient: why is there a "-"? -)
    
    h_0 = np.repeat(delta, n)
    d = - h_0 * g

    # main loop - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    iteration = 0

    while True:

        # output stats
        print(f"{feval}\t{x[0]:4.2f} ; {x[1]:4.2f}\t{v:6.4f}\t\t{ng:6.6f}", end='')

        #stopping criteria - - - - - - - - - - - - - - - - - - - - - - - - - -
        if ng <= eps * ng0:
            status = 'optimal'
            break
        if feval > max_feval:
            status = 'stopped for reached max_feval'
            break

        #black magic by Nocedal (1980) - - - - - - - - - - - - - - - - - - - - -

        if iteration > M:
            bound = M
        else:
            bound = iteration

        saved_alpha = deque([])
        q = g

        for i in range (bound-1, -1, -1):
            new_alpha = saved_rho[i] * np.dot( saved_s[i], q)
            q -= new_alpha * saved_y[i]
            saved_alpha.appendleft(new_alpha)

        if iteration > 0:
            k = np.dot(saved_s[-1], saved_y[-1]) / np.dot(saved_y[-1], saved_y[-1])
        else:
            k = 1

        new_d = k * h_0 * q

        for i in range (bound):
            beta = saved_rho[i] * np.dot(saved_y[i], new_d)
            new_d += (saved_alpha[i] - beta) * saved_s[i]

        d = - new_d

        # compute step size - - - - - - - - - - - - - - - - - - - - - - - - - -
        # as in Newton's method, the default initial stepsize is 1

        phip0 = np.dot(g,d)

        alpha, v, feval, new_x, new_g = ArmijoWolfeLS( f, x, d, feval, v, phip0 , 1 , m1 , m2 , tau, sfgrd, max_feval, mina)
    
        # output statistics - - - - - - - - - - - - - - - - - - - - - - - - - -

        print( '\t{:6.4f}'.format(alpha), end='')
        if alpha <= mina:
           status = 'alpha <= mina ----> A-W not satisfied by any point far enough from x'
           break
    
        if v <= m_inf:
            status = 'unbounded'
            break
        
        # - - compute and store s,y and rho - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        
        s = np.subtract(new_x, x)   # s^i = x^{i + 1} - x^i
        y = np.subtract(new_g, g)   # y^i = \nabla f( x^{i + 1} ) - \nabla f( x^i )
        rho = np.dot(y, s)

        if rho < 1e-16:
            print( '\n\nError: y^i s^i = {:6.4f}'.format(rho))
            status = 'Rho < threshold'
            break
        rho = 1 / rho
        print( "\t{:3.2f}".format(rho) )
        
        saved_s.append(s)
        saved_y.append(y)
        saved_rho.append(rho)

        if iteration > M:
            saved_s.popleft()
            saved_y.popleft()
            saved_rho.popleft()

        # - - possibly plot the trajectory - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        if n == 2 and Plotf:
            ax.scatter(new_x[0],new_x[1],color = 'r', marker = '.')
            ax.quiver(x[0], x[1], new_x[0]-x[0], new_x[1]-x[1], scale_units = 'xy', angles = 'xy', scale = 1, color = 'r', alpha = .3)
    
        # - - update and iterate - - - - - - - - - - - - - - - - - - - - - - - - - - -
        x = new_x
        g = new_g
        ng = np.linalg.norm( g )
        iteration += 1

    print("\n\nSTATUS:\t", status)
    print("\n Last x was:\t", new_x)
    print("\n Last gradient was:\t", new_g)
    plt.show()

def f2phi (f, alpha, x, d, feval):

    # phi( alpha ) = f( x + alpha * d )
    # phi'( alpha ) = < \nabla f( x + alpha * d ) , d >

   new_x = x + alpha * d
   phi = f.function( new_x )
   new_g = f.gradient( new_x )
   phip = np.dot(d, new_g)
   feval = feval + 1
   return phi, phip, new_x, new_g, feval

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def ArmijoWolfeLS(f, x, d, feval, phi0 , phip0 , alpha_sup , m1 , m2 , tau, sfgrd, max_feval, mina):

    lsiter = 1  # count iterations of first phase

    while feval <= max_feval:
        phia, phips, new_x, new_g, feval = f2phi(f, alpha_sup, x, d, feval)
        if ( phia <= phi0 + m1 * alpha_sup * phip0 ) and ( abs( phips ) <= - m2 * phip0 ):
            print( '\t(A)' , lsiter , end='')
            alpha = alpha_sup
            return alpha, phia, feval, new_x, new_g  # Armijo + strong Wolfe satisfied, we are done
        if phips >= 0:  # derivative is positive, break
            break
        alpha_sup = alpha_sup / tau
        lsiter += 1

    lsiter = 1  # count iterations of second phase

    alpha_inf = 0
    alpha = alpha_sup
    phipm = phip0

    while ( feval <= max_feval ) and ( ( alpha_sup - alpha_inf ) ) > mina and ( phips > 1e-12 ):

        # compute the new value by safeguarded quadratic interpolation
        alpha = ( alpha_inf * phips - alpha_sup * phipm ) / ( phips - phipm )
        alpha = max( alpha_inf + ( alpha_sup - alpha_inf ) * sfgrd, min( alpha_sup - ( alpha_sup - alpha_inf ) * sfgrd, alpha ) )

        # compute phi( alpha )
        phia, phip, new_x, new_g, feval = f2phi(f, alpha, x, d, feval)

        if ( phia <= phi0 + m1 * alpha * phip0 ) and ( abs( phip ) <= - m2 * phip0 ):
            break  # Armijo + strong Wolfe satisfied, we are done
        
        # restrict the interval based on sign of the derivative in x + alpha * d
        if phip < 0:
            alpha_inf = alpha
            phipm = phip
        else:
            alpha_sup = alpha
            if alpha_sup <= mina:
                break
            phips = phip
        lsiter = lsiter + 1
        
    print('\t(B)' , lsiter, end='' )

    return alpha, phia, feval, new_x, new_g

def check_input(f, x, delta, eps, max_feval, m1, m2, tau, sfgrd, m_inf, mina):

    if not callable(f.function):
        print("Error: f not a function")
        return False

    try:
        x = np.asarray(x, dtype=np.float)
        delta = np.float(delta)
        eps = np.float(eps)
        max_feval = np.float(max_feval)
        m1 = np.float(m1)
        m2 = np.float(m2)
        tau = np.float(tau)
        sfgrd = np.float(sfgrd)
        if m_inf != - np.Inf:
            m_inf = np.float(m_inf)
        mina = np.float(mina)
    except:
        print("Error: double check arguments. they must be real numbers")
        return False

    if delta < 1e-10:
        print("delta must be > 0")
        return False

    if m1 <= 0 or m1 >= 1:
        print("m1 is not in (0, 1)")
        return False

    if tau <= 0 or tau >= 1:
        print("tau is not in (0, 1)")
        return False

    if sfgrd <= 0 or sfgrd >= 1:
        print("sfgrd is not in (0, 1)")
        return False

    if mina < 0:
        print("mina is < 0")
        return False

    return True

if __name__ == "__main__":
    LBFGS(ackley,[0.3,0])