import numpy as np
import simple_f
import matplotlib.pyplot as plt

def BFGS (f, x, delta = 1, eps = 1e-6, max_feval = 1000, m1 = 0.01, m2 = 0.9, tau = 0.9, sfgrd = 0.01, m_inf = - np.Inf, mina = 1e-16):
    
    # setup plotting
    Plotf = True  # if f and the trajectory have to be plotted when n = 2
    fig, ax = plt.subplots()
    ax.scatter(x[0],x[1],color = 'r', marker = '.')
    simple_f.plot_general(fig)

    # reading and checking input - - - - - - - - - - - - - - - - - - - - -
    if not check_input(f, x, delta, eps, max_feval, m1, m2, tau, sfgrd, m_inf, mina):
        return 

    n = len(x)
    awls = m2 > 0 and m2 < 1

    print("BFGS method starts")

    # initializations - - - - - - - - - - - - - - - - - - - - - - - -
    
    last_x = np.zeros(n)  # last point visited in the line search
    last_g = np.zeros(n) # gradient of last_x
    feval = 1
    print('feval\tf(x)\t\t|| g(x) ||\tls\tfev\talpha*\t rho\n\n')
    
    v = simple_f.function(x)
    g = simple_f.gradient(x)
    ng = np.linalg.norm(g)
    ng0 = 1     # un-scaled stopping criterion
    if eps < 0:
        ng0 = - ng  # norm of first subgradient: why is there a "-"? -)
    
    B = init_B(delta, x, n)

    # main loop - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    while True:

        # output stats
        print(feval, "\t{:6.4f}\t\t{:6.6f}".format(v , ng), end='')

        #stopping criteria - - - - - - - - - - - - - - - - - - - - - - - - - -
        if ng <= eps * ng0:
            status = 'optimal'
            break
        if feval > max_feval:
            status = 'stopped'
            break

        #compute approximation to Newton's direction - - - - - - - - - - - - -
        d = - np.dot(B, g)

        # compute step size - - - - - - - - - - - - - - - - - - - - - - - - - -
        # as in Newton's method, the default initial stepsize is 1

        phip0 = np.dot(g,d)

        if awls:
            alpha, v, feval, last_x, last_g = ArmijoWolfeLS( f, x, d, feval, v, phip0 , 1 , m1 , m2 , tau, sfgrd, max_feval, mina)
        else:
            alpha, v, feval, last_x, last_g = BacktrackingLS( f, x, d, feval, v, phip0 , 1 , m1 , m2 , tau, sfgrd, max_feval, mina )

        # output statistics - - - - - - - - - - - - - - - - - - - - - - - - - -

        print( '\t{:6.4f}'.format(alpha), end='')
        if alpha <= mina:
           status = 'error'
           break
    
        if v <= m_inf:
            status = 'unbounded'
            break

        # update approximation of the Hessian - - - - - - - - - - - - - - - - -
        # warning: magic at work! Broyden-Fletcher-Goldfarb-Shanno formula

        s = np.subtract(last_x, x)   # s^i = x^{i + 1} - x^i
        y = np.subtract(last_g, g)   # y^i = \nabla f( x^{i + 1} ) - \nabla f( x^i )
  
        rho = np.dot(y, s)
        if rho < 1e-16:
            print( '\nError: y^i s^i = {:6.4f}'.format(rho))
            status = 'error'
            break
        rho = 1 / rho
        print( "\t{:3.2f}".format(rho) )

        D = np.dot(B, np.outer(y, s))
        B = B + rho * ( ( 1 + rho * np.dot(y, np.dot(B, y) )) * np.outer( s, s ) - D - np.transpose(D))
        # compute new point - - - - - - - - - - - - - - - - - - - - - - - - - -

        # possibly plot the trajectory
        if n == 2 and Plotf:
            ax.scatter(last_x[0],last_x[1],color = 'r', marker = '.')
            ax.quiver(x[0], x[1], last_x[0]-x[0], last_x[1]-x[1], scale_units = 'xy', angles = 'xy', scale = 1, color = 'r', alpha = .3)
    
        # update and iterate - - - - - - - - - - - - - - - - - - - - - - - - - - -
        
        x = last_x
        g = last_g
        ng = np.linalg.norm( g )

    print("\n\nSTATUS:\t", status)
    print("\n Last x was:\t", last_x)
    print("\n Last gradient was:\t", last_g)
    plt.show()

def f2phi (f, alpha, x, d, feval):

    # phi( alpha ) = f( x + alpha * d )
    # phi'( alpha ) = < \nabla f( x + alpha * d ) , d >

   last_x = x + alpha * d
   phi = f.function( last_x )
   last_g = f.gradient( last_x )
   phip = np.dot(d, last_g)
   feval = feval + 1
   return phi, phip, last_x, last_g, feval

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def ArmijoWolfeLS(f, x, d, feval, phi0 , phip0 , alpha_sup , m1 , m2 , tau, sfgrd, max_feval, mina):

    lsiter = 1  # count iterations of first phase
    
    while feval <= max_feval:
        phia, phips, last_x, last_g, feval = f2phi(f, alpha_sup, x, d, feval)
        if ( phia <= phi0 + m1 * alpha_sup * phip0 ) and ( abs( phips ) <= - m2 * phip0 ):
            print( '\t' , lsiter , end='')
            alpha = alpha_sup
            return alpha, phia, feval, last_x, last_g  # Armijo + strong Wolfe satisfied, we are done
        if phips >= 0:  # derivative is positive, break
            break
        alpha_sup = alpha_sup / tau
        lsiter += 1

    print('\t' , lsiter, end='' )
    lsiter = 1  # count iterations of second phase

    alpha_inf = 0
    alpha = alpha_sup
    phipm = phip0

    while ( feval <= max_feval ) and ( ( alpha_sup - alpha_inf ) ) > mina and ( phips > 1e-12 ):

        # compute the new value by safeguarded quadratic interpolation
        alpha = ( alpha_inf * phips - alpha_sup * phipm ) / ( phips - phipm )
        alpha = max( alpha_inf + ( alpha_sup - alpha_inf ) * sfgrd, min( alpha_sup - ( alpha_sup - alpha_inf ) * sfgrd, alpha ) )

        # compute phi( alpha )
        phia, phip, last_x, last_g, feval = f2phi(f, alpha, x, d, feval)

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
        
    print('\t' , lsiter, end='' )

    return alpha, phia, feval, last_x, last_g

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

def init_B(delta, x, n):

    if delta > 0:
        # initial approximation of inverse of Hessian = scaled identity
        B = delta * np.eye( n )
    else:
        # initial approximation of inverse of Hessian computed by finite
        # differences of gradient
        smallsetp = max(- delta , 1e-8 )
        B = np.zeros((n, n))
        for i in range (n):
            xp = x
            xp[i] = xp[i] + smallsetp
            gp = simple_f.gradient( xp )
            B[i] = ( gp - g ) / smallsetp
        B = ( B + np.transpose(B)) / 2  # ensure it is symmetric
        lambdan = eigs( B , 1 , 'sa' )  # smallest eigenvalue
        if lambdan < 1e-6:
            B = B + ( 1e-6 - lambdan ) * np.eye( n )
        B = np.inv( B )
    return B

if __name__ == "__main__":
    BFGS(simple_f,[1,1.33])