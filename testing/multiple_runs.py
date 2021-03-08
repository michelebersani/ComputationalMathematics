from scipy.optimize import minimize
import time
import numpy as np


def multi_run(solver, f, x, type='scipy', n=10):
    """Run the solver `n` times, and return a list containing:
    - final f value (mean)
    - final f value (std)
    - f evals (mean)
    - f evals (std)
    - seconds (mean)
    - seconds (std)
    - number of failures

    Argument `type` must be one of the following: 'scipy', 'LBFGS', 'LevelMethod'
    """
    final_fv = []
    f_evals = []
    seconds = []
    n_failures = 0

    for _ in range(n):
        start_time = time.process_time()
        solver_out = solver(f, x)
        delta_s = time.process_time() - start_time
        seconds.append(delta_s)

        if type == 'scipy':
            if not solver_out.success:
                n_failures += 1
            else:
                final_fv.append(solver_out.fun)
                f_evals.append(solver_out.nfev)
        elif type == 'LBFGS':
            if solver_out != "optimal":
                n_failures += 1
            else:
                final_fv.append(solver.f_value)
                f_evals.append(solver.feval)

        elif type == 'LevelMethod':
            raise NotImplementedError
        else:
            raise Exception('Invalid value for `type` argument')


    final_fv = np.array(final_fv)
    f_evals = np.array(f_evals)
    seconds = np.array(seconds)
    return (
        final_fv.mean(),
        final_fv.std(),
        f_evals.mean(),
        f_evals.std(),
        seconds.mean(),
        seconds.std(),
        n_failures
    )