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
    if type == 'scipy':
        return _multi_run_scipy(solver, f, x, n)
    else:
        return _multi_run_local(solver, f, x, n)


def _multi_run_scipy(solver, f, x, n=10):
    final_fv = []
    f_evals = []
    seconds = []
    n_failures = 0

    for _ in range(n):
        start_time = time.process_time()
        solver_out = solver(f, x)
        delta_s = time.process_time() - start_time

        if not solver_out.success:
            n_failures += 1
        else:
            final_fv.append(solver_out.fun)
            f_evals.append(solver_out.nfev)

    final_fv = np.array(final_fv)
    f_evals = np.array(f_evals)
    seconds = np.array(delta_s)
    return (
        final_fv.mean(),
        final_fv.std(),
        f_evals.mean(),
        f_evals.std(),
        seconds.mean(),
        seconds.std(),
        n_failures
    )


def _multi_run_local(solver, f, x, n=10):
    final_fv = []
    f_evals = []
    seconds = []
    n_failures = 0

    for _ in range(n):
        start_time = time.process_time()
        solver_out = solver(f, x)
        delta_s = time.process_time() - start_time

        if solver_out != "optimal":
            n_failures += 1
        else:
            final_fv.append(solver.f_value)
            f_evals.append(solver.feval)
            seconds.append(delta_s)


    final_fv = np.array(final_fv)
    f_evals = np.array(f_evals)
    seconds = np.array(delta_s)
    return (
        final_fv.mean(),
        final_fv.std(),
        f_evals.mean(),
        f_evals.std(),
        seconds.mean(),
        seconds.std(),
        n_failures
    )