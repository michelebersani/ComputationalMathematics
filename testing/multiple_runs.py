from scipy.optimize import minimize
import time
import numpy as np

def multi_run(solver, f, n=10, type=None):
    """Run the solver `n` times, and return a list containing:
    - final f value (mean)
    - final f value (std)
    - f evals (mean)
    - f evals (std)
    - seconds (mean)
    - seconds (std)
    - number of failures

    Argument `type` must be None if you use a local algorithm, otherwise 'scipy'
    """
    if type is not None:
        return _multi_run_scipy(solver, f, n)
    else:
        return _multi_run_local(solver, f, n)


def _multi_run_scipy(solver, f, n=10):
    final_fv = []
    f_evals = []
    seconds = []
    n_failures = 0

    for _ in range(n):
        # set starting x
        f.model.init_weights()
        x = f.model.Weights
        start_time = time.process_time()
        solver_out = solver(f, x)
        delta_s = time.process_time() - start_time
        seconds.append(delta_s)

        if not solver_out.success:
            n_failures += 1
        else:
            final_fv.append(solver_out.fun)
            f_evals.append(solver_out.nfev)

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


def _multi_run_local(solver, f, n=10):
    final_fv = []
    f_evals = []
    seconds = []
    n_failures = 0

    for _ in range(n):
        # set starting x
        f.model.init_weights()
        x = f.model.Weights
        start_time = time.process_time()
        solver_out = solver.solve(f, x)
        delta_s = time.process_time() - start_time
        seconds.append(delta_s)

        # if solver_out != "optimal":
        #     n_failures += 1
        # else:
        final_fv.append(solver.f_value)
        f_evals.append(solver.feval)


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