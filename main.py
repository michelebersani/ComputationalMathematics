from LBFGS import LBFGS_2D, LBFGS
import testing_functions.ackley
import logging
import numpy as np

if __name__ == '__main__':
    # set level to WARNING to avoid printing INFOs
    logging.basicConfig(level='INFO')

    solver = LBFGS()
    status = solver.solve(testing_functions.ackley, [2,3.1])
    print('')
    print(f'Exited with status: {status}')
    print(f"Minimum function value found is: {solver.f_value}")
    print(f"f evaluations: {solver.feval}")
    print(f"Last gradient was: {solver.g}")
    print('')

    LBFGS_2D(testing_functions.ackley, [2,3.1])