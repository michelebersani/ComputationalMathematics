from LBFGS import LBFGS_2D, LBFGS
import ackley
import logging
import numpy as np

if __name__ == '__main__':
    # set level to WARNING to avoid printing INFOs
    logging.basicConfig(level='INFO')

    solver = LBFGS()
    status = solver.solve(ackley, [2,3.1])
    print('')
    print(f'Exited with status: {status}')
    print(f"Minimum function value found is: {solver.f_value}")
    print(f"f evaluations: {solver.feval}")
    print(f"Last gradient was: {solver.g}")
    print('')

    #LBFGS_2D(ackley, [2,3])