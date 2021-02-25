from LBFGS import LBFGS_2D, LBFGS
import ackley
import logging

if __name__ == '__main__':
    # set level to WARNING to avoid printing INFOs
    logging.basicConfig(level='INFO')

    solver = LBFGS()
    status = solver.solve(ackley, [2,3])
    print(f'Exited with status: {status}')
    print("Minimum function value found is:")
    print(solver.f_value)
    print(f"f evaluations: {solver.feval}")

    LBFGS_2D(ackley, [2,3])