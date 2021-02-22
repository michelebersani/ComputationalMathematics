from LBFGS import LBFGS
import ackley

if __name__ == '__main__':
    LBFGS(ackley, [4,3])