import numpy as np


def check_input(f, x, delta, eps, max_feval, m1, m2, tau, mina):
    assert callable(f.function), "f is not a function"
    assert delta > 1e-10, "delta must be > 0"
    assert 0 <= m1 <= 1, "m1 must be in (0, 1)"
    assert 0 <= tau <= 1, "tau must be in (0, 1)"
    # assert 0 <= sfgrd <= 1, "sfgrd must be in (0, 1)"
    assert mina > 0, "mina must be > 0"
