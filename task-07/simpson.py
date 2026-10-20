import numpy as np
from tools import calculate_quadrature


def simpson(F: callable, a: float, b: float, n: int) -> np.float64:
    if n % 2 != 0:
        raise ValueError('again? n must be even for simpson rule')

    nodes = np.linspace(start=a, stop=b, num=n + 1)
    h = (b - a) / n
    weights = np.full(n + 1, h / 3)
    weights[1:-1:2] *= 4
    weights[2:-1:2] *= 2

    return calculate_quadrature(F, nodes, weights)
