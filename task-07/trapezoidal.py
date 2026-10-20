import numpy as np
from tools import calculate_quadrature


def trapezoidal(F: callable, a: float, b: float, n: int = 50) -> np.float64:
    nodes = np.linspace(start=a, stop=b, num=n + 1)
    weights = np.full(n + 1, (b - a) / n)
    weights[0] /= 2
    weights[-1] /= 2

    return calculate_quadrature(F, nodes, weights)


def trapezoidal_simplified(f: callable, a: float, b: float, n: int) -> np.float64:
    nodes = np.array([a, b])
    weights = np.array([(b - a) / 2, (b - a) / 2])
    return calculate_quadrature(f, nodes, weights)
