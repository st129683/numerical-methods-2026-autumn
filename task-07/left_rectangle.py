import numpy as np
from tools import calculate_quadrature


def left_rectangle(F: callable, a: float, b: float, n: int = 50) -> np.float64:
    nodes = np.linspace(start=a, stop=b, num=n + 1)[:-1]
    weights = np.full(n, (b - a) / n)

    return calculate_quadrature(F, nodes, weights)


def left_rectangle_simplified(f: callable, a: float, b: float) -> np.float64:
    nodes = np.array([a, a])
    weights = np.array([b, -a])

    return calculate_quadrature(f, nodes, weights)
