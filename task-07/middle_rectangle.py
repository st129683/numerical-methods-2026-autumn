import numpy as np
from tools import calculate_quadrature


def middle_rectangle(F: callable, a: float, b: float, n: int = 50) -> float:
    edges = np.linspace(start=a, stop=b, num=n + 1)
    nodes = (edges[:-1] + edges[1:]) / 2
    weights = np.full(n, (b - a) / n)

    return calculate_quadrature(F, nodes, weights)


def middle_rectangle_simplified(f: callable, a: float, b: float) -> np.float64:
    nodes = np.array([a + b]) / 2
    weights = np.array([b - a]) / 2

    return calculate_quadrature(f, nodes, weights)
