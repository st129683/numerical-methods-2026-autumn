import numpy as np
from tools import compute_moments, f


def newton_cotes(F: callable, a: float, b: float, n: int, beta: float = 1/6) -> float:
    edges = np.linspace(start=a, stop=b, num=n + 1)
    total_integral = 0.0

    for i in range(n):
        z_left = edges[i]
        z_right = edges[i + 1]

        c = b - z_right
        d = b - z_left

        t1 = c
        t2 = (c + d) / 2
        t3 = d

        moments = compute_moments(z_left, z_right, b, beta, max_j=2)
        mu0, mu1, mu2 = moments[0], moments[1], moments[2]

        vandermonde = np.array([
            [1, 1, 1],
            [t1, t2, t3],
            [t1**2, t2**2, t3**2]
        ])
        weights = np.linalg.solve(vandermonde, np.array([mu0, mu1, mu2]))
        A1, A2, A3 = weights[0], weights[1], weights[2]

        x1 = z_right
        x2 = (z_left + z_right) / 2
        x3 = z_left

        total_integral += A1 * f(x1) + A2 * f(x2) + A3 * f(x3)

    return total_integral
