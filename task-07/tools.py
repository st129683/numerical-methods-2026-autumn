import numpy as np


EXACT_VALUE = -41.8881634400363060689123568290029002745967882134004197398579955054837995836256242598606170450003270671325624121620861724162807011438603568917509254212


def f(x: float | np.ndarray) -> float | np.ndarray:
    return (
        3 * np.cos(7 * x / 2) * np.exp(4 * x / 3)
        + 2 * np.sin(7 * x / 2) * np.exp(-2 * x / 3)
        + 4 * x
    )


def p(x: float) -> float:
    return 1 / np.power(3 - x, 1 / 6)


def F(x: float) -> float:
    return f(x) * p(x)


def calculate_quadrature(F_func: callable, nodes: np.ndarray, weights: np.ndarray) -> float:
    if len(nodes) != len(weights):
        raise ValueError('bruuuv, nodes and weights arrays must have the same size, wha the heeelll')
    return float(np.sum(F_func(nodes) * weights))


def solve_cubic_cardano(a: float, b: float, c: float, d: float) -> np.ndarray:
    q = b**3 / (27 * a**3) - b * c / (6 * a**2) + d / (2 * a)
    p_coeff = (3 * a * c - b**2) / (9 * a**2)

    D = q**2 + p_coeff**3
    if D > 0 or p_coeff >= 0:
        raise ValueError('D > 0 or p >= 0: the polynomial does not have 3 real roots :(')

    r = np.sqrt(np.abs(p_coeff))
    if q < 0:
        r *= -1
    phi = np.arccos(q / (r**3))

    y1 = -2 * r * np.cos(phi / 3)
    y2 = 2 * r * np.cos((np.pi - phi) / 3)
    y3 = 2 * r * np.cos((np.pi + phi) / 3)

    roots = np.array([y1, y2, y3]) - b / (3 * a)
    return np.sort(roots)


def compute_moments(z_left: float, z_right: float, b: float, beta: float, max_j: int) -> np.ndarray:
    moments = np.zeros(max_j + 1)
    for j in range(max_j + 1):
        power = j - beta + 1
        moments[j] = ((b - z_left) ** power - (b - z_right) ** power) / power
    return moments


def estimate_convergence_rate(S1: float, S2: float, S3: float, L: float = 2.0) -> float:
    diff1 = S2 - S1
    diff2 = S3 - S2
    if diff1 == 0 or diff2 / diff1 <= 0:
        return 1.0
    m = -np.log(abs(diff2 / diff1)) / np.log(L)
    return max(m, 1.0)


def richardson_extrapolation(S1: float, S2: float, S3: float, m: float, L: float = 2.0) -> tuple:
    L_m = L**m
    L_m_plus_1 = L**(m + 1)
    J_rich = S3 + (S3 - S2) / (L_m - 1) + (S3 - 2*S2 + S1) / ((L_m - 1) * (L_m_plus_1 - 1))
    error = abs((S3 - S2) / (L_m - 1))

    return J_rich, error
