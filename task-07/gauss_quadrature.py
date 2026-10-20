import numpy as np
from tools import calculate_quadrature, solve_cubic_cardano, compute_moments, f


def gauss_quadrature(F: callable, a: float, b: float, n: int, beta: float = 1/6) -> float:
    edges = np.linspace(start=a, stop=b, num=n + 1)
    total_integral = 0.0

    for i in range(n):
        z_left = edges[i]
        z_right = edges[i + 1]

        # Замена t = b - x (файл 4). Отрезок [z_left, z_right] -> [c, d]
        c = b - z_right
        d = b - z_left

        # 1. Моменты по формуле (1) из файла 4 (через compute_moments)
        moments = compute_moments(z_left, z_right, b, beta, max_j=5)

        # 2. СЛАУ (19) из файла 1 для коэффициентов узлового многочлена
        matrix_A = np.array([
            [moments[0], moments[1], moments[2]],
            [moments[1], moments[2], moments[3]],
            [moments[2], moments[3], moments[4]]
        ])
        rhs = np.array([-moments[3], -moments[4], -moments[5]])

        coeffs = np.linalg.solve(matrix_A, rhs)
        a0, a1, a2 = coeffs[0], coeffs[1], coeffs[2]

        # 3. Корни узлового многочлена (20) через Кардано (файл 5)
        t_nodes = solve_cubic_cardano(1.0, a2, a1, a0)

        # 4. СЛАУ (21) из файла 1 для весов (матрица Вандермонда)
        vandermonde = np.array([
            [1, 1, 1],
            [t_nodes[0], t_nodes[1], t_nodes[2]],
            [t_nodes[0]**2, t_nodes[1]**2, t_nodes[2]**2]
        ])
        weights = np.linalg.solve(vandermonde, moments[:3])

        # 5. Обратная замена x = b - t и суммирование
        x_nodes = b - t_nodes
        total_integral += calculate_quadrature(f, x_nodes, weights)

    return total_integral
