import numpy as np

from tools import f, p, F, estimate_convergence_rate, richardson_extrapolation
from left_rectangle import left_rectangle
from middle_rectangle import middle_rectangle
from trapezoidal import trapezoidal
from simpson import simpson
from newton_cotes import newton_cotes
from gauss_quadrature import gauss_quadrature
from plot_tools import plot_errors


def compute_with_richardson(method_func, F_func, a, b, n, L=2.0):
    S1 = method_func(F_func, a, b, n)
    S2 = method_func(F_func, a, b, int(n * L))
    S3 = method_func(F_func, a, b, int(n * L**2))

    m = estimate_convergence_rate(S1, S2, S3, L)
    J_rich, error = richardson_extrapolation(S1, S2, S3, m, L)

    return J_rich, error, m


def main():
    a, b = 1, 3
    n_values_p1 = [26, 50, 100, 200, 500, 1000, 2000]
    n_values_p2 = [3, 6, 12, 25, 50]

    print('=' * 80)
    print('part 1.1: p(x) ≡ 1')
    print('=' * 80)
    
    methods_p1 = {
        'left_rect': left_rectangle,
        'middle_rect': middle_rectangle,
        'trapezoidal': trapezoidal,
        'simpson': simpson
    }

    errors_p1 = {name: [] for name in methods_p1}

    print(f"{'n':>6} | {'left_rect':>12} | {'middle_rect':>12} | {'trapezoidal':>12} | {'simpson':>12}")
    print('-' * 80)

    for n in n_values_p1:
        row = [f'{n:6}']
        for name, func in methods_p1.items():
            try:
                J_rich, error, m = compute_with_richardson(func, f, a, b, n)
                errors_p1[name].append(error)
                row.append(f'{error:12.6e}')
            except Exception as e:
                errors_p1[name].append(np.nan)
                row.append(f'{"error":>12}')
        print(' | '.join(row))

    for name in errors_p1:
        errors_p1[name] = np.array(errors_p1[name])

    print('\n' + '=' * 80)
    print('part 1.2: p(x) = (b-x)^{{-1/6}}')
    print('=' * 80)

    methods_p2 = {
        'newton_cotes': newton_cotes,
        'gauss': gauss_quadrature
    }

    errors_p2 = {name: [] for name in methods_p2}

    print(f"{'n':>6} | {'newton_cotes':>12} | {'gauss':>12}")
    print('-' * 80)

    for n in n_values_p2:
        row = [f'{n:6}']
        for name, func in methods_p2.items():
            try:
                J_rich, error, m = compute_with_richardson(func, F, a, b, n)
                errors_p2[name].append(error)
                row.append(f'{error:12.6e}')
            except Exception as e:
                errors_p2[name].append(np.nan)
                row.append(f'{"error":>12}')
        print(' | '.join(row))

    for name in errors_p2:
        errors_p2[name] = np.array(errors_p2[name])

    plot_errors(n_values_p1, errors_p1, title='convergence for p(x) = 1')
    plot_errors(n_values_p2, errors_p2, title='convergence for p(x) = (b-x)^{-1/6}')


if __name__ == '__main__':
    main()
