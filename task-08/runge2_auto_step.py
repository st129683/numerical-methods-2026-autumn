import tools

ORDER = 2

# тот же самый двухэтапный ЯМРК второго порядка,
# но используется алгоритм автоматического выбора шага

C2 = tools.XI
A21 = C2
B2 = 1.0 / (2.0 * C2)
B1 = 1.0 - B2


def step(f, x, y, h):
    k1 = h * f(x, y)
    k2 = h * f(x + C2 * h, y + A21 * k1)
    return y + B1 * k1 + B2 * k2


def solve(f=None, tol=None):
    if f is None:
        f = tools.Counter(tools.rhs)

    if tol is None:
        tol = tools.RHO_LOCAL

    h0 = tools.initial_step(
        f,
        tools.X0,
        tools.X_END,
        tools.Y0,
        ORDER,
        tol
    )

    return tools.solve_auto(
        f,
        step,
        ORDER,
        tol,
        h0=h0
    )


if __name__ == '__main__':
    result = solve()
    print('steps =', result['n_steps'])
    print('rhs calls =', result['rhs_calls'])
