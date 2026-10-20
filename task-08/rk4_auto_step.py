import tools

ORDER = 4


def step(f, x, y, h):
    # Классический четырехэтапный метод Рунге-Кутты четвертого порядка.
    k1 = h * f(x, y)
    k2 = h * f(x + 0.5 * h, y + 0.5 * k1)
    k3 = h * f(x + 0.5 * h, y + 0.5 * k2)
    k4 = h * f(x + h, y + k3)

    return y + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


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