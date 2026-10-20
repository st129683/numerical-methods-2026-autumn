import tools

ORDER = 4


def step(f, x, y, h):
    # Классический четырехэтапный метод Рунге-Кутты четвертого порядка,
    # формула вида:
    # y(x + h) = y(x) + (k1 + 2*k2 + 2*k3 + k4) / 6
    k1 = h * f(x, y)
    k2 = h * f(x + 0.5 * h, y + 0.5 * k1)
    k3 = h * f(x + 0.5 * h, y + 0.5 * k2)
    k4 = h * f(x + h, y + k3)

    return y + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def solve(f=None, tol=None):
    if f is None:
        f = tools.Counter(tools.rhs)

    if tol is None:
        tol = tools.EPS_GLOBAL

    h0 = tools.initial_step(
        f,
        tools.X0,
        tools.X_END,
        tools.Y0,
        ORDER,
        tol
    )

    return tools.solve_constant(
        f,
        step,
        ORDER,
        tol,
        h0=h0
    )


if __name__ == '__main__':
    result = solve()
    print('h =', result['h'])
    print('steps =', result['n_steps'])
    print('rhs calls =', result['rhs_calls'])
