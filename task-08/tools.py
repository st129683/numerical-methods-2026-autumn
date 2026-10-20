import numpy as np

XI = 1.0 / 17.0
A = 1.0 / 35.0
B = 1.0 / 10.0

X0 = 0.0
X_END = float(np.pi)
Y0 = np.array([B * np.pi, A * np.pi], dtype=float)

EPS_GLOBAL = 1.0e-4
RHO_LOCAL = 1.0e-5

OMEGA = float(np.sqrt(A * B))


def rhs(x, y):
    return np.array([A * y[1], -B * y[0]], dtype=float)


class Counter:
    def __init__(self, func):
        self.func = func
        self.calls = 0

    def __call__(self, x, y):
        self.calls += 1
        return self.func(x, y)

    def reset(self):
        self.calls = 0


def norm_inf(v):
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return 0.0
    return float(np.max(np.abs(v)))


def max_norm_rows(mat):
    mat = np.asarray(mat, dtype=float)
    if mat.size == 0:
        return 0.0
    if mat.ndim == 1:
        return norm_inf(mat)
    return float(np.max(np.max(np.abs(mat), axis=1)))


def exact_solution(x):
    x_arr = np.asarray(x, dtype=float)
    c = np.cos(OMEGA * x_arr)
    s = np.sin(OMEGA * x_arr)

    y1 = B * np.pi * c + (A * A * np.pi / OMEGA) * s
    y2 = A * np.pi * c - (B * OMEGA / A) * np.pi * s

    if x_arr.ndim == 0:
        return np.array([float(y1), float(y2)], dtype=float)

    return np.vstack((y1, y2)).T


def exact_flow(x, y, h):
    c = np.cos(OMEGA * h)
    s = np.sin(OMEGA * h)
    y = np.asarray(y, dtype=float)

    return np.array([
        c * y[0] + (A / OMEGA) * s * y[1],
        c * y[1] - (B / OMEGA) * s * y[0]
    ], dtype=float)


def _initial_step_from_state(x_state, x_end, y_state, f_state, order, tol):
    scale_x = max(abs(x_state), abs(x_end), 1.0)
    term_x = (1.0 / scale_x) ** (order + 1)
    term_f = norm_inf(f_state) ** (order + 1)
    denom = term_x + term_f
    interval = abs(x_end - x_state)

    if denom <= 0.0:
        return interval

    h = (tol / denom) ** (1.0 / (order + 1))

    if not np.isfinite(h) or h <= 0.0:
        h = interval

    return min(h, interval)


def initial_step(f, x0, xk, y0, order, tol):
    y0 = np.asarray(y0, dtype=float)
    f0 = f(x0, y0)
    h = _initial_step_from_state(x0, xk, y0, f0, order, tol)

    if norm_inf(f0) < 1.0e-12:
        y1 = y0 + h * f0
        f1 = f(x0 + h, y1)
        h2 = _initial_step_from_state(x0 + h, xk, y1, f1, order, tol)
        h = min(h, h2)

    return h


def integrate_fixed(step_func, f, x0, y0, x_end, n_steps):
    x = float(x0)
    y = np.asarray(y0, dtype=float).copy()

    xs = [x]
    ys = [y.copy()]

    if n_steps < 1:
        return np.array(xs, dtype=float), np.array(ys, dtype=float)

    h = (x_end - x0) / n_steps

    for i in range(n_steps):
        if i == n_steps - 1:
            hi = x_end - x
        else:
            hi = h

        y = step_func(f, x, y, hi)
        x += hi

        xs.append(float(x))
        ys.append(y.copy())

    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def runge_global_error_fine(y_coarse, y_fine, order):
    diff = y_fine[::2] - y_coarse
    denom = float(2 ** order - 1)

    if denom == 0.0:
        return 0.0

    return max_norm_rows(diff / denom)


def true_global_errors(x, y):
    exact = exact_solution(x)
    return np.max(np.abs(exact - np.asarray(y)), axis=1)


def solve_constant(
    f,
    step_func,
    order,
    tol,
    x0=X0,
    y0=Y0,
    x_end=X_END,
    h0=None,
    max_iter=50,
    max_steps=1000000
):
    interval = x_end - x0

    if h0 is None:
        h0 = initial_step(f, x0, x_end, y0, order, tol)

    if h0 <= 0.0:
        h0 = interval

    n_fine = max(2, int(np.ceil(interval / h0)))

    if n_fine % 2 != 0:
        n_fine += 1

    for _ in range(max_iter):
        n_coarse = n_fine // 2

        x_coarse, y_coarse = integrate_fixed(
            step_func,
            f,
            x0,
            y0,
            x_end,
            n_coarse
        )

        x_fine, y_fine = integrate_fixed(
            step_func,
            f,
            x0,
            y0,
            x_end,
            n_fine
        )

        err_est = runge_global_error_fine(y_coarse, y_fine, order)

        if np.isfinite(err_est) and err_est <= tol:
            true_err = max_norm_rows(exact_solution(x_fine) - y_fine)

            return {
                'x': x_fine,
                'y': y_fine,
                'h': interval / n_fine,
                'n_steps': n_fine,
                'error_estimate': float(err_est),
                'true_error': float(true_err),
                'rhs_calls': getattr(f, 'calls', None),
                'order': order
            }

        if not np.isfinite(err_est):
            factor = 0.1
        else:
            factor = 0.9 * (tol / err_est) ** (1.0 / order)
            factor = min(max(factor, 0.1), 0.9)

        n_fine = int(np.ceil(n_fine / factor))

        if n_fine % 2 != 0:
            n_fine += 1

        if n_fine > max_steps:
            raise RuntimeError('solve_constant: too many steps')

    raise RuntimeError('solve_constant: did not converge')


def solve_auto(
    f,
    step_func,
    order,
    tol,
    x0=X0,
    y0=Y0,
    x_end=X_END,
    h0=None,
    safety=0.9,
    min_factor=0.2,
    max_factor=2.0,
    max_steps=100000
):
    interval = x_end - x0

    if h0 is None:
        h0 = initial_step(f, x0, x_end, y0, order, tol)

    if h0 <= 0.0:
        h0 = interval

    x = float(x0)
    y = np.asarray(y0, dtype=float).copy()
    h = min(max(float(h0), 1.0e-12), interval)

    xs = [x]
    ys = [y.copy()]

    steps = []
    estimates = []
    true_locals = []

    denom = float(2 ** order - 1)
    step_count = 0

    while x < x_end - 1.0e-14:
        remaining = x_end - x

        if h > remaining:
            h = remaining

        if h <= 1.0e-15:
            break

        while True:
            y_full = step_func(f, x, y, h)
            y_mid = step_func(f, x, y, h / 2.0)
            y_double = step_func(f, x + h / 2.0, y_mid, h / 2.0)

            err_vec = (y_double - y_full) / denom
            err = norm_inf(err_vec)

            if not np.isfinite(err):
                err = float('inf')

            if err <= tol or h <= 1.0e-12:
                y_exact_local = exact_flow(x, y, h)
                true_l = norm_inf(y_exact_local - y_double)

                x = x + h
                y = y_double

                xs.append(float(x))
                ys.append(y.copy())
                steps.append(float(h))
                estimates.append(float(err))
                true_locals.append(float(true_l))

                step_count += 1

                if step_count > max_steps:
                    raise RuntimeError('solve_auto: too many steps')

                if err == 0.0:
                    factor = max_factor
                else:
                    factor = safety * (tol / err) ** (1.0 / (order + 1))
                    factor = min(max(factor, min_factor), max_factor)

                h = h * factor
                remaining = x_end - x

                if h > remaining:
                    h = remaining

                break

            else:
                factor = safety * (tol / err) ** (1.0 / (order + 1))
                factor = min(max(factor, min_factor), 1.0)
                h = h * factor

                if h < 1.0e-12:
                    h = 1.0e-12

    return {
        'x': np.array(xs, dtype=float),
        'y': np.array(ys, dtype=float),
        'h_steps': np.array(steps, dtype=float),
        'error_estimates': np.array(estimates, dtype=float),
        'true_local_errors': np.array(true_locals, dtype=float),
        'n_steps': len(steps),
        'rhs_calls': getattr(f, 'calls', None),
        'order': order
    }
