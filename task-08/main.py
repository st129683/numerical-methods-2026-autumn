import tools
import plot_tools
import runge2_constant_step
import runge2_auto_step
import rk4_constant_step
import rk4_auto_step


def fmt(value):
    if value is None:
        return '-'
    return '{:.8e}'.format(value)


def fmt_int(value):
    if value is None:
        return '-'
    return str(value)


def max_global_error(res):
    return tools.max_norm_rows(tools.exact_solution(res['x']) - res['y'])


def endpoint_error(res):
    exact_end = tools.exact_solution(tools.X_END)
    return tools.norm_inf(exact_end - res['y'][-1])


def main():
    print('Variant: xi = 1/17, A = 1/35, B = 1/10')

    print('RK2 coefficients: c2 = {0}, a21 = {1}, b1 = {2}, b2 = {3}'.format(
        runge2_constant_step.C2,
        runge2_constant_step.A21,
        runge2_constant_step.B1,
        runge2_constant_step.B2
    ))

    exact_end = tools.exact_solution(tools.X_END)

    print('Exact y1(pi) = {0}'.format(fmt(exact_end[0])))
    print('Exact y2(pi) = {0}'.format(fmt(exact_end[1])))
    print()

    # Расчеты с постоянным шагом
    f = tools.Counter(tools.rhs)
    rk2_c = runge2_constant_step.solve(f)

    f = tools.Counter(tools.rhs)
    rk4_c = rk4_constant_step.solve(f)

    # Расчеты с автоматическим шагом
    f = tools.Counter(tools.rhs)
    rk2_a = runge2_auto_step.solve(f)

    f = tools.Counter(tools.rhs)
    rk4_a = rk4_auto_step.solve(f)

    # Таблица для методов с постоянным шагом
    header = '{:<22} {:>14} {:>7} {:>8} {:>16} {:>16} {:>14}'.format(
        'method',
        'h',
        'steps',
        'f calls',
        'est max err',
        'true max err',
        'end err'
    )

    print(header)
    print('-' * len(header))

    for name, res in [('RK2 constant', rk2_c), ('RK4 constant', rk4_c)]:
        end_err = endpoint_error(res)

        print('{:<22} {:>14} {:>7} {:>8} {:>16} {:>16} {:>14}'.format(
            name,
            fmt(res['h']),
            fmt_int(res['n_steps']),
            fmt_int(res['rhs_calls']),
            fmt(res['error_estimate']),
            fmt(res['true_error']),
            fmt(end_err)
        ))

    print()

    # Таблица для методов с автоматическим шагом
    header = '{:<22} {:>7} {:>8} {:>14} {:>16} {:>16} {:>16} {:>14}'.format(
        'method',
        'steps',
        'f calls',
        'mean h',
        'max loc est',
        'max true loc',
        'max glob err',
        'end err'
    )

    print(header)
    print('-' * len(header))

    for name, res in [('RK2 auto', rk2_a), ('RK4 auto', rk4_a)]:
        max_glob = max_global_error(res)
        end_err = endpoint_error(res)

        if res['n_steps'] > 0:
            mean_h = (tools.X_END - tools.X0) / res['n_steps']
            max_est = tools.norm_inf(res['error_estimates'])
            max_true_loc = tools.norm_inf(res['true_local_errors'])
        else:
            mean_h = 0.0
            max_est = 0.0
            max_true_loc = 0.0

        print('{:<22} {:>7} {:>8} {:>14} {:>16} {:>16} {:>16} {:>14}'.format(
            name,
            fmt_int(res['n_steps']),
            fmt_int(res['rhs_calls']),
            fmt(mean_h),
            fmt(max_est),
            fmt(max_true_loc),
            fmt(max_glob),
            fmt(end_err)
        ))

    print()

    # График истинной полной погрешности для постоянного шага
    const_plot_data = []

    for name, res in [('RK2 constant', rk2_c), ('RK4 constant', rk4_c)]:
        errors = tools.true_global_errors(res['x'], res['y'])
        const_plot_data.append({
            'label': name,
            'x': res['x'],
            'error': errors
        })

    plot_tools.plot_global_errors(
        const_plot_data,
        filename='global_error.png'
    )

    # Данные для графиков автоматического шага
    auto_plot_data = [
        {
            'label': 'RK2 auto',
            'x': rk2_a['x'],
            'h_steps': rk2_a['h_steps'],
            'error_estimates': rk2_a['error_estimates'],
            'true_local_errors': rk2_a['true_local_errors']
        },
        {
            'label': 'RK4 auto',
            'x': rk4_a['x'],
            'h_steps': rk4_a['h_steps'],
            'error_estimates': rk4_a['error_estimates'],
            'true_local_errors': rk4_a['true_local_errors']
        }
    ]

    plot_tools.plot_step_history(
        auto_plot_data,
        filename='step_history.png'
    )

    plot_tools.plot_local_error_ratio(
        auto_plot_data,
        filename='local_error_ratio.png'
    )

    # Зависимость числа вычислений правой части от точности
    eps_values = [1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5]

    calls_rk2 = []
    calls_rk4 = []

    for eps in eps_values:
        f = tools.Counter(tools.rhs)
        runge2_auto_step.solve(f, tol=eps)
        calls_rk2.append(f.calls)

        f = tools.Counter(tools.rhs)
        rk4_auto_step.solve(f, tol=eps)
        calls_rk4.append(f.calls)

    plot_tools.plot_calls_vs_accuracy(
        eps_values,
        {
            'RK2 auto': calls_rk2,
            'RK4 auto': calls_rk4
        },
        filename='calls_vs_accuracy.png'
    )

    print('Plots saved:')
    print('global_error.png')
    print('step_history.png')
    print('local_error_ratio.png')
    print('calls_vs_accuracy.png')


if __name__ == '__main__':
    main()
