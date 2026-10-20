import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _finish(fig, filename, show):
    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


def plot_global_errors(results, filename='global_error.png', show=False):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for item in results:
        err = np.asarray(item['error'], dtype=float)
        err_plot = np.where(err > 0.0, err, np.nan)
        ax.plot(item['x'], err_plot, label=item['label'], marker='.', linestyle='-')
    ax.set_yscale('log')
    ax.set_xlabel('x')
    ax.set_ylabel('true total error, max norm')
    ax.set_title('True total error for constant step methods')
    ax.grid(True, which='both', alpha=0.4)
    ax.legend(loc='upper left')
    _finish(fig, filename, show)


def plot_step_history(results, filename='step_history.png', show=False):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for item in results:
        x = item['x'][:-1]
        ax.step(x, item['h_steps'], where='post', label=item['label'], linewidth=1.6)
    ax.set_xlabel('x')
    ax.set_ylabel('step h')
    ax.set_title('Automatic step size')
    ax.grid(True, alpha=0.4)
    ax.legend(loc='upper left')
    _finish(fig, filename, show)


def plot_local_error_ratio(results, filename='local_error_ratio.png', show=False):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for item in results:
        x = item['x'][:-1]
        est = np.asarray(item['error_estimates'], dtype=float)
        true = np.asarray(item['true_local_errors'], dtype=float)
        mask = est > 0.0
        if np.any(mask):
            ax.plot(x[mask], true[mask] / est[mask], marker='.', linestyle='-', label=item['label'])
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1.0)
    ax.set_xlabel('x')
    ax.set_ylabel('true local error / Runge estimate')
    ax.set_title('Reliability of local error estimate')
    ax.grid(True, alpha=0.4)
    ax.legend(loc='upper left')
    _finish(fig, filename, show)


def plot_calls_vs_accuracy(eps_values, calls_by_label, filename='calls_vs_accuracy.png', show=False):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label, calls in calls_by_label.items():
        ax.loglog(eps_values, calls, marker='o', linestyle='-', label=label)
    ax.invert_xaxis()
    ax.set_xlabel('tolerance eps (stricter to the right)')
    ax.set_ylabel('right hand side calls')
    ax.set_title('Cost versus accuracy')
    ax.grid(True, which='both', alpha=0.4)
    ax.legend(loc='upper right')
    _finish(fig, filename, show)
