import matplotlib.pyplot as plt
import numpy as np


def plot_errors(n_values, errors_dict, title='convergence of quadrature methods'):
    plt.figure(figsize=(10, 6))
    for method_name, errors in errors_dict.items():
        valid = ~np.isnan(errors)
        plt.loglog(
            np.array(n_values)[valid],
            errors[valid],
            marker='o',
            label=method_name
        )

    plt.xlabel('number of intervals (n)')
    plt.ylabel('absolute error')
    plt.title(title)
    plt.grid(True, which='both', ls='-', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
