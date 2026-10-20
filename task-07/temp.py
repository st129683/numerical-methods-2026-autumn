import numpy as np
from tools import compute_moments

for n in [25, 100, 400, 800]:
    h = 2 / n
    mu = compute_moments(2.0, 2.0 + h, 3.0, 1/6, 5)
    A = np.array([
        [mu[0], mu[1], mu[2]],
        [mu[1], mu[2], mu[3]],
        [mu[2], mu[3], mu[4]]
    ])
    print(n, np.linalg.cond(A))
