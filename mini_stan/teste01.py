import numpy as np
from hmc import sample


def log_prob(x):
    return -0.5 * np.sum(x**2)


def grad_log_prob(x):
    return -x


samples = sample(
    initial_position=[0.0, 0.0],
    n_samples=10000,
    step_size=0.1,
    n_steps=10,
    log_prob=log_prob,
    grad_log_prob=grad_log_prob
)

print(samples.mean(axis=0))

print("mean:", samples.mean(axis=0))
print("std :", samples.std(axis=0))

import matplotlib.pyplot as plt

plt.figure()
plt.scatter(samples[:,0], samples[:,1], s=2)
plt.axis("equal")
plt.title("HMC samples")

plt.savefig("hmc_samples.png", dpi=150)
print("plot salvo em hmc_samples.png")