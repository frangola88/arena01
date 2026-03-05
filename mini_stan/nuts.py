import numpy as np
from hmc import HamiltonianDynamics


def nuts_step(q_current, step_size, log_prob, grad_log_prob, mass_matrix):

    dynamics = HamiltonianDynamics(
        log_prob,
        grad_log_prob,
        step_size,
        mass_matrix,
    )

    p0 = dynamics.sample_momentum()

    q_new, p_new = dynamics.leapfrog(q_current, p0, 1)

    H0 = dynamics.hamiltonian(q_current, p0)
    H1 = dynamics.hamiltonian(q_new, p_new)

    accept_prob = np.exp(H0 - H1)

    if np.random.rand() < accept_prob:
        return q_new
    else:
        return q_current