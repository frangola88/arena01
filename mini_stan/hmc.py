import numpy as np


# -----------------------------------------------------------
# Hamiltonian Dynamics
# -----------------------------------------------------------

class HamiltonianDynamics:

    def __init__(self, log_prob, grad_log_prob, step_size, mass_matrix):

        self.log_prob = log_prob
        self.grad_log_prob = grad_log_prob

        self.epsilon = step_size

        self.mass_matrix = np.array(mass_matrix)
        self.inv_mass = 1.0 / self.mass_matrix


    # -------------------------------------------------------
    # Momentum sampling
    # -------------------------------------------------------

    def sample_momentum(self):

        dim = len(self.mass_matrix)

        return np.random.normal(size=dim) * np.sqrt(self.mass_matrix)


    # -------------------------------------------------------
    # Kinetic energy
    # -------------------------------------------------------

    def kinetic_energy(self, p):

        return 0.5 * np.sum(p * p * self.inv_mass)


    # -------------------------------------------------------
    # Hamiltonian
    # -------------------------------------------------------

    def hamiltonian(self, q, p):

        return -self.log_prob(q) + self.kinetic_energy(p)


    # -------------------------------------------------------
    # Leapfrog integrator
    # -------------------------------------------------------

    def leapfrog(self, q, p, n_steps):

        q = q.copy()
        p = p.copy()

        p += 0.5 * self.epsilon * self.grad_log_prob(q)

        for i in range(n_steps):

            q += self.epsilon * (p * self.inv_mass)

            if i != n_steps - 1:
                p += self.epsilon * self.grad_log_prob(q)

        p += 0.5 * self.epsilon * self.grad_log_prob(q)

        return q, p


# -----------------------------------------------------------
# Single HMC transition
# -----------------------------------------------------------

def hmc_step(q_current, dynamics, n_steps):

    p_current = dynamics.sample_momentum()

    q_new, p_new = dynamics.leapfrog(
        q_current,
        p_current,
        n_steps,
    )

    H_current = dynamics.hamiltonian(q_current, p_current)
    H_new = dynamics.hamiltonian(q_new, p_new)

    accept_prob = np.exp(H_current - H_new)

    if np.random.rand() < accept_prob:
        return q_new
    else:
        return q_current


# -----------------------------------------------------------
# HMC sampler
# -----------------------------------------------------------

def sample(
    initial_position,
    n_samples,
    step_size,
    n_steps,
    log_prob,
    grad_log_prob,
    mass_matrix=None,
):

    q = np.array(initial_position)

    dim = len(q)

    if mass_matrix is None:
        mass_matrix = np.ones(dim)

    dynamics = HamiltonianDynamics(
        log_prob,
        grad_log_prob,
        step_size,
        mass_matrix,
    )

    samples = np.zeros((n_samples, dim))

    for i in range(n_samples):

        q = hmc_step(
            q,
            dynamics,
            n_steps,
        )

        samples[i] = q

    return samples