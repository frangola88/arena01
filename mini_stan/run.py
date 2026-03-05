import numpy as np
from hmc import sample


def run_model(log_prob, grad_log_prob, initial_position):

    samples = sample(
        initial_position=initial_position,
        n_samples=1000,
        step_size=0.01,
        n_steps=20,
        log_prob=log_prob,
        grad_log_prob=grad_log_prob,
        mass_matrix=None,
    )

    return samples