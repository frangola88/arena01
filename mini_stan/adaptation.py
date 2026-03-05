import numpy as np


class MassMatrixAdaptation:

    def __init__(self, dim):

        self.dim = dim
        self.samples = []

    def update(self, q):

        self.samples.append(q.copy())

    def compute_mass_matrix(self):

        samples = np.array(self.samples)

        var = np.var(samples, axis=0)

        var[var < 1e-6] = 1e-6

        return var