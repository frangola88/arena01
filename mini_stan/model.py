import numpy as np
from distributions import normal_lpdf_vec
from autodiff import grad


class Model:

    def __init__(self, data):

        self.x = data["x"]
        self.y = data["y"]


    def log_prob(self, theta):

        a, b, sigma = theta

        mu = a * self.x + b

        lp = normal_lpdf_vec(self.y, mu, sigma)

        return lp


def compile_model(data):

    m = Model(data)

    log_prob = m.log_prob

    grad_log_prob = grad(log_prob)

    return log_prob, grad_log_prob