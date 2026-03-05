import numpy as np


def normal_logpdf(x, mu, sigma):

    return -0.5 * (
        np.log(2 * np.pi * sigma**2)
        + ((x - mu) ** 2) / (sigma**2)
    )


def normal_lpdf_vec(x, mu, sigma):

    return np.sum(normal_logpdf(x, mu, sigma))