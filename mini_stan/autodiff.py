import numpy as np


class Var:

    def __init__(self, value, parents=None, grad_fn=None):

        self.value = value
        self.parents = parents or []
        self.grad_fn = grad_fn
        self.grad = 0


    def backward(self, g=1.0):

        self.grad += g

        if self.grad_fn is not None:

            grads = self.grad_fn(g)

            for p, gp in zip(self.parents, grads):
                p.backward(gp)


def wrap(x):

    if isinstance(x, Var):
        return x

    return Var(x)


def add(a, b):

    a, b = wrap(a), wrap(b)

    def grad_fn(g):
        return [g, g]

    return Var(a.value + b.value, [a, b], grad_fn)


def mul(a, b):

    a, b = wrap(a), wrap(b)

    def grad_fn(g):
        return [g * b.value, g * a.value]

    return Var(a.value * b.value, [a, b], grad_fn)


def log(x):

    x = wrap(x)

    def grad_fn(g):
        return [g / x.value]

    return Var(np.log(x.value), [x], grad_fn)


def exp(x):

    x = wrap(x)

    v = np.exp(x.value)

    def grad_fn(g):
        return [g * v]

    return Var(v, [x], grad_fn)


def grad(f):

    def g(x):

        vars = [Var(v) for v in x]

        out = f(vars)

        out.backward()

        return np.array([v.grad for v in vars])

    return g