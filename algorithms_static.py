import numpy as np
from pyomo_solver import PyomoSolver


class StaticAlgorithmIncremental:
    def __init__(self, N, M, T):

        self.N = N
        self.M = M
        self.T = T

        self.A = np.zeros((N, M))
        self.B = np.ones((N, M))
        self.beta = 0.5 * np.ones((N, M))

        self.eps = 10 / (self.M * self.T)

        # use a MIP solver to calculate optimal allocations efficiently
        self.solver = PyomoSolver(self.N, self.M)

    def get_offers(self, E, D):
        X = self.solver.solve_system(self.B, E, D)
        learning_phase = (self.B - self.A) > self.eps
        p = np.sum(learning_phase * X * (self.A + self.beta) + (1 - learning_phase) * X * self.A, axis=0)
        return X, p

    def update(self, X, p, accepted):
        p_extended = p[np.newaxis, :]
        rejected = X - accepted
        self.A = accepted * np.maximum(self.A, p_extended) + (1 - accepted) * self.A
        self.B = rejected * np.minimum(self.B, p_extended) + (1 - rejected) * self.B
        update_beta = (self.B - self.A) < (self.beta + 1e-5)
        self.beta = update_beta * (self.beta ** 2) + (1 - update_beta) * self.beta


class StaticAlgorithmBinary:
    def __init__(self, N, M, T):

        self.N = N
        self.M = M
        self.T = T

        self.A = np.zeros((N, M))
        self.B = np.ones((N, M))

        self.eps = 1 / (self.M * self.T)

        # use a MIP solver to calculate optimal allocations efficiently
        self.solver = PyomoSolver(self.N, self.M)

    def get_offers(self, E, D):
        X = self.solver.solve_system(self.B, E, D)
        learning_phase = (self.B - self.A) > self.eps
        p = np.sum(learning_phase * X * (self.B + self.A) / 2 + (1 - learning_phase) * X * self.A, axis=0)
        return X, p

    def update(self, X, p, accepted):
        p_extended = p[np.newaxis, :]
        rejected = X - accepted
        self.A = accepted * np.maximum(self.A, p_extended) + (1 - accepted) * self.A
        self.B = rejected * np.minimum(self.B, p_extended) + (1 - rejected) * self.B





