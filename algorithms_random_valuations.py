import numpy as np
from pyomo_solver import PyomoSolver
import numpy.lib.index_tricks as ndi
from utils import replace_submatrix

class RandomValuationsAlgorithm:
    def __init__(self, N, M, T):

        self.N = N
        self.M = M
        self.T = T

        self.K = int(np.ceil(4 * (T / N)**(1/4)))
        self.conf_const = np.sqrt(8*np.log((self.N * self.M)**2 * self.K * self.T)) / 80

        print(self.K)

        self.counter = np.zeros((N, M, self.K))
        self.sum = np.zeros((N, M, self.K))

    @staticmethod
    def add_to_choice(a, b, idx):
        for I in ndi.ndindex(idx.shape):
            a[I][idx[I]] += b[I]
        return a

    def get_offers(self, E, D):
        with np.errstate(divide='ignore', invalid='ignore'):
            conf_level = np.minimum(self.conf_const / np.sqrt(self.counter), 1)
            R = np.minimum(np.nan_to_num(self.sum / self.counter + conf_level, nan=1),  1)

        self.best_price_idx = np.argmax(R, axis=-1)

        active_items = (E > 0)
        active_users = (D > 0)
        solver = PyomoSolver(sum(active_users), sum(active_items))
        X = np.zeros((self.N, self.M))
        X = replace_submatrix(X, np.where(active_users)[0], np.where(active_items)[0],
                              solver.solve_system(np.max(R, axis=-1)[active_users][:, active_items],
                                                  E[active_items], D[active_users]))

        p = np.sum(X * self.best_price_idx, axis=0) / self.K
        return X, p

    def update(self, X, p, accepted):
        p_extended = p[np.newaxis, :]
        self.counter = self.add_to_choice(self.counter, X, self.best_price_idx)
        self.sum = self.add_to_choice(self.sum, accepted * p_extended, self.best_price_idx)



