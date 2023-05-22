import numpy as np
from pyomo_solver import PyomoSolver
import numpy.lib.index_tricks as ndi
from utils import replace_submatrix


class RandomExperienceAlgorithm:
    def __init__(self, N, M, T):

        self.N = N
        self.M = M
        self.T = T

        self.A = np.zeros((N, M))
        self.B = np.ones((N, M))
        self.counter = np.zeros((N, M))

        self.conf_const = np.sqrt(8*np.log((self.N * self.M)**2 * self.T)) / 50

        self.learning_rounds = 2 ** np.linspace(0, int(np.log2(self.T)+1), int(np.log2(self.T)+1)+1, dtype=int)

        print(self.learning_rounds)

        self.to_learn = np.zeros((N, M))

    def get_offers(self, E, D):
        active_items = (E > 0)
        active_users = (D > 0)
        solver = PyomoSolver(sum(active_users), sum(active_items))
        X = np.zeros((self.N, self.M))
        X = replace_submatrix(X, np.where(active_users)[0], np.where(active_items)[0], solver.solve_system(self.B[active_users][:, active_items],
                                                                E[active_items], D[active_users]))

        p = np.sum(X * (1 - self.to_learn) * self.A + X * self.to_learn * (self.A + self.B)/2, axis=0)
        self.remove_scheduled_learning(X * self.to_learn)
        return X, p

    def update(self, X, p, accepted):
        p_extended = p[np.newaxis, :]
        rejected = X - accepted
        conf_level = np.minimum(self.conf_const / np.sqrt(self.counter), 1)
        self.A = (1-accepted) * self.A + accepted * np.maximum(self.A, p_extended - conf_level)
        self.B = (1-rejected) * self.B + rejected * np.minimum(self.B, p_extended + conf_level)
        self.counter = self.counter + accepted
        self.schedule_learning(accepted)

    def schedule_learning(self, accepted):
        for i in range(self.N):
            for j in range(self.M):
                self.to_learn[i, j] = self.to_learn[i, j] or (self.counter[i, j] in self.learning_rounds and accepted[i][j])

    def remove_scheduled_learning(self, learning_offers):
        self.to_learn = self.to_learn - learning_offers
