import numpy as np
from pyomo_solver import PyomoSolver
from tqdm import tqdm
from scipy.stats import beta


class Market:
    def __init__(self, N, M, T, p_user, p_item, model="static", verbose=True):

        self.N = N
        self.M = M
        self.V_bar = np.random.uniform(0, 1, (N, M))
        self.T = T
        self.model = model
        self.V = np.zeros((self.N, self.M))
        self.num_obs = np.zeros((self.N, self.M))

        self.E = np.random.choice(len(p_item), size=(T, M), p=p_item)
        self.D = np.random.choice(len(p_user), size=(T, N), p=p_user)

        self.alpha = 4 * self.V_bar
        self.beta = 4 * (1 - self.V_bar)

        self.opt = np.zeros(T)
        if model == "static" or model == "random_exp":
            for t in tqdm(range(T)):
                active_items = (self.E[t] > 0)
                active_users = (self.D[t] > 0)
                solver = PyomoSolver(sum(active_users), sum(active_items))
                X = solver.solve_system(self.V_bar[active_users][:, active_items],
                                        self.E[t][active_items], self.D[t][active_users])
                self.opt[t] = np.sum(self.V_bar[active_users][:, active_items] * X)
        elif model == "random_val":
            price_points = np.linspace(0, 1, 100)
            sf_values = np.array([[beta.sf(price_points, self.alpha[i, j], self.beta[i, j])
                            for j in range(self.M)] for i in range(self.N)])
            expected_rev = sf_values * price_points
            best_expected_rev = np.max(expected_rev, axis=-1)
            for t in tqdm(range(T)):
                active_items = (self.E[t] > 0)
                active_users = (self.D[t] > 0)
                solver = PyomoSolver(sum(active_users), sum(active_items))
                X = solver.solve_system(best_expected_rev[active_users][:, active_items],
                                        self.E[t][active_items], self.D[t][active_users])
                self.opt[t] = np.sum(best_expected_rev[active_users][:, active_items] * X)

        self.t = 0
        self.initial_mask = None
        self.opt_rewards = None

    def reset_market(self):
        self.t = 0
        self.num_obs = np.zeros((self.N, self.M))
        if self.model == "static":
            self.V = self.V_bar
        elif self.model == "random_val":
            self.V = np.random.beta(self.alpha, self.beta)
        elif self.model == "random_exp":
            self.V = 0.01 * np.ones((self.N, self.M))

    def update_valuations(self, accepted):
        if self.model == "static":
            self.V = self.V_bar
        elif self.model == "random_val":
            self.V = np.random.beta(self.alpha, self.beta)
        elif self.model == "random_exp":
            self.V = accepted * (self.num_obs * self.V + np.random.beta(self.alpha, self.beta))/(self.num_obs + 1) +\
                     (1 - accepted) * self.V
        self.num_obs = self.num_obs + accepted

    def make_offers(self, X, p):
        return X * (self.V >= p[np.newaxis, :])

    def run_simulations(self, alg):
        self.reset_market()
        regrets = np.zeros(self.T)
        pbar = tqdm(total=self.T)

        while True:
            X, p = alg.get_offers(self.E[self.t], self.D[self.t])
            accepted = self.make_offers(X, p)
            alg.update(X, p, accepted)

            p_extended = p[np.newaxis, :]
            regrets[self.t] = self.opt[self.t] - np.sum(accepted * p_extended)

            self.t = self.t + 1
            pbar.update(1)
            if self.t == self.T:
                break
            self.update_valuations(accepted)

        return regrets