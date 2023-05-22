from market import Market
import algorithms_random_experience
import matplotlib.pyplot as plt
import numpy as np
from time import time

N = 120
M = 80
T = 3000

reps = 10
algorithm_list = [algorithms_random_experience.RandomExperienceAlgorithm]
regrets = np.zeros((len(algorithm_list), reps, T))

for rep in range(reps):
    market = Market(N, M, T, p_user=[0.4, 0.4, 0.2], p_item=[0.5, 0.5], model="random_exp")
    for a in range(len(algorithm_list)):
        regrets[a, rep, :] = market.run_simulations(algorithm_list[a](N, M, T))

for a in range(len(algorithm_list)):
    mu = np.mean(regrets[a], axis=0)
    std = np.std(regrets[a], axis=0)
    ts = np.linspace(1, len(mu), len(mu))
    plt.plot(ts, mu)
    plt.fill_between(ts, mu - std, mu + std, alpha=0.5)
plt.show()

for a in range(len(algorithm_list)):
    mu = np.mean(np.cumsum(regrets[a], axis=-1), axis=0)
    plt.plot(ts, mu)
plt.show()

np.save(f"logs/regret_{int(time())}.npy", regrets)
