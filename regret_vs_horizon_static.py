import logging
from market import Market
import algorithms_static
import matplotlib.pyplot as plt
import numpy as np
from time import time

exp_save_path = "logs"

logging.basicConfig(filename=f"{exp_save_path}/info.log",
                    filemode='a',
                    format='[%(asctime)s] %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

N = 30
M = 20
Ts = np.logspace(6, 14, 9, base=2.0)

print(Ts)

reps = 5
algorithm_list = [algorithms_static.StaticAlgorithmIncremental]
regret_total = np.zeros((len(algorithm_list), len(Ts), reps))

for T_idx in range(len(Ts)):
    T = int(Ts[T_idx])
    for rep in range(reps):
        market = Market(N, M, T, p_user=[0.4, 0.4, 0.2], p_item=[0.5, 0.5], model="static")
        for a in range(len(algorithm_list)):
            regret_total[a, T_idx, rep] = np.sum(market.run_simulations(algorithm_list[a](N, M, T)))

np.savez(f"logs/regret_total_{int(time())}.npz", regret_total=regret_total, Ts=Ts)

for a in range(len(algorithm_list)):
    mu = np.mean(regret_total[a], axis=-1)
    std = np.std(regret_total[a], axis=-1)
    ts = np.linspace(1, len(mu), len(mu))
    plt.plot(Ts, mu)
    plt.fill_between(Ts, mu - std, mu + std, alpha=0.5)
plt.show()