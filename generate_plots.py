import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import matplotlib.ticker as ticker

plt.style.use(['seaborn-deep', 'paper.mplstyle'])
matplotlib.rcParams.update({"axes.grid": False})

# # matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'text.usetex': True,
#     'pgf.rcfonts': True,
# })

matplotlib.rc('font', **{'family': 'serif', 'serif': ['Nimbus Roman No9 L']})

names = ["Fixed Valuations", "Random Experiences", "Random Valuations"]
exp_files = ["logs_final/regret_1675980865.npy", "logs_final/regret_1675984506.npy", "logs_final/regret_1675914640.npy"]
num_exp = len(exp_files)

colors = ["#ff1f5b", "#22bb3b", "#009ade"]

fig = plt.figure(constrained_layout=True, figsize=(12, 4), dpi=500)
# fig.tight_layout(w_pad=0.4, h_pad=0.4, rect=(0.01, 0.01, 0.99, 0.99))
# fig.subplots_adjust(wspace=0.25, hspace=2)

subfigs = fig.subfigures(nrows=1, ncols=num_exp, wspace=0.1, hspace=0.01)

for e in range(num_exp):

    ax = subfigs[e].subplots(nrows=2, ncols=1)
    # subfigs[e].subplots_adjust(wspace=0.25, hspace=0.5)

    regrets = np.load(exp_files[e])

    num_algos, reps, T = regrets.shape

    max_regret = np.max(np.mean(regrets, axis=1))

    ax[0].set_title(names[e])
    # ax[1].set_title(names[e])

    for a in range(num_algos):
        mu = np.mean(regrets[a], axis=0)
        std = np.std(regrets[a], axis=0)
        ts = np.linspace(1, len(mu), len(mu))
        ax[0].plot(ts, mu, color=colors[e])
        ax[0].fill_between(ts, mu - 2 * std, mu + 2 * std, color=colors[e], alpha=0.3)

    ax[0].grid()
    ax[0].set(xlabel='Iteration', ylabel='Instantaneous Regret')
    ax[0].set_ylim(bottom=1e-4)
    ax[0].set_xlim(left=1e-4)
    ax[0].set_xticks([T//4, 2*T//4, 3*T//4, T])
    ax[0].set_yticks([10, 20, 30, 40])
    ax[0].ticklabel_format(axis="y", scilimits=(-3, 4))

    for a in range(num_algos):
        mu = np.mean(np.cumsum(regrets[a], axis=-1), axis=0)
        std = np.std(np.cumsum(regrets[a], axis=-1), axis=0)
        ts = np.linspace(1, len(mu), len(mu))
        ax[1].plot(ts, mu, color=colors[e])
        ax[1].fill_between(ts, mu - 2 * std, mu + 2 * std, color=colors[e], alpha=0.3)

    ax[1].grid()
    ax[1].set(xlabel='Iteration', ylabel='Cumulative Regret')
    ax[1].set_ylim(bottom=1e-4)
    ax[1].set_xlim(left=1e-4)
    ax[1].set_yticks([1e4, 2e4, 3e4, 4e4])
    ax[1].set_xticks([T//4, 2*T//4, 3*T//4, T])
    ax[1].ticklabel_format(scilimits=(-3, 4))
    subfigs[e].align_ylabels(ax)

plt.savefig('experimental_results.pdf')
plt.show()

