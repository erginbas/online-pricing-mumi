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
exp_files = ["logs_final/regret_total_1684278371.npz", "logs_final/regret_total_1684276295.npz", "logs_final/regret_total_1684276216.npz"]
num_exp = len(exp_files)

colors = ["#ff1f5b", "#22bb3b", "#009ade"]

fig = plt.figure(constrained_layout=True, figsize=(12, 2.5), dpi=500)
# fig.tight_layout(w_pad=0.4, h_pad=0.4, rect=(0.01, 0.01, 0.99, 0.99))
# fig.subplots_adjust(wspace=0.25, hspace=2)

subfigs = fig.subfigures(nrows=1, ncols=num_exp, wspace=0.02, hspace=0.01)

for e in range(num_exp):

    ax = subfigs[e].subplots()
    # subfigs[e].subplots_adjust(wspace=0.25, hspace=0.5)

    file = np.load(exp_files[e])

    regret_total = file['regret_total']
    Ts = file['Ts']

    num_algos, num_Ts, reps = regret_total.shape

    ax.set_title(names[e])
    # ax[1].set_title(names[e])

    for a in range(num_algos):
        mu = np.mean(regret_total[a], axis=-1)
        std = np.std(regret_total[a], axis=-1)
        ax.plot(Ts, mu, color=colors[e])
        ax.fill_between(Ts, mu - 2 * std, mu + 2 * std, color=colors[e], alpha=0.3)

    ax.grid()
    ax.set(xlabel=r'Time Horizon $( \; T \;)$', ylabel=r'Regret')
    ax.set_ylim(bottom=1e-4)
    if e == 0:
        ax.set_xscale('log', base=2)
    else:
        ax.set_xlim(left=1e-4)
    # ax.set_xticks(Ts)
    ax.ticklabel_format(axis="y", scilimits=(-3, 4))

plt.savefig('experimental_results_regret_vs_horizon.pdf')
plt.show()

