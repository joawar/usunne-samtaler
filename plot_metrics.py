import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pathlib
import numpy as np
import scipy


def plot_metric(all_runs, metric, ax, show=False, save=False):
    metrics = list()
    max_vals = list()
    last_vals = list()
    for run in all_runs:
        metrics.append(run[metric])
        max_vals.append(max(run[metric]))
        last_vals.append(run[metric][-1])
    legends = list()
    for i in range(n_folds):
        ax.plot(metrics[i], label=i)
        # legends.append(f'fold {i+1}. max: {round(max_vals[i], 3)}. end: {round(last_vals[i], 3)}')
    ax.set_title(metric)
    ax.set_ylabel(metric)
    ax.get_xaxis().set_visible(False)
    # ax.set_xlabel('log step')
    # ax.legend(legends)
    if save:
        fig.savefig(f'plots/{run_name}/{metric}')
    if show:
        plt.show()
    return ax


def plot_mean_sd(all_runs, metric, run_name, ax, look_at='last', show=False, save=False):
    max_vals = list()
    last_vals = list()
    for run in all_runs:
        max_vals.append(max(run[metric]))
        last_vals.append(run[metric][-1])
    legends = list()
    if look_at == 'last':
        mean = np.mean(last_vals)
        sd = np.std(last_vals)
        for val in last_vals:
            ax.plot(val, 0, '.')
    elif look_at == 'max':
        mean = np.mean(max_vals)
        sd = np.std(max_vals)
        for val in max_vals:
            ax.plot(val, 0, '.')
    x = np.linspace(mean-3*sd, mean+3*sd, 100)
    ax.set_title(f'{look_at} value')
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel(metric)
    props = {
        'boxstyle': 'round',
        'facecolor': 'wheat',
        'alpha': 0.5
    }
    # ax.plot(x, scipy.stats.norm.pdf(x, mean, sd))
    ax.text(
        x=0.05,
        y=0.95,
        s=f'mean: {round(mean, 2)}\nsd: {round(sd, 2)}',
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment='top',
        bbox=props
    )
    if save:
        fig.savefig(f'plots/{run_name}/{metric}-{look_at}')
    if show:
        plt.show()
    return ax


if __name__ == '__main__':
    n_folds = 10
    run_name = 'vanilla'
    characteristic = 'antagonise'
    metric_path = f'out/metrics/{characteristic}/{run_name}/nb-bert-antagonise-undersampling'
    pathlib.Path(f'plots/{characteristic}/{run_name}/').mkdir(exist_ok=True, parents=True)
    all_runs = list()
    for i in range(n_folds):
        with open(f'{metric_path}_{i}.json') as f:
            dct = json.load(f)
            all_runs.append(dct)
    metrics = list(all_runs[0].keys())
    metrics.remove('Conf_Mat')
    for metric in metrics:
        fig = plt.figure(figsize=(10, 8), constrained_layout=True)
        gs = fig.add_gridspec(4, 4)
        fig_ax1 = fig.add_subplot(gs[:3, :])
        run = plot_metric(all_runs, metric, fig_ax1)
        fig_ax2 = fig.add_subplot(gs[3, :2])
        last_mean_sd = plot_mean_sd(
            all_runs,
            metric,
            run_name,
            fig_ax2,
            look_at='last',
            show=False
        )
        fig_ax3 = fig.add_subplot(gs[3, 2:])
        max_mean_sd = plot_mean_sd(
            all_runs,
            metric,
            run_name,
            fig_ax3,
            look_at='max',
            show=False
        )
        # plt.show()
        fig.savefig(f'plots/{characteristic}/{run_name}/{metric}-full')