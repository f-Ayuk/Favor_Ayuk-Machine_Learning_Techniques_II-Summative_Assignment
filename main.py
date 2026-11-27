"""
Smart irrigation RL: train, tune, evaluate and plot everything for the report.

    python main.py                    # full run: 4 algorithms x 10 configs
    python main.py --jobs 4           # same, 4 runs in parallel (much faster)
    python main.py --quick            # smoke test; writes to quick/
    python main.py --output-dir run2  # write outputs outside the default locations
    python main.py --skip-training    # re-evaluate / re-plot from saved models + logs

Outputs
    logs/{algo}_run_{k}.csv           per-episode training history of every run
    logs/hyperparameter_results.csv   one row per run: config + validation scores
    logs/final_comparison.csv         best agent per algorithm vs baselines (test seasons)
    logs/stress_test.csv              best agents on unseen harsher climates
    models/{algo}/run_{k}.pt          every trained model;  models/{algo}/best.pt
    assets/*.png                      figures for the report
    quick/                            isolated outputs from --quick
"""

import argparse
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from training.algorithms import load_agent
from training.evaluation import (STRESS_SCENARIOS, agent_policy, baseline_policies,
                                 evaluate, summarise)
from training.experiments import CONFIGS, run_experiments

ALGORITHMS = ["DQN", "REINFORCE", "A2C", "PPO"]
COLORS = {"DQN": "#2a78d6", "REINFORCE": "#eb6834", "A2C": "#1baf7a", "PPO": "#eda100"}
BASELINE_COLOR = "#52514e"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})


# ---------------------------------------------------------------------------
def select_best(results, output_dir="."):
    """Best run per algorithm = highest mean reward on the validation seasons."""
    best = results.loc[results.groupby("algorithm")["val_reward_mean"].idxmax()]
    for _, row in best.iterrows():
        folder = os.path.join(output_dir, "models", row.algorithm.lower())
        shutil.copy(os.path.join(folder, f"run_{row.run}.pt"),
                    os.path.join(folder, "best.pt"))
    best[["algorithm", "run", "config", "val_reward_mean"]].to_csv(
        os.path.join(output_dir, "logs", "best_runs.csv"), index=False)
    return best


def final_comparison(algorithms, n_episodes, output_dir="."):
    rows, per_episode = [], []
    policies = {
        a: agent_policy(load_agent(os.path.join(
            output_dir, "models", a.lower(), "best.pt")))
        for a in algorithms
    }
    policies.update(baseline_policies())
    for name, policy in policies.items():
        df = evaluate(policy, n_episodes=n_episodes)
        rows.append({"policy": name, **summarise(df)})
        per_episode.append(df.assign(policy=name))
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(output_dir, "logs", "final_comparison.csv"), index=False)
    pd.concat(per_episode).to_csv(
        os.path.join(output_dir, "logs", "final_comparison_episodes.csv"), index=False)
    return table


def stress_test(algorithms, n_episodes, output_dir="."):
    rows = []
    policies = {
        a: agent_policy(load_agent(os.path.join(
            output_dir, "models", a.lower(), "best.pt")))
        for a in algorithms
    }
    policies["Threshold rule (<0.45)"] = baseline_policies()["Threshold rule (<0.45)"]
    for scenario, kwargs in STRESS_SCENARIOS.items():
        for name, policy in policies.items():
            s = summarise(evaluate(policy, n_episodes=n_episodes, **kwargs))
            rows.append({"scenario": scenario, "policy": name, **s})
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(output_dir, "logs", "stress_test.csv"), index=False)
    return table


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _history(algorithm, run, output_dir="."):
    return pd.read_csv(os.path.join(
        output_dir, "logs", f"{algorithm.lower()}_run_{run}.csv"))


def plot_learning_curves(results, best, baselines, window=50, output_dir="."):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharey=True)
    for ax, algorithm in zip(axes.flat, ALGORITHMS):
        runs = results[results.algorithm == algorithm]
        if runs.empty:
            ax.set_visible(False)
            continue
        best_run = int(best.set_index("algorithm").loc[algorithm, "run"])
        for run in runs.run:
            h = _history(algorithm, run, output_dir)
            is_best = run == best_run
            ax.plot(h.episode, h.reward.rolling(window, min_periods=1).mean(),
                    color=COLORS[algorithm], lw=2.2 if is_best else 1.0,
                    alpha=1.0 if is_best else 0.28,
                    label=f"best (run {run})" if is_best else None)
        for name, style in [("Threshold rule (<0.45)", "--"), ("Always OFF", ":")]:
            ax.axhline(baselines[name], color=BASELINE_COLOR, ls=style, lw=1.2, label=name)
        ax.set_title(f"{algorithm}: {len(runs)} recorded configurations")
        ax.set_xlabel("Episode (growing season)")
        ax.set_ylabel(f"Season reward ({window}-episode mean)")
        ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "assets", "learning_curves.png"), dpi=200)
    plt.close(fig)


def plot_best_runs(best, baselines, window=50, output_dir="."):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for _, row in best.iterrows():
        h = _history(row.algorithm, row.run, output_dir)
        smooth = h.reward.rolling(window, min_periods=1).mean()
        ax.plot(h.episode, smooth, color=COLORS[row.algorithm], lw=2,
                label=f"{row.algorithm} (run {row.run})")
    ax.axhline(baselines["Threshold rule (<0.45)"], color=BASELINE_COLOR, ls="--",
               lw=1.2, label="Threshold rule")
    ax.axhline(baselines["Always OFF"], color=BASELINE_COLOR, ls=":", lw=1.2, label="Always OFF")
    ax.set_title("Best configuration of each algorithm")
    ax.set_xlabel("Episode (growing season)")
    ax.set_ylabel(f"Season reward ({window}-episode mean)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "assets", "best_runs_comparison.png"), dpi=200)
    plt.close(fig)


def plot_objectives(best, window=20, output_dir="."):
    """DQN loss curve + policy entropy curves of the PG methods."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    if "DQN" in set(best.algorithm):
        h = _history("DQN", int(best.set_index("algorithm").loc["DQN", "run"]), output_dir)
        ax1.plot(h.episode, h.loss.rolling(window, min_periods=1).mean(),
                 color=COLORS["DQN"], lw=1.8)
    ax1.set_title("DQN objective (TD loss, best run)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Smooth-L1 TD loss")
    for _, row in best[best.algorithm != "DQN"].iterrows():
        h = _history(row.algorithm, row.run, output_dir)
        ax2.plot(h.episode, h.entropy.ffill().rolling(window, min_periods=1).mean(),
                 color=COLORS[row.algorithm], lw=1.8, label=row.algorithm)
    ax2.set_title("Policy entropy (best runs)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Entropy of the 512-action policy (nats)")
    ax2.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "assets", "objective_and_entropy.png"), dpi=200)
    plt.close(fig)


def plot_final(table, output_dir="."):
    t = table.sort_values("reward_mean")
    colors = [COLORS.get(p, BASELINE_COLOR) for p in t.policy]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, col, label in [(axes[0], "reward_mean", "Season reward"),
                           (axes[1], "plots_alive", "Plots alive at harvest (of 9)"),
                           (axes[2], "water_used", "Reservoir used (fraction)")]:
        err = t["reward_std"] if col == "reward_mean" else None
        ax.barh(t.policy, t[col], xerr=err, color=colors, height=0.6,
                error_kw={"lw": 1, "ecolor": "#888"})
        ax.set_xlabel(label)
        ax.grid(axis="y", visible=False)
    axes[1].set_yticklabels([])
    axes[2].set_yticklabels([])
    fig.suptitle("Test seasons (unseen weather): trained agents vs rule-based baselines")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "assets", "final_comparison.png"), dpi=200)
    plt.close(fig)


def plot_stress(table, output_dir="."):
    scenarios = list(dict.fromkeys(table.scenario))
    policies = list(dict.fromkeys(table.policy))
    width = 0.8 / len(policies)
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, p in enumerate(policies):
        sub = table[table.policy == p].set_index("scenario").loc[scenarios]
        xs = [k + i * width - 0.4 + width / 2 for k in range(len(scenarios))]
        ax.bar(xs, sub.reward_mean, width * 0.92, color=COLORS.get(p, BASELINE_COLOR),
               label=p)
    ax.set_xticks(range(len(scenarios)), scenarios)
    ax.set_ylabel("Mean season reward")
    ax.set_title("Robustness: climates never seen during training")
    ax.legend(frameon=False, fontsize=9, ncol=len(policies))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "assets", "stress_test.png"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=600,
                        help="training seasons per run (same for every algorithm)")
    parser.add_argument("--jobs", type=int, default=1, help="runs in parallel")
    parser.add_argument("--algos", nargs="+", default=ALGORITHMS)
    parser.add_argument("--quick", action="store_true", help="tiny smoke test")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--test-episodes", type=int, default=50)
    parser.add_argument("--output-dir", default=None,
                        help="directory for logs, models, and figures")
    args = parser.parse_args()

    algorithms = [a.upper() for a in args.algos]
    output_dir = args.output_dir or ("quick" if args.quick else ".")
    runs = None
    if args.quick:
        args.episodes, args.test_episodes, runs = 20, 5, 2
    os.makedirs(os.path.join(output_dir, "assets"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)

    if args.skip_training:
        results = pd.read_csv(os.path.join(output_dir, "logs", "hyperparameter_results.csv"))
        results = results[results.algorithm.isin(algorithms)]
    else:
        n_runs = runs or 10
        print(f"Training {len(algorithms)} algorithms x {n_runs} configs x "
              f"{args.episodes} seasons ({args.jobs} in parallel)\n")
        results = run_experiments(algorithms, episodes=args.episodes,
                                  jobs=args.jobs, runs=runs, output_dir=output_dir)

    best = select_best(results, output_dir)
    print("\nBest configuration per algorithm (validation seasons):")
    print(best[["algorithm", "run", "config", "val_reward_mean", "val_plots_alive"]]
          .to_string(index=False))

    print(f"\nFinal test on {args.test_episodes} unseen seasons...")
    final = final_comparison(algorithms, args.test_episodes, output_dir)
    print(final.round(2).to_string(index=False))

    print("\nStress test on harsher climates...")
    stress = stress_test(algorithms, max(5, args.test_episodes // 2), output_dir)
    print(stress.pivot(index="policy", columns="scenario", values="reward_mean")
          .round(1).to_string())

    baselines = final.set_index("policy")["reward_mean"]
    plot_learning_curves(results, best, baselines, output_dir=output_dir)
    plot_best_runs(best, baselines, output_dir=output_dir)
    plot_objectives(best, output_dir=output_dir)
    plot_final(final, output_dir)
    plot_stress(stress, output_dir)
    print(f"\nFigures saved to {os.path.join(output_dir, 'assets')}, "
          f"tables to {os.path.join(output_dir, 'logs')}.")


if __name__ == "__main__":
    main()
