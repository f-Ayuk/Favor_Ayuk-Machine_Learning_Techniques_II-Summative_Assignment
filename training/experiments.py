"""
Hyperparameter experiments: 10 configurations per algorithm.

Each configuration only changes parameters the algorithm actually uses.
Every run trains on the same sequence of seasons (same seeds) for the same
number of episodes, so the comparison is fair. Each run is then scored on
20 validation seasons the agent never trained on; the best run per
algorithm is kept for the final test.
"""

import os
import time
from multiprocessing import Pool

import pandas as pd

from training.algorithms import save_agent
from training.dqn_training import train_dqn
from training.evaluation import agent_policy, evaluate, summarise
from training.pg_training import train_policy

VAL_SEED_BASE = 800_000

# ---------------------------------------------------------------------------
# Hyperparameter grids (run 1 is the baseline, the rest change 1-2 things)
# ---------------------------------------------------------------------------
DQN_CONFIGS = [
    {},                                                  # 1 defaults
    {"lr": 3e-4},                                        # 2 smaller learning rate
    {"lr": 3e-3},                                        # 3 larger learning rate
    {"gamma": 0.95},                                     # 4 short-sighted
    {"batch_size": 128, "buffer_size": 100_000},         # 5 bigger batch/buffer
    {"target_update": 100},                              # 6 frequent target sync
    {"target_update": 2000},                             # 7 rare target sync
    {"eps_decay_steps": 50_000, "eps_end": 0.02},        # 8 longer exploration
    {"double_dqn": False},                               # 9 vanilla DQN target
    {"head": "flat"},                                    # 10 no tap structure (512 outputs)
]

REINFORCE_CONFIGS = [
    {},                                                  # 1 defaults (with baseline)
    {"lr": 1e-4},
    {"lr": 1e-3},
    {"gamma": 0.95},
    {"gamma": 0.999},
    {"use_baseline": False},                             # 6 vanilla REINFORCE
    {"episodes_per_batch": 1},
    {"episodes_per_batch": 8},
    {"hidden": 64},
    {"head": "factored"},                                # 10 no weight sharing
]

A2C_CONFIGS = [
    {},
    {"lr": 1e-4},
    {"lr": 1e-3},
    {"gamma": 0.95},
    {"gae_lambda": 0.8},
    {"gae_lambda": 1.0},
    {"entropy_coef": 0.0},
    {"entropy_coef": 0.05},
    {"critic_coef": 1.0, "episodes_per_batch": 8},
    {"head": "factored"},
]

PPO_CONFIGS = [
    {},
    {"lr": 1e-4},
    {"lr": 1e-3},
    {"gamma": 0.95},
    {"clip_epsilon": 0.1},
    {"clip_epsilon": 0.3},
    {"ppo_epochs": 10},
    {"entropy_coef": 0.0},
    {"episodes_per_batch": 8, "minibatch_size": 512},
    {"head": "factored"},
]

CONFIGS = {"DQN": DQN_CONFIGS, "REINFORCE": REINFORCE_CONFIGS,
           "A2C": A2C_CONFIGS, "PPO": PPO_CONFIGS}


def _run_one(task):
    algorithm, run, config, episodes, seed, output_dir = task
    start = time.time()
    if algorithm == "DQN":
        history, agent = train_dqn(config, episodes=episodes, seed=seed, verbose=False)
    else:
        history, agent = train_policy(algorithm, config, episodes=episodes,
                                      seed=seed, verbose=False)

    history["algorithm"], history["run"] = algorithm, run
    log_dir = os.path.join(output_dir, "logs")
    model_dir = os.path.join(output_dir, "models", algorithm.lower())
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    history.to_csv(os.path.join(log_dir, f"{algorithm.lower()}_run_{run}.csv"), index=False)
    save_agent(agent, config, os.path.join(model_dir, f"run_{run}.pt"))

    val = summarise(evaluate(agent_policy(agent), n_episodes=20, seed_base=VAL_SEED_BASE))
    rolling = history["reward"].rolling(50, min_periods=1).mean()
    converged = int((rolling >= 0.9 * rolling.max()).idxmax())

    print(f"{algorithm:9s} run {run:2d} | val reward {val['reward_mean']:7.1f} | "
          f"alive {val['plots_alive']:.1f} | {time.time() - start:5.0f}s | {config}",
          flush=True)
    return {"algorithm": algorithm, "run": run, "config": str(config or "defaults"),
            **{f"val_{k}": v for k, v in val.items()},
            "train_reward_last100": history["reward"].tail(100).mean(),
            "episodes_to_90pct": converged,
            "train_seconds": time.time() - start}


def run_experiments(algorithms=("DQN", "REINFORCE", "A2C", "PPO"), episodes=600,
                    seed=0, jobs=1, runs=None, output_dir="."):
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    tasks = []
    for algorithm in algorithms:
        configs = CONFIGS[algorithm][:runs] if runs else CONFIGS[algorithm]
        for run, config in enumerate(configs, start=1):
            tasks.append((algorithm, run, config, episodes, seed, output_dir))

    # DQN runs are the slowest, so start them first when running in parallel
    tasks.sort(key=lambda t: t[0] != "DQN")
    if jobs > 1:
        with Pool(jobs) as pool:
            rows = pool.map(_run_one, tasks, chunksize=1)
    else:
        rows = [_run_one(t) for t in tasks]

    results = pd.DataFrame(rows).sort_values(["algorithm", "run"])
    results.to_csv(os.path.join(output_dir, "logs", "hyperparameter_results.csv"), index=False)
    return results
