"""
Evaluation helpers: run trained agents greedily on held-out seasons,
compare against rule-based baselines, and stress-test on harsher climates.
"""

import numpy as np
import pandas as pd

from environment.custom_env import OBS_LAYOUT
from environment.wrappers import encode_taps, make_gym_env

EVAL_SEED_BASE = 900_000          # never used during training

# Harsher seasons the agents never saw during training
STRESS_SCENARIOS = {
    "normal": {},
    "drought (50% rain)": {"rain_scale": 0.5},
    "heatwave (+0.2 temp)": {"heat_shift": 0.2},
    "small reservoir (70%)": {"reservoir_capacity": 0.7},
}


def episode_metrics(total_reward, info):
    return {
        "reward": total_reward,
        "days": info["day"],
        "plots_alive": info["plots_alive"],
        "water_used": info["water_used"],
        "time_in_band": info["time_in_band"],
        "mean_health": info["mean_health"],
        "switches": info["switches"],
    }


def run_episode(policy, seed, env=None, **env_kwargs):
    """policy(obs) -> Discrete action index."""
    env = env or make_gym_env(**env_kwargs)
    obs, info = env.reset(seed=seed)
    done, total = False, 0.0
    while not done:
        obs, reward, terminated, truncated, info = env.step(policy(obs))
        total += reward
        done = terminated or truncated
    return episode_metrics(total, info)


def evaluate(policy, n_episodes=30, seed_base=EVAL_SEED_BASE, **env_kwargs):
    env = make_gym_env(**env_kwargs)
    rows = [run_episode(policy, seed_base + i, env=env) for i in range(n_episodes)]
    return pd.DataFrame(rows)


def agent_policy(agent, greedy=None):
    """DQN is evaluated greedily (argmax Q). Policy-gradient agents are
    evaluated by sampling from their learned stochastic policy: they learn to
    pulse taps ON with a moderate probability, and taking the single most
    likely action would turn that into "never irrigate"."""
    if greedy is None:
        greedy = agent.name == "DQN"

    def policy(obs):
        action = agent.act(obs, greedy=greedy)
        return action[0] if isinstance(action, tuple) else action
    return policy


# ---------------------------------------------------------------------------
# Rule-based baselines (what a farmer's timer or simple sensor rule would do)
# ---------------------------------------------------------------------------
def baseline_policies():
    m = OBS_LAYOUT["moisture"]
    rng = np.random.default_rng(0)
    return {
        "Always OFF": lambda obs: 0,
        "Random": lambda obs: int(rng.integers(512)),
        "Threshold rule (<0.45)": lambda obs: encode_taps(obs[m] < 0.45),
        "Threshold + forecast": lambda obs: encode_taps(
            (obs[m] < 0.45) & ~((obs[OBS_LAYOUT["rain_forecast"]] > 0.6) & (obs[m] > 0.38))),
    }


def summarise(df):
    return {
        "reward_mean": df["reward"].mean(),
        "reward_std": df["reward"].std(),
        "plots_alive": df["plots_alive"].mean(),
        "water_used": df["water_used"].mean(),
        "time_in_band": df["time_in_band"].mean(),
        "switches": df["switches"].mean(),
        "full_season_rate": (df["days"] >= 120).mean(),
    }
