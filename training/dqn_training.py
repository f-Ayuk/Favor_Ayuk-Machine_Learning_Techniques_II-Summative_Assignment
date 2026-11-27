"""Training loop for the value-based agent (DQN)."""

import numpy as np
import pandas as pd

from environment.wrappers import make_gym_env
from training.algorithms import DQNAgent, set_seed
from training.evaluation import episode_metrics


def train_dqn(config, episodes=300, seed=0, verbose=True):
    set_seed(seed)
    env = make_gym_env()
    agent = DQNAgent(**config)
    history = []

    for ep in range(episodes):
        # Training seasons use the same seeds for every run (fair comparison).
        obs, info = env.reset(seed=seed * 100_000 + ep)
        done, total, losses = False, 0.0, []
        while not done:
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            loss = agent.observe(obs, action, reward, next_obs, terminated, truncated)
            if loss is not None:
                losses.append(loss)
            obs, total = next_obs, total + reward
            done = terminated or truncated

        row = episode_metrics(total, info)
        row.update(episode=ep, loss=np.mean(losses) if losses else np.nan,
                   epsilon=agent.epsilon)
        history.append(row)
        if verbose and ep % 50 == 0:
            print(f"  DQN ep {ep:4d} | reward {total:8.1f} | alive {info['plots_alive']} "
                  f"| eps {agent.epsilon:.2f}")

    return pd.DataFrame(history), agent
