"""Training loop for the policy-gradient agents (REINFORCE, A2C, PPO)."""

import numpy as np
import pandas as pd
import torch

from environment.wrappers import make_gym_env
from training.algorithms import PolicyGradientAgent, set_seed, transition_td
from training.evaluation import episode_metrics


def train_policy(algorithm, config, episodes=300, seed=0, verbose=True):
    set_seed(seed)
    env = make_gym_env()
    agent = PolicyGradientAgent(algorithm, **config)
    history, batch = [], []
    loss, entropy = np.nan, np.nan

    for ep in range(episodes):
        obs, info = env.reset(seed=seed * 100_000 + ep)
        done, total, steps = False, 0.0, []
        while not done:
            action, log_prob = agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            steps.append(transition_td(obs, action, reward, next_obs,
                                       terminated, truncated, log_prob))
            obs, total = next_obs, total + reward
            done = terminated or truncated

        batch.append(torch.stack(steps))           # one episode -> TensorDict [T]
        if len(batch) == agent.episodes_per_batch:
            loss, entropy = agent.update(batch)
            batch = []

        row = episode_metrics(total, info)
        row.update(episode=ep, loss=loss, entropy=entropy)
        history.append(row)
        if verbose and ep % 50 == 0:
            print(f"  {agent.name} ep {ep:4d} | reward {total:8.1f} | "
                  f"alive {info['plots_alive']}")

    return pd.DataFrame(history), agent
