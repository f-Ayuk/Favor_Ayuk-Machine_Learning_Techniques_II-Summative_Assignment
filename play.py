"""
Watch an agent manage one growing season in the MuJoCo farm.

    python play.py --agent ppo                 # live MuJoCo viewer (best PPO model)
    python play.py --agent random              # random actions (for the demo video)
    python play.py --agent dqn --record videos/dqn_season.mp4   # save a video instead
    python play.py --agent models/a2c/run_3.pt                  # any saved checkpoint

On macOS the live viewer must be started with `mjpython play.py ...`.
Each step prints what the agent did, so the terminal output can be shown
alongside the video.
"""

import argparse
import os
import time

import numpy as np

from environment.wrappers import make_gym_env
from training.algorithms import load_agent
from training.evaluation import agent_policy


def get_policy(name, seed):
    if name == "random":
        rng = np.random.default_rng(seed)
        return lambda obs: int(rng.integers(512)), "Random"
    path = name if name.endswith(".pt") else f"models/{name.lower()}/best.pt"
    agent = load_agent(path)
    return agent_policy(agent), agent.name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="dqn",
                        help="dqn | reinforce | a2c | ppo | random | path/to/model.pt")
    parser.add_argument("--seed", type=int, default=900_000)
    parser.add_argument("--record", default=None, help="save an .mp4 instead of opening the viewer")
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()

    policy, label = get_policy(args.agent, args.seed)
    env = make_gym_env(render_mode="rgb_array" if args.record else "human")
    farm = env.unwrapped
    obs, info = env.reset(seed=args.seed)

    frames = []
    total, done = 0.0, False
    print(f"\n{label} agent | season seed {args.seed}\n")
    while not done:
        action = policy(obs)
        taps = env.action(action)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        done = terminated or truncated

        grid = "".join("■" if t else "·" for t in taps)
        print(f"day {info['day']:3d} | taps {grid[:3]} {grid[3:6]} {grid[6:]} | "
              f"rain {info['rain']:.2f} | reservoir {info['reservoir']:5.1%} | "
              f"alive {info['plots_alive']} | reward {reward:6.2f}")

        if args.record:
            frames.append(farm.render())
        else:
            time.sleep(1.0 / args.fps)

    print(f"\nSeason total reward {total:.1f} | plots alive {info['plots_alive']}/9 | "
          f"water used {info['water_used']:.0%} | time in healthy band "
          f"{info['time_in_band']:.0%}")

    if args.record:
        import imageio.v2 as imageio
        os.makedirs(os.path.dirname(args.record) or ".", exist_ok=True)
        imageio.mimsave(args.record, frames, fps=args.fps)
        print(f"Saved {args.record}")
    env.close()


if __name__ == "__main__":
    main()
