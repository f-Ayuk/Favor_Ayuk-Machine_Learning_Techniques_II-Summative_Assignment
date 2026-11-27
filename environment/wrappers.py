"""
Wrappers that give every algorithm exactly the same interface.

The farm natively takes MultiBinary(9) actions (one ON/OFF bit per tap).
DQN needs a single discrete action, so all four algorithms see the same
Discrete(512) action space: action k is the 9-bit pattern of tap states
(e.g. 0 = all taps OFF, 511 = all taps ON). This keeps the comparison fair.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environment.custom_env import SmartIrrigationEnv


class TapPatternAction(gym.ActionWrapper):
    """Discrete(2**n) -> MultiBinary(n)."""

    def __init__(self, env):
        super().__init__(env)
        self.n = env.action_space.n
        self.action_space = spaces.Discrete(2 ** self.n)
        self._bits = 1 << np.arange(self.n)

    def action(self, action):
        return ((int(action) & self._bits) > 0).astype(np.int8)


def encode_taps(taps):
    """MultiBinary tap pattern -> Discrete action index (for baselines)."""
    return int(np.sum(np.asarray(taps, dtype=int) << np.arange(len(taps))))


def make_gym_env(render_mode=None, **env_kwargs):
    """Plain Gymnasium env with the shared Discrete(512) action space."""
    return TapPatternAction(SmartIrrigationEnv(render_mode=render_mode, **env_kwargs))
