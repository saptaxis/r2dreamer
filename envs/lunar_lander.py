"""LunarLander-v3 environment wrapper for r2dreamer.

Wraps gymnasium's LunarLander-v3 to conform to r2dreamer's env interface:
- reset() returns obs_dict (single dict with is_first/is_last/is_terminal)
- step(action) returns (obs_dict, reward, done, info) 4-tuple
- observation_space is a Dict space
- action_space is Discrete (OneHotAction wrapper applied by factory)

The 8D observation vector:
  0: x (horizontal position)      4: angle (radians)
  1: y (vertical position)        5: angular_vel
  2: vx (horizontal velocity)     6: left_leg_contact (binary)
  3: vy (vertical velocity)       7: right_leg_contact (binary)
"""

import gymnasium
import numpy as np


class LunarLander:
    """LunarLander-v3 wrapper for r2dreamer.

    Returns observations as a dict with a single 'state' key containing
    the 8D float32 observation vector. The encoder's mlp_keys='.*' regex
    matches this key and routes it through the MLP encoder path.
    """

    metadata = {}

    def __init__(self, task="default", size=(64, 64), seed=0):
        # task: reserved for future physics config variants (e.g. 'heavy_gravity')
        # size: unused for state-vector mode, required for interface compat
        self._env = gymnasium.make("LunarLander-v3")
        self._seed = seed
        self.reward_range = [-np.inf, np.inf]

    @property
    def observation_space(self):
        # Single 'state' key with the full 8D observation vector.
        # The MLP encoder concatenates all mlp_keys-matched values.
        spaces = {"state": self._env.observation_space}
        return gymnasium.spaces.Dict(spaces)

    @property
    def action_space(self):
        # Discrete(4): 0=noop, 1=left engine, 2=main engine, 3=right engine
        # OneHotAction wrapper (applied by make_env factory) converts to Box one-hot
        return self._env.action_space

    def step(self, action):
        # action is an int (OneHotAction converts one-hot back to index before calling)
        obs, reward, terminated, truncated, info = self._env.step(action)
        done = terminated or truncated
        obs_dict = {
            "state": obs.astype(np.float32),
            "is_first": False,
            "is_last": done,
            "is_terminal": terminated,  # true terminal (not just truncation)
        }
        return obs_dict, float(reward), done, info

    def reset(self, **kwargs):
        obs, info = self._env.reset(seed=self._seed)
        # Only use seed on first reset
        self._seed = None
        return {
            "state": obs.astype(np.float32),
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
        }

    def close(self):
        self._env.close()
