import numpy as np
from collections import deque

import env_wrapper


class MarioEnv(object):
    def __init__(self, world, stage):
        from nes_py.wrappers import JoypadSpace
        import gym_super_mario_bros
        from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
        env = gym_super_mario_bros.make(f"SuperMarioBros-{world}-{stage}-v0")
        env = JoypadSpace(env, COMPLEX_MOVEMENT)
        env = env_wrapper.SkipEnv(env, [])
        env = env_wrapper.WarpFrame(env, width=84, height=84)
        env = env_wrapper.MaxAndSkipEnv(env, 4)
        env = env_wrapper.FrameStack(env, 4)
        self._env = env

    def _preprocess(self, obs):
        obs = np.array(obs, dtype=np.float32) / 255.0
        obs = obs.transpose((2, 0, 1))
        return obs

    def step(self, action):
        obs, _, done, info = self._env.step(action)
        # print(done, info)
        obs = self._preprocess(obs)
        return obs, done, info

    def reset(self):
        return self._preprocess(self._env.reset())

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()
