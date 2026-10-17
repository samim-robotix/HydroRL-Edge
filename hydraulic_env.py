# --- hydraulic_env.py ---

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

class HydraulicActuatorEnv(gym.Env):
    def __init__(self):
        super(HydraulicActuatorEnv, self).__init__()
        self.m = 2.5
        self.dc = 0.04
        self.dr = 0.02
        self.fc = 51.1
        self.av = 2000
        self.cva = 3.46e-08
        self.cvb = 1.73e-08
        self.cla = 3.8e-09
        self.clb = 1.9e-09
        self.pp = 120e5
        self.k = 20000
        self.ac = math.pi * 0.25 * self.dc ** 2
        self.ar = math.pi * 0.25 * (self.dc ** 2 - self.dr ** 2)

        self.dt = 1e-3
        self.simulation_time = 0.5
        self.reset()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.observation_space = spaces.Box(
            low=np.array([0.0, -1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([0.2, 1.0, self.pp, self.pp], dtype=np.float32),
            dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.time = 0.0
        self.y = 0.1
        self.ydt = 0.0
        self.pcold = self.pp / 2
        self.prold = self.pp / 2
        self.rf = 0.9
        self.target = 0.15
        obs = np.array([self.y, self.ydt, self.pcold, self.prold], dtype=np.float32)
        return obs, {}

    def _get_obs(self):
        return np.array([self.y, self.ydt, self.pcold, self.prold], dtype=np.float32)

    def step(self, action):
        v = np.clip(action[0], -1.0, 1.0) * 10.0
        pp = self.pp

        if v >= 0.0:
            pc = (((self.cva * v + self.cla) * math.sqrt(pp - self.pcold) - self.ac * self.ydt) / self.cla) ** 2
            pr = ((self.clb * math.sqrt(pp - self.prold) + self.ar * self.ydt) / (self.cvb * v + self.clb)) ** 2
        else:
            pc = ((self.cla * math.sqrt(pp - self.pcold) - self.ac * self.ydt) / (self.cla - self.cva * v)) ** 2
            pr = (((self.clb - self.cvb * v) * math.sqrt(pp - self.prold) + self.ar * self.ydt) / self.clb) ** 2

        pc = np.clip(pc, 0, pp)
        pr = np.clip(pr, 0, pp)

        ff = self.fc * np.sign(self.ydt) + self.av * self.ydt
        fp = pc * self.ac - pr * self.ar
        fk = self.k * self.y
        fnet = fp - fk - ff
        yddt = fnet / self.m

        self.ydt += yddt * self.dt
        self.y += self.ydt * self.dt

        self.y = np.clip(self.y, 0.0, 0.2)
        if (self.y == 0.0 and self.ydt < 0) or (self.y == 0.2 and self.ydt > 0):
            self.ydt = 0.0

        self.time += self.dt
        self.pcold = self.rf * pc
        self.prold = self.rf * pr

        # Updated reward function
        error = self.y - self.target
        reward = - (100 * error ** 2 + 0.1 * self.ydt ** 2 + 0.01 * v ** 2)

        if abs(error) < 0.01:
            reward += 5.0

        if self.y >= 0.2:
            reward -= 10.0

        done = self.time >= self.simulation_time
        return self._get_obs(), reward, done, False, {}

    def render(self):
        pass