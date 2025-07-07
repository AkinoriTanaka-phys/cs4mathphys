import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

class Slots(gym.Env):
    ''' `num_slots` 台のスロットマシン ''' 
    def __init__(self, num_slots=3, init_seed=None):
        self.action_space = gym.spaces.Discrete(num_slots, seed=init_seed)
        self.observation_space = gym.spaces.Discrete(1)
        # k 個のスロットがそれぞれランダムな値の平均と分散を持つように初期化 
        rng = np.random.default_rng(init_seed)
        self.means = rng.uniform(-10, 10, size=(num_slots,)) 
        self.stds = rng.uniform(.1, 1, size=(num_slots,))

    def reset(self, seed=None, options={}):
        super().reset(seed=seed) 
        self.seed = seed
        self._internal_s = 0 # 内部状態 なし
        info = {"seed": seed}
        return self._internal_s, info

    def step(self, a):
        # ランダム処理には self.np_random を使う
        self._internal_s = 0 # 内部状態 なし
        r = self.np_random.normal(self.means[a], self.stds[a]) # p(r|a) からサンプリング 
        terminated, truncated, info = True, False, {"seed": self.seed}
        return self._internal_s, r, terminated, truncated, info

class ContinuumSlots(gym.Env):
    ''' $\mathbb{R}$ 台のスロットマシン
        `self.step(a)` で `r, done, info` が出力される'''             
    def __init__(self, init_seed=None):
        self.action_space = gym.spaces.Box(-np.inf, np.inf, seed=init_seed)
        self.observation_space = gym.spaces.Discrete(1)
        # $p(r|a) =$ 平均 $\sin(a+random)$ 標準偏差 0.1 のガウス分布とする
        rng = np.random.default_rng(init_seed)
        self.random = rng.uniform(-3, 3)

    def reset(self, seed=None, options={}):
        super().reset(seed=seed) 
        self.seed = seed
        self._internal_s = 0 # 内部状態 なし
        info = {"seed": seed}
        return self._internal_s, info

    def step(self, a):
        # ランダム処理には self.np_random を使う
        self._internal_s = 0 # 内部状態 なし
        r = self.np_random.normal(np.sin(a + self.random)[0], .1) # $p(r|a)$ からサンプリング
        terminated, truncated, info = True, False, {"seed": self.seed}
        return self._internal_s, r, terminated, truncated, info

# gym.register(
#     id="my_envs/slots-discrete",
#     entry_point=Slots,
# )
# gym.register(
#     id="my_envs/slots-continuum",
#     entry_point=ContinuumSlots,
# )

# To use the class definitions, you need to call the corresponding class by gym.make().
env_ = gym.make('CliffWalking-v1', render_mode="rgb_array")
env_ = gym.make('FrozenLake-v1', map_name="4x4", render_mode="rgb_array")

class CustomizedCliffWalking(gym.envs.toy_text.CliffWalkingEnv):
    def __init__(self, render_mode, max_episode_steps):
        super().__init__(render_mode=render_mode)
        self.max_episode_steps = max_episode_steps

    def reset(self, seed=None):
        s0, info_env = super().reset(seed=seed)
        self.t = 0
        return s0, info_env

    def step(self, action):
        s, r, terminated, truncated, info_env = super().step(action)
        self.t += 1
        if self.t >= self.max_episode_steps:
            truncated = True
        return s, r, terminated, truncated, info_env 

    def plot_action(self, Q: np.array, clim=None):
        plt.imshow(self.render())
        scale=60
        x = np.arange(12); y = np.arange(4)
        X, Y = np.meshgrid(x, y)
        X*=scale
        Y*=scale
        action2direction = {0: ("up",    [0  , .5*scale]), 
                            1: ("right", [.5*scale, 0]),
                            2: ("down",  [0  , -.5*scale]), 
                            3: ("left",  [-.5*scale, 0])}
        for a in range(4):
            direction, v = action2direction[a]
            plt.quiver(X+scale*.5, Y+scale*.5, v[0], v[1], Q[:, a].reshape(4, 12), 
                        width=.006,
                        scale_units='xy', scale=1,
                        cmap="Blues", alpha=.8,
                        clim=clim, 
                        )
        plt.colorbar(shrink=0.62)

class EasyCliffWalking(CustomizedCliffWalking):
     def step(self, a):
          s, r, terminated, truncated, info_env = super().step(a)
          if terminated:
                r += 2
          if r < -50:
                terminated = True
          return s, r, terminated, truncated, info_env

class CustomizedFrozenLake(gym.envs.toy_text.FrozenLakeEnv):
    def __init__(self, is_slippery, render_mode, max_episode_steps):
        super().__init__(is_slippery=is_slippery, render_mode=render_mode)
        self.max_episode_steps = max_episode_steps

    def reset(self, seed=None):
        s0, info_env = super().reset(seed=seed)
        self.t = 0
        return s0, info_env

    def step(self, action):
        s, r, terminated, truncated, info_env = super().step(action)
        self.t += 1
        if self.t >= self.max_episode_steps:
            truncated = True
        return s, r, terminated, truncated, info_env 

    def plot_action(self, Q: np.array, clim=None):
        plt.imshow(self.render())
        scale=64
        x = np.arange(4); y = np.arange(4)
        X, Y = np.meshgrid(x, y)
        X*=scale
        Y*=scale
        action2direction = {0: ("left",  [-.5*scale, 0]), 
                            1: ("down",  [0, -.5*scale]),
                            2: ("right", [.5*scale,  0]), 
                            3: ("up",    [0, .5*scale ])}
        for a in range(4):
            direction, v = action2direction[a]
            plt.quiver(X+scale*.5, Y+scale*.5, v[0], v[1], Q[:, a].reshape(4, 4), 
                        width=.006,
                        scale_units='xy', scale=1,
                        cmap="Blues", 
                        alpha=.8,
                        clim=clim
                        )
        plt.colorbar(shrink=0.99)

class EasyFrozenLake(CustomizedFrozenLake):
     def step(self, a):
          s, r, terminated, truncated, info_env = super().step(a)
          if terminated:
               if s==15:
                    r = 1
               else:
                    r = -1
          return s, r, terminated, truncated, info_env



#from pettingzoo import ParallelEnv
import functools

from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector, wrappers

class Count(AECEnv):
    def __init__(self, N_lose=13, N_action=3):
        self.observation_space = gym.spaces.Discrete(N_lose+1)
        self.action_space = gym.spaces.Discrete(N_action, start=1)
        self.possible_agents = ["player_" + str(r) for r in range(2)]
        # which player wins in the end?
        # formula from https://mcm-www.jwu.ac.jp/~mathphys/mejirosai/m/2023/3.aiki/images/count30.pdf
        if (N_lose-1)%(N_action+1)==0:
            self.theoretical_winner = "player_1"
        else: 
            self.theoretical_winner = "player_0"
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_space

    def render(self):
        print(self.n)

    def observe(self, agent):
        return np.array(self.observations[agent])

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random, self.np_random_seed = seeding.np_random(seed)
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.terminated = False
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.n = 0
        self.observations = {agent:self.n for agent in self.agents}
        self.num_moves = 0
        
        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        agent = self.agent_selection

        self.n += action

        for i in self.agents:
            self.observations[i] = self.n

        # lose
        if self.n >= self.observation_space.n-1:
            self.rewards[agent] = -1
            self.terminations[agent] = True  
            self.terminated = True  
            # winners
            for agt in self.agents:
                self.observations[agt] = self.observation_space.n-1
                if agt!=agent:
                    self.rewards[agt] = +1
                    self.terminations[agt] = True               

        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()