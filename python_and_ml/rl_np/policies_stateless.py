import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

class Agent:
    ''' エージェントクラス、ノートを通して使う '''
    def __init__(self, policy, critic=None, model=None):
        ''' 初期化で方策オブジェクトを読み込む '''
        self.policy = policy
        self.critic = critic
        self.model = model        

    def planning(self):
        self.model.step()

    def step(self, s=None):
        if s is None:
            return self.policy.sample()
        else:
            return self.policy.sample(s)


class Policy:
    ''' 方策のプロトタイプ '''
    def sample():
        ''' このメソッドを定義していればなんでもOK '''
        pass

class YourPolicy(Policy):
    ''' 0, 1, ..., num_slots-1 から 1 つユーザーが選んで返す方策
        返り値は (a: int/[float], info: dict) のタプル '''
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.params = None

    def sample(self):
        while True:
            try:
                a = int(input(f"Your turn: type a value in {self.action_space}\n"))
            except ValueError:
                print(f"\tPlease type a value in {self.action_space}")
                continue
            if a >= self.action_space.n:
                print(f"\tPlease type a value in {self.action_space}")
            else:
                info = {}
                return a, info

class RandomPolicy(Policy):
    ''' env.action_space から 1 つ等確率で選んで返す方策
        sample() の返り値は (a: int/[float], info: dict) のタプル '''
    def __init__(self, env, seed=None):
        self.env = env
        self.action_space = env.action_space
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.params = None

    def sample(self):
        a = self.rng.integers(self.action_space.start, (self.action_space.n))
        info = {"seed": self.seed}
        return a, info

def get_init_discrete_array(params, discrete_space):
    if params is None:
        return np.zeros(discrete_space.n)
    else:
        return params

def uniform_sample_from_argmax(params: np.ndarray, rng): # params.shape = (n,) 
    max_params = np.max(params)
    argmax_params = np.where(params == max_params)[0]
    return rng.choice(argmax_params)

class EpsilonGreedyPolicy(Policy):
    ''' `f` の値と `epsilon>0` に よる epsilon-greedy 方策
        sample() の返り値は (a: int, info: dict) のタプル''' 
    def __init__(self, env, epsilon: float, params: np.array = None, seed=None):
        self.env = env
        self.action_space = env.action_space
        self.epsilon = epsilon
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.params = get_init_discrete_array(params, env.action_space)

    def p(self):
        max_params = np.max(self.params)
        p_argmax = (self.params == max_params)
        p_argmax = p_argmax/np.sum(p_argmax)
        return (1-self.epsilon)*p_argmax + self.epsilon*np.ones_like(self.params)/self.action_space.n

    def is_greedy(self):
        return self.rng.uniform(0, 1) < (1-self.epsilon) # 確率 (1 − ε) で True を返す

    def sample(self):
        if self.is_greedy():
            a = uniform_sample_from_argmax(self.params, self.rng)
            info = {"is_greedy": True, "epsilon":self.epsilon, "seed": self.seed}
        else:
            a = self.rng.choice(self.action_space.n)
            info = {"is_greedy": False, "epsilon":self.epsilon, "seed": self.seed}
        return a, info


def softmax(logits: np.array): # logits.shape = (n,)
    weights = np.exp(logits)
    return weights/np.sum(weights, axis=-1)
    
class SoftmaxPolicy(Policy):
    ''' `T>0` と `f` の値 に よる softmax 方策
        sample() の返り値は (a: int, info: dict) のタプル'''
    def __init__(self, env, epsilon: float = 1, params: np.array = None, seed=None):
        self.env = env
        self.action_space = env.action_space
        self.epsilon = epsilon
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.params = get_init_discrete_array(params, env.action_space)

    def p(self, T=1):
        return softmax(self.params/T)

    def grad_log_p(self, a, T=1):
        grad = np.array([((a==b) - self.p()[a])/T for b in range(self.action_space.n)])
        return grad
        
    def sample(self):
        a = self.rng.choice(np.arange(self.action_space.n), p=self.p(T=self.epsilon))
        info = {"epsilon(temperature)":self.epsilon, "seed": self.seed}
        return a, info