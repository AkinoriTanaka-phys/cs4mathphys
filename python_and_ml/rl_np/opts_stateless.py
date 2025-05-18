import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from policies_stateless import get_init_discrete_array

class Estimator:
    def __init__(self, in_shape: tuple, out_shape: tuple):
        self.params = np.zeros(shape=in_shape+out_shape)
        self.in_shape = in_shape
        self.out_shape = out_shape    

class MonteCarloEstimator(Estimator):
    def __init__(self, in_shape, out_shape):
        super().__init__(in_shape, out_shape)
        self.reset()

    def reset(self):
        ''' $\hat{E}(a)$ の推定値をリセットする '''
        self.params = np.zeros_like(self.params)
        self.N_visit = np.zeros(self.in_shape)

    def mc_update(self, f, a=None):
        ''' 新たな f が与えられた時に $\hat{E}(a)$ の値を更新する '''
        self.N_visit[a] += 1
        assert np.shape(f)==self.out_shape, f"f.shape is expected to be {self.out_shape}, but {np.shape(f)}."
        self.params[a] += (f - self.params[a])/self.N_visit[a]

class Optimizer:
    def __init__(self, policy):
        self.policy = policy
        
    def test(self):
        env = self.policy.env
        policy = self.policy
        rs = []
        for episode in range(10):
            a, info_agt = policy.sample()
            s, r, terminated, truncated, info_env = env.step(a)
            rs.append(r)
            episode_over = terminated or truncated
        self.training_history["value"].append(np.mean(rs))

    def plot_result(self):
        plt.figure(figsize=(13, 5))
        for n, key in enumerate(self.training_history.keys()):
            plt.subplot(1, len(self.training_history), n+1)
            plt.xlabel("# policy updates")
            plt.plot(self.training_history[key], label=key)
            plt.legend()

class MC_Optimizer(Optimizer):
    def __init__(self, policy, decay_r=0.99):
        super().__init__(policy)
        self.decay_r = decay_r
        self.training_history = {"epsilon"        : [], 
                                 "value"          : [], 
                                 "estimated_value": []}
        self.Q_estimator = MonteCarloEstimator(in_shape=policy.params.shape, out_shape=())

    def reset_record(self):
        ''' **1エピソードあたり** のサンプルを貯めておく辞書を初期化する '''
        self.sample = {}
    
    def record(self, **data):
        ''' サンプルデータを辞書に記録する '''
        for key in data.keys():
            self.sample[key] = data[key]

    def test(self):
        self.training_history["epsilon"].append(self.policy.epsilon)
        super().test()
        self.training_history["estimated_value"].append(np.sum(self.Q_estimator.params*self.policy.p()))

    def mc_update(self):
        ''' Qのモンテカルロ更新 '''
        self.Q_estimator.mc_update(a=self.sample["action"], 
                                   f=self.sample["reward"])

    def policy_update(self):
        ''' 方策の更新 '''
        self.policy.params = self.Q_estimator.params     # 方策のパラメータを現在のQの推定値にする
        self.policy.epsilon *= self.decay_r    # $\epsilon$ を少し小さくする

class ReinforceOptimizer_NoBaseline(Optimizer):
    def __init__(self, policy, lr=1e-1):
        super().__init__(policy)
        self.training_history = {"value": []}
        self.lr = lr
        # 内部に $\nabla J$ の推定器を持ち
        #     - それを更新: mc_update(self)
        #     - 現在の推定値を方策に反映: policy_update(self)
        self.policy_grad_estimator = MonteCarloEstimator(in_shape=(), out_shape=policy.params.shape)

    def reset_record(self):
        ''' **1エピソードあたり** のサンプルを貯めておく辞書を初期化する '''
        self.sample = {}
    
    def record(self, **data):
        ''' サンプルデータを辞書に記録する '''
        for key in data.keys():
            self.sample[key] = data[key]

    def mc_update(self):
        ''' $nabla J$のモンテカルロ更新 '''
        grad_log_p_a = self.policy.grad_log_p(self.sample["action"]) # $\nabla_\theta \log \pi_\theta(a)$
        grad = self.sample["reward"]*grad_log_p_a
        self.policy_grad_estimator.mc_update(f=grad)

    def policy_update(self):
        ''' 方策の更新 '''
        self.policy.params += self.policy_grad_estimator.params*self.lr
        self.policy_grad_estimator.reset()

class ReinforceOptimizer(ReinforceOptimizer_NoBaseline):
    def __init__(self, policy, lr=1e-1):
        super().__init__(policy, lr)
        self.training_history = {"value"           : [], 
                                 "estimated_value" : []}
        # ベースライン用の推定器
        self.baseline = MonteCarloEstimator(in_shape=(), out_shape=())

    def test(self):
        ''' せっかくなので [ベースラインの値＝推定した方策価値] もテストログに入れる '''
        super().test()
        self.training_history["estimated_value"].append(self.baseline.params)

    def mc_update(self):
        ''' 方策勾配のモンテカルロ更新 '''
        # advantage のサンプル近似
        A = self.sample["reward"] - self.baseline.params
        # ベースライン推定値の更新
        self.baseline.mc_update(f = self.sample["reward"])
        # 方策勾配の推定値の更新
        grad_log_p_a = self.policy.grad_log_p(self.sample["action"])
        grad = A*grad_log_p_a
        self.policy_grad_estimator.mc_update(f=grad)