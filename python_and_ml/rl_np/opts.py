import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym


class Estimator:
    def __init__(self, in_shape: tuple, out_shape: tuple):
        ''' in_shape should be 
                - (dim(s), dim(a)) or  
                - (dim(s),) or
                - ()                     '''
        self.params = np.zeros(shape=in_shape+out_shape)
        self.in_shape = in_shape
        self.out_shape = out_shape    

class MonteCarloEstimator(Estimator):
    def __init__(self, in_shape, out_shape):
        super().__init__(in_shape, out_shape)
        self.reset()

    def reset(self):
        ''' $\hat{E}(s, a)$ の推定値をリセットする '''
        self.params = np.zeros_like(self.params)
        self.N_visit = np.zeros(self.in_shape)

    def mc_update(self, f, s=None, a=None):
        ''' 新たな f が与えられた時に $\hat{E}(s, a)$ の値を更新する '''
        self.N_visit[s, a] += 1
        assert np.shape(f)==self.out_shape, f"f.shape is expected to be {self.out_shape}, but {np.shape(f)}."
        self.params[s, a] += (f - self.params[s, a])/self.N_visit[s, a]

class Optimizer:
    ''' 訓練手法のプロトタイプ '''
    def __init__(self, policy, gamma=0.99):
        self.policy = policy
        self.gamma = gamma

    def reset_record(self):
        ''' **1エピソードあたり** のサンプルを貯めておく辞書を初期化する '''
        self.sample_history = {"state_action": [], "reward": [None]}
    
    def record(self, **data):
        ''' サンプルデータを辞書に記録する '''
        for key in data.keys():
            self.sample_history[key].append(data[key])

    def test(self, T_upper=300):
        env = self.policy.env
        policy = self.policy
        Js = []
        for episode in range(10):
            J = 0
            episode_over = False
            s, info_env = env.reset()
            t = 0
            s0 = s
            self.reset_record()
            while not episode_over:
                s_ = s
                a_, info_agt = policy.sample(s_)
                s, r, terminated, truncated, info_env = env.step(a_)
                self.record(state_action=(s_, a_), reward=r)

                t += 1
                if t > T_upper:
                    terminated = True
                episode_over = terminated or truncated
            gammas = self.gamma**np.arange(t)
            J = np.sum(gammas*self.sample_history["reward"][1:])
            Js.append(J)
        self.training_history["value"].append(np.mean(Js))

    def plot_result(self):
        plt.figure(figsize=(13, 5))
        for n, key in enumerate(self.training_history.keys()):
            plt.subplot(1, len(self.training_history), n+1)
            plt.xlabel("# policy updates")
            plt.plot(self.training_history[key], label=key)
            plt.legend()
        #plt.show()

class MC_Optimizer(Optimizer):
    def __init__(self, policy, gamma=0.99, decay_r=0.99):
        super().__init__(policy, gamma)
        self.decay_r = decay_r # epsilon を更新のたびに何倍するか
        self.training_history = {"epsilon"        : [], 
                                 "value"          : [], 
                                 "estimated_value": []}
        # 以下が $Q(a, a)$ の推定をするオブジェクト
        self.Q_estimator = MonteCarloEstimator(
                            in_shape=(policy.observation_space.n, policy.action_space.n),
                            out_shape=())
    
    def test(self):
        ''' 学習曲線を書くためのログ用 '''
        super().test()
        self.training_history["epsilon"].append(self.policy.epsilon)
        s0, _ = self.policy.env.reset()
        self.training_history["estimated_value"].append(np.sum(self.Q_estimator.params[s0]*self.policy.p(s0)))

    def mc_update(self, t_final):
        ''' $\hat{Q}$ のモンテカルロ更新 '''
        # サンプルの逆時間方向にモンテカルロ計算してゆく
        G = 0
        for t_rev in np.arange(t_final)[::-1]: # 時間を逆方向にループ
            G = self.gamma*G + self.sample_history["reward"][t_rev + 1]
            s, a = self.sample_history["state_action"][t_rev]
            self.Q_estimator.mc_update(s=s, a=a, f=G) # $Q(s, a)$の推定器を更新

    def policy_update(self, a=None):
        ''' 方策の更新 '''
        self.policy.params = self.Q_estimator.params     # 方策のパラメータを現在のQの推定値にする
        self.policy.epsilon *= self.decay_r              # $\epsilon$ を少し小さくする
        #self.Q_estimator.reset()  # リセットしない方が良い

class ReinforceOptimizer(Optimizer):
    def __init__(self, policy, gamma=1e-1, lr=1e-1, is_baseline_applied=True):
        super().__init__(policy, gamma)
        self.training_history = {"value"           : [], 
                                 "estimated_value" : []}
        self.lr = lr
        self.policy_grad_estimator = MonteCarloEstimator(
                            in_shape=(),
                            out_shape=(policy.observation_space.n, policy.action_space.n))
        
        self.is_baseline_applied = is_baseline_applied
        self.baseline = MonteCarloEstimator(
                            in_shape=(policy.observation_space.n,),
                            out_shape=())

    def test(self):
        super().test()
        s0, _ = self.policy.env.reset()
        self.training_history["estimated_value"].append(self.baseline.params[s0])

    def mc_update(self, t_final):
        ''' 方策勾配 のモンテカルロ更新 '''
        # サンプルの逆時間方向にモンテカルロ計算してゆく
        G = 0
        grad = np.zeros_like(self.policy.params)
        for t_rev in np.arange(t_final)[::-1]: # 時間を逆方向にループ
            G = self.gamma*G + self.sample_history["reward"][t_rev + 1]
            s, a = self.sample_history["state_action"][t_rev]
            # baseline Monte-Carlo update
            self.baseline.mc_update(s=s, f=G)
            # policy grad sum
            advantage = G - self.is_baseline_applied*self.baseline.params[s]
            grad_log_p = self.policy.grad_log_p(s, a)
            grad[s] += (self.gamma**t_rev)*advantage*grad_log_p
        # policy grad Monte-Carlo update
        self.policy_grad_estimator.mc_update(f=grad)

    def policy_update(self):
        ''' 方策の更新 '''
        self.policy.params += self.policy_grad_estimator.params*self.lr
        self.policy_grad_estimator.reset()


class TemporalDifferenceOptimizer(Optimizer):
    pass



class SarsaOptimizer(TemporalDifferenceOptimizer):
    def __init__(self, policy, gamma = 0.99, decay_r=0.99, lr=.1):
        super().__init__(policy, gamma)
        self.training_history = {"epsilon"        : [], 
                                 "value"          : [], 
                                 "estimated_value": []}
        self.decay_r = decay_r
        self.lr = lr

    def test(self):
        super().test()
        self.training_history["epsilon"].append(self.policy.epsilon)
        s0, _ = self.policy.env.reset()
        self.training_history["estimated_value"].append(np.sum(self.policy.params[s0]*self.policy.p(s0)))

    def policy_update(self, s_, a_, r, s, t=0):
        a, _ = self.policy.sample(s)           # 更新用に環境とは相互作用しない 行動サンプルを出すやり方
        # 以下は行動のスタートが0からでは無い場合の処理です
        a_ -= self.policy.action_space.start
        a  -= self.policy.action_space.start
        # TD更新部分
        TD_error = self.policy.params[s_, a_] - (r + self.gamma*self.policy.params[s, a])
        self.policy.params[s_, a_] -= self.lr*TD_error

        self.policy.epsilon *= self.decay_r

class QlearningOptimizer(TemporalDifferenceOptimizer):
    def __init__(self, policy, gamma = 0.99, decay_r=0.99, lr=.1):
        super().__init__(policy, gamma)
        self.training_history = {"epsilon"        : [], 
                                 "value"          : [], 
                                 "estimated_value": []}
        self.decay_r = decay_r
        self.lr = lr

    def test(self):
        super().test()
        self.training_history["epsilon"].append(self.policy.epsilon)
        s0, _ = self.policy.env.reset()
        self.training_history["estimated_value"].append(np.sum(self.policy.params[s0]*self.policy.p(s0)))

    def policy_update(self, s_, a_, r, s, t=0):
        a_ -= self.policy.action_space.start
        TD_error = self.policy.params[s_, a_] \
                    - (r + self.gamma*np.max(self.policy.params[s]))
        self.policy.params[s_, a_] -= self.lr*TD_error

        self.policy.epsilon *= self.decay_r

class Critic():
    def __init__(self, policy):
        self.params = np.zeros_like(policy.params[:, 0])  

class ActorCriticOptimizer(TemporalDifferenceOptimizer):
    def __init__(self, policy, critic, gamma=0.99, lr_c=1e-3, lr_a=1e-3):
        super().__init__(policy, gamma)
        self.V = critic.params
        self.training_history = {"value"           : [], 
                                 "estimated_value" : []}
        self.lr_c = lr_c
        self.lr_a = lr_a

    def test(self):
        super().test()
        s0, _ = self.policy.env.reset()
        self.training_history["estimated_value"].append(self.V[s0])

    def policy_update(self, s_, a_, r, s, t=0):
        #print(s, a)
        TD_error = self.V[s_] - (r + self.gamma*self.V[s])

        # critic update
        self.V[s_] -= self.lr_c*TD_error

        # actor
        A = -TD_error
        grad_log_p = self.policy.grad_log_p(s_, a_)
        grad = A*grad_log_p
        self.policy.params[s_] += self.lr_a*grad*self.gamma**t
