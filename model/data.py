import random

import numpy as np

from utils.delta import calculate_delta


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


class Loop:
    def __init__(self, N, K, T, dis="beta", cate="normal", seed=None):
        set_seed(seed)
        self.N = N
        self.K = K
        self.T = T
        self.dis = dis

        while True:
            if self.dis == "beta":
                self.alpha = np.random.rand(K) * 5
                self.beta = np.random.rand(K) * 5
                self.rewards = np.random.beta(self.alpha, self.beta, (T, K))
                self.mu = self.alpha / (self.alpha + self.beta)
            elif self.dis == "bernoulli":
                self.mu = np.random.rand(K)
                self.rewards = np.random.binomial(1, self.mu, (T, K))

            if cate == "normal":
                self.weights = np.random.rand(self.N, self.K)
            elif cate == "same":
                self.weights = np.random.rand(self.N)
                self.weights = np.tile(self.weights, [self.K, 1]).T
            
            print(self.weights)

            # delta_pne, delta_nopne, self.delta = calculate_delta(self.weights, self.mu)
            delta_pne, delta_nopne, self.delta = 1, 1, 1
            print("delta:", delta_pne, delta_nopne, self.delta)
            if delta_pne < 500:
                break

    def pull(self, choices, t):
        weight = np.zeros(self.K)
        weight_choices = []
        for i, choice in enumerate(choices):
            weight[choice] += self.weights[i][choice]
            weight_choices.append(self.weights[i][choice])
        
        weight_choices = np.array(weight_choices)
        arm_rewards = self.rewards[t][choices]
        personal_rewards = (self.rewards[t] / (weight + 1e-6))[choices]
        personal_rewards = personal_rewards * weight_choices

        personal_expected_rewards = (self.mu / (weight + 1e-6))[choices]
        personal_expected_rewards = personal_expected_rewards * weight_choices

        weight = np.tile(weight, self.N).reshape(self.N, -1)
        for i, choice in enumerate(choices):
            weight[i][choice] -= self.weights[i][choice]
        
        weight_choices = np.tile(weight_choices.reshape(self.N, 1), [1, self.K])
        weight = weight + weight_choices
        reward_deviation = np.tile(self.mu.reshape(-1, self.K), [self.N, 1]) / (weight + 1e-6) * weight_choices
        reward_best_deviation = np.max(reward_deviation, axis=1)
        regrets = np.maximum(0, reward_best_deviation - personal_expected_rewards)

        regrets = np.array(regrets)
        is_pne = (np.sum(regrets) <= 1e-6)
        return arm_rewards, personal_rewards, is_pne, regrets
