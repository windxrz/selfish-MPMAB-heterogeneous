import os
import pickle as pkl
import random

import numpy as np

from utils.delta import calculate_delta, calculate_SMAA
from utils.utils import DENOMINATOR_DELTA, THRESHOLD, set_seed


class Loop:
    def __init__(self, N, K, T, dis="beta", cate="normal", seed=None, seed_reward=0):
        set_seed(seed)
        self.N = N
        self.K = K
        self.T = T
        self.dis = dis

        if not os.path.exists("data"):
            os.mkdir("data")
        filename = "data/N_{}_K_{}_dis_{}_cate_{}_seed_{}.pkl".format(
            N, K, dis, cate, seed if seed_reward == 0 else seed_reward - 1
        )
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                dic = pkl.load(f)
                f.close()

            self.weights = dic["weights"]
            self.mu = dic["mu"]
            self.delta = dic["delta"]
            if "beta" in self.dis:
                self.alpha = dic["alpha"]
                self.beta = dic["beta"]
            if "welfare" in dic:
                self.welfare = dic["welfare"]
            else:
                delta_pne, delta_nopne, self.delta, self.welfare = calculate_delta(
                    self.weights, self.mu
                )
                dic["welfare"] = self.welfare
                with open(filename, "wb") as f:
                    pkl.dump(dic, f)

        else:
            while True:
                if self.dis == "beta":
                    self.alpha = np.random.rand(K) * 10
                    self.beta = np.random.rand(K) * 10
                    self.mu = self.alpha / (self.alpha + self.beta)
                elif self.dis == "bernoulli":
                    self.mu = np.random.rand(K)
                
                if self.dis == "beta_power":
                    alpha = 1.05
                    mu_max = 0.98

                    log_mu = [0]  # start with log(mu_max) = 0
                    for _ in range(K - 1):
                        gap = np.random.uniform(np.log(alpha), np.log(alpha) + 0.1)  # the "+ 0.5" adds variation
                        log_mu.append(log_mu[-1] - gap)

                    mu_vals = np.exp(log_mu)

                    mu_vals = mu_vals / np.max(mu_vals) * mu_max

                    self.mu = mu_vals

                    total = 1000
                    self.alpha = self.mu * total
                    self.beta = (1 - self.mu) * total

                if cate == "normal":
                    self.weights = np.random.rand(self.N, self.K)
                elif cate == "same":
                    self.weights = np.random.rand(self.N)
                    self.weights = np.tile(self.weights, [self.K, 1]).T
                elif cate == "rewardsame":
                    self.weights = np.random.rand(self.N, self.K)
                    self.alpha = np.tile(np.random.rand() * 5, K)
                    self.beta = np.tile(np.random.rand() * 5, K)
                    self.mu = self.alpha / (self.alpha + self.beta)
                elif cate == "smaa":
                    self.weights = np.ones((self.N, self.K))

                if N <= 5:
                    delta_pne, delta_nopne, self.delta, self.welfare = calculate_delta(
                        self.weights, self.mu
                    )
                else:
                    delta_pne, delta_nopne, self.delta, self.welfare = 0, 0, 0, 0
                
                if N > 10 and "power" not in self.dis:
                    z = min(np.min(self.alpha), np.min(self.beta))
                    print(z)
                    self.alpha = 1000 / z * self.alpha
                    self.beta = 1000 / z * self.beta
                

                print("delta:", delta_pne, delta_nopne, self.delta)
                if delta_pne >= 500:
                    continue

                break

            smaa = calculate_SMAA(self.weights, self.mu)

            dic = {}
            dic["weights"] = self.weights
            dic["mu"] = self.mu
            dic["delta"] = self.delta
            dic["welfare"] = self.welfare
            if "beta" in self.dis:
                dic["alpha"] = self.alpha
                dic["beta"] = self.beta
            print(dic)
            with open(filename, "wb") as f:
                pkl.dump(dic, f)

        if seed_reward != 0:
            set_seed(seed * 10)

        if "beta" in self.dis:
            self.rewards = np.random.beta(self.alpha, self.beta, (T, K))
        elif self.dis == "bernoulli":
            self.rewards = np.random.binomial(1, self.mu, (T, K))
        
    def pull(self, choices, t):
        weight = np.zeros(self.K)
        weight_choices = []
        for i, choice in enumerate(choices):
            weight[choice] += self.weights[i][choice]
            weight_choices.append(self.weights[i][choice])

        weight_choices = np.array(weight_choices)
        arm_rewards = self.rewards[t][choices]
        personal_rewards = (self.rewards[t] / (weight + DENOMINATOR_DELTA))[choices]
        personal_rewards = personal_rewards * weight_choices

        personal_expected_rewards = (self.mu / (weight + DENOMINATOR_DELTA))[choices]
        personal_expected_rewards = personal_expected_rewards * weight_choices

        weight = np.tile(weight, self.N).reshape(self.N, -1)
        for i, choice in enumerate(choices):
            weight[i][choice] -= self.weights[i][choice]

        weight_choices = self.weights
        weight = weight + weight_choices
        reward_deviation = (
            np.tile(self.mu.reshape(-1, self.K), [self.N, 1])
            / (weight + DENOMINATOR_DELTA)
            * weight_choices
        )
        reward_best_deviation = np.max(reward_deviation, axis=1)
        regrets = np.maximum(0, reward_best_deviation - personal_expected_rewards)

        regrets = np.array(regrets)
        is_pne = (np.max(regrets) <= THRESHOLD) and (
            np.sum(personal_expected_rewards) >= self.welfare - THRESHOLD
        )
        self.tmp_personal_expected_rewards = np.sum(personal_expected_rewards)
        return arm_rewards, personal_rewards, is_pne, regrets
