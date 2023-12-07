import os
import pickle as pkl
import random

import numpy as np

from utils.delta import calculate_delta, calculate_SMAA
from utils.utils import set_seed


class Loop:
    def __init__(self, N, K, T, dis="beta", cate="normal", seed=None):
        set_seed(seed)
        self.N = N
        self.K = K
        self.T = T
        self.dis = dis

        if not os.path.exists("data"):
            os.mkdir("data")
        filename = "data/N_{}_K_{}_dis_{}_cate_{}_seed_{}.pkl".format(
            N, K, dis, cate, seed
        )
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                dic = pkl.load(f)
                f.close()

            self.weights = dic["weights"]
            self.mu = dic["mu"]
            self.delta = dic["delta"]
            if self.dis == "beta":
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

                delta_pne, delta_nopne, self.delta, self.welfare = calculate_delta(
                    self.weights, self.mu
                )
                # print("delta:", delta_pne, delta_nopne, self.delta)
                # if smaa < 0.5:
                #     continue
                if delta_pne >= 500:
                    continue
                if self.K == 2:
                    if self.N == 10 and self.delta > 1e-3:
                        break
                elif self.N == 2:
                    if self.K == 10 and self.delta > 1e-3:
                        break
                elif (
                    self.N <= 4
                    and self.delta > 0.03
                    or (self.N == 5 and self.delta > 0.01)
                    or (self.N > 5 and self.delta > 1e-4)
                    or (self.N < self.K and cate == "rewardsame")
                ):
                    break

            smaa = calculate_SMAA(self.weights, self.mu)
            print("smaa:", smaa)

            dic = {}
            dic["weights"] = self.weights
            dic["mu"] = self.mu
            dic["delta"] = self.delta
            dic["welfare"] = self.welfare
            if self.dis == "beta":
                dic["alpha"] = self.alpha
                dic["beta"] = self.beta
            with open(filename, "wb") as f:
                pkl.dump(dic, f)

        if self.dis == "beta":
            self.rewards = np.random.beta(self.alpha * 10, self.beta * 10, (T, K))
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
        personal_rewards = (self.rewards[t] / (weight + 1e-8))[choices]
        personal_rewards = personal_rewards * weight_choices

        personal_expected_rewards = (self.mu / (weight + 1e-8))[choices]
        personal_expected_rewards = personal_expected_rewards * weight_choices

        weight = np.tile(weight, self.N).reshape(self.N, -1)
        for i, choice in enumerate(choices):
            weight[i][choice] -= self.weights[i][choice]

        weight_choices = np.tile(weight_choices.reshape(self.N, 1), [1, self.K])
        weight = weight + weight_choices
        reward_deviation = (
            np.tile(self.mu.reshape(-1, self.K), [self.N, 1])
            / (weight + 1e-8)
            * weight_choices
        )
        reward_best_deviation = np.max(reward_deviation, axis=1)
        regrets = np.maximum(0, reward_best_deviation - personal_expected_rewards)

        regrets = np.array(regrets)
        is_pne = (np.sum(regrets) <= 1e-6) and (
            np.sum(personal_expected_rewards) >= self.welfare - 1e-6
        )
        self.tmp_personal_expected_rewards = np.sum(personal_expected_rewards)
        return arm_rewards, personal_rewards, is_pne, regrets
