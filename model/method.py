import numpy as np

from model.data import set_seed


def calculate_PNE(mu, N):
    mu_tmp = mu.reshape(-1, 1)
    divide = 1.0 / np.arange(1, N + 1)
    mul = mu_tmp * divide
    ind = np.unravel_index(np.argsort(mul, axis=None), mul.shape)
    max_ = mu[ind[0][-N]] / (ind[1][-N] + 1)
    tmp = np.floor((mu + 1e-6) / (max_ + 1e-8)).astype(int)
    if np.sum(tmp) == N:
        return tmp, max_, True
    else:
        s = np.sum(tmp) - N
        for i in range(tmp.shape[0]):
            if np.abs(max_ * tmp[i] - mu[i]) < 1e-6:
                tmp[i] = tmp[i] - 1
                s -= 1
            if s == 0:
                break
        return tmp, max_, False


def kl_divergence(a, b):
    return a * np.log(a / b + 1e-10) + (1 - a) * np.log((1 - a) / (1 - b) + 1e-10)


class SMAA:
    def __init__(self, N, K, T, rank, loop, tolerance, beta, seed=0):
        set_seed(seed)
        self.N = N
        self.K = K
        self.T = T
        self.rank = rank
        self.rewards = np.zeros(self.K)
        self.count = np.zeros(self.K)
        self.tolerance = tolerance
        self.beta = beta
        self.pull_list = []
        self.loop = loop

        self.coin = np.random.rand(T) > 0.5
        self.coin_count = 0

        self.KK = int(np.ceil(self.K / self.N)) * self.N

    def pull(self, t):
        if t < self.K:
            k = t
        elif t < self.KK:
            k = np.random.randint(0, self.K)
        else:
            if t % self.N == 0:
                mu = self.rewards / self.count
                pne, z, _ = calculate_PNE(mu, self.N)
                pne_backup = pne.copy()
                self.pne_list = []
                i = 0
                while i < pne.shape[0]:
                    if pne[i] > 0:
                        self.pne_list.append(i)
                        pne[i] -= 1
                    else:
                        i += 1

                self.candidate = []
                tmp = self.count * kl_divergence(mu, z)
                target = self.beta * (np.log(t + 1) + 4 * np.log(np.log(t + 1)))
                for i in range(self.K):
                    if pne_backup[i] == 0 and tmp[i] <= target:
                        self.candidate.append(i)

                self.explore_idx = -1

                for i in range(self.N):
                    if (
                        np.abs(z * pne_backup[self.pne_list[i]] - mu[self.pne_list[i]])
                        < self.tolerance
                    ):
                        self.explore_idx = i

            idx = (self.rank + t) % self.N
            k = self.pne_list[idx]
            if idx == self.explore_idx and len(self.candidate) > 0:
                if self.coin[self.coin_count]:
                    k = np.random.choice(self.candidate)
                self.coin_count += 1
        self.pull_list.append(k)
        return k

    def update(self, arm_reward, personal_reward, choices):
        k = self.pull_list[-1]
        self.rewards[k] += arm_reward
        self.count[k] += 1


class TotalReward:
    def __init__(self, N, K, T, rank, loop, alpha, seed=0):
        self.N = N
        self.K = K
        self.T = T
        self.rank = rank
        self.alpha = alpha
        self.loop = loop

        self.rewards = np.zeros(self.K)
        self.count = np.zeros(self.K)
        self.T0 = self.alpha * np.log(T)
        self.pne_list = None
        self.pull_list = []
        set_seed(seed)

    def pull(self, t):
        if t < self.K:
            k = t
        elif t < self.T0:
            k = np.random.randint(0, self.K)
        else:
            if self.pne_list is None:
                if self.K >= self.N:
                    mu = self.rewards / self.count
                    idx = np.argsort(mu)[::-1]
                    self.pne_list = idx[: self.N]
                else:
                    mu = self.rewards / self.count
                    pne = [1] * self.K
                    pne[0] += self.N - self.K
                    flag = True
                    while flag:
                        flag = False
                        for i in range(self.K):
                            if pne[i] == 1:
                                continue
                            for j in range(self.K):
                                if i == j:
                                    continue
                                if mu[j] / (pne[j] + 1) > mu[i] / pne[i]:
                                    pne[i] -= 1
                                    pne[j] += 1
                                    flag = True
                                    break
                            if flag:
                                break
                    self.pne_list = []
                    i = 0
                    while i < self.K:
                        if pne[i] > 0:
                            self.pne_list.append(i)
                            pne[i] -= 1
                        else:
                            i += 1
                print(self.pne_list)
            k = self.pne_list[(self.rank + t) % self.N]
        self.pull_list.append(k)
        return k

    def update(self, arm_reward, personal_reward, choices):
        k = self.pull_list[-1]
        self.rewards[k] += arm_reward
        self.count[k] += 1


class Ours:
    def __init__(self, N, K, T, rank, loop, c1, c2, c3, eta, epsilon, seed=0, debug=False):
        set_seed(seed)
        self.N = N
        self.K = K
        self.T = T
        self.rank = rank
        self.rewards = np.zeros(self.K)
        self.count = np.zeros(self.K)
        self.pull_list = []
        self.loop = loop

        if loop.delta > 1e-6:
            self.gamma = c1 * loop.delta / 4 / self.N
            self.c1 = min(np.ceil(c1 * np.log(T) * K * K / loop.delta / loop.delta), self.T / 10 / K)
            self.gammas = np.random.normal(0, self.gamma, K)
        else:
            self.c1 = np.ceil(c1 * np.log(T) * K * K)
            self.gamma = 0
            self.gammas = np.zeros(K)
        print("Exploring phase:", self.c1)
        self.c2 = c2
        self.c3 = c3
        self.eta = eta
        self.epsilon = epsilon

        self.mood = "discontent"
        self.action = 0
        self.utility = 0
        self.last_utility = 0

        self.rewards = np.zeros(self.K)
        self.count = np.zeros(self.K)

        self.remaining = 0
        self.round = 0
        self.phase = "exploring"

        self.count_best = np.zeros(self.K)

        self.threshold = 1e-5
    
        self.probabilities = np.random.rand(self.T * 10)
        self.prob_idx = 0

        self.debug = debug

    def next_prob(self):
        self.prob_idx += 1
        return self.probabilities[self.prob_idx - 1]

    def F(self, u):
        return -u / (self.N * (4 + 3 * self.gamma)) + 1 / 3 / self.N

    def G(self, u):
        return -u / (4 + 3 * self.gamma) + 1 / 3

    def pull(self, t):
        if t < self.c1 * self.K:
            k = t % self.K
        else:
            if t == self.c1 * self.K:
                self.mu_hat = self.rewards / (self.count + 1e-6)
                self.phase = "exploiting"
                if self.debug:
                    self.mu_hat = self.loop.mu
                print(self.loop.mu, self.mu_hat)
            if self.remaining == 0:
                if self.phase == "exploiting":
                    self.round += 1
                    self.phase = "learning"
                    self.remaining = np.ceil(self.c2 * np.power(self.round, self.eta))
                    self.mu_hat = self.rewards / (self.count + 1e-6)
                    if self.debug:
                        self.mu_hat = self.loop.mu
                else:
                    self.phase = "exploiting"
                    self.remaining = np.ceil(self.c3 * np.power(2, self.round))
            if self.phase == "exploiting":
                k = np.argmax(self.count_best)
                if self.remaining == 1:
                    self.mood = "content"
                    self.action = k
                    self.utility = (
                        self.mu_hat[k]
                        * self.last_personal_reward
                        / (self.last_arm_reward + 1e-6)
                        + self.gammas[k]
                    )
            else:
                if self.mood == "discontent":
                    k = np.random.randint(0, self.K)
                elif self.mood == "content":
                    p = self.next_prob()
                    if p >= self.epsilon:
                        k = self.action
                    else:
                        k = np.random.randint(0, self.K - 1)
                        if k >= self.action:
                            k += 1
                else:
                    k = self.action

            self.remaining -= 1

        self.pull_list.append(k)
        return k

    def update(self, arm_reward, personal_reward, choices):
        k = self.pull_list[-1]
        self.rewards[k] += arm_reward
        self.count[k] += 1
        self.last_personal_reward, self.last_arm_reward = personal_reward, arm_reward

        if self.phase == "learning":
            if arm_reward == 0:
                utility = 0
            else:
                utility = (
                    personal_reward / (arm_reward) * self.mu_hat[k] + self.gammas[k]
                )
            self.last_utility = utility
            p = self.next_prob()
            if self.mood == "discontent":
                if p <= np.power(self.epsilon, self.F(utility)):
                    self.mood = "content"
                    self.action = k
                    self.utility = utility
            elif self.mood == "content":
                if k == self.action:
                    if utility > self.utility + self.threshold:
                        self.mood = "hopeful"
                    elif np.abs(utility - self.utility) < self.threshold:
                        self.mood = "content"
                    else:
                        self.mood = "watchful"
                else:
                    p = self.next_prob()
                    if utility >= self.utility + self.threshold and p <= np.power(
                        self.epsilon, self.G(utility - self.utility)
                    ):
                        self.mood = "content"
                        self.action = k
                        self.utlity = utility
            elif self.mood == "watchful":
                if utility > self.utility + self.threshold:
                    self.mood = "hopeful"
                elif np.abs(utility - self.utility) < self.threshold:
                    self.mood = "content"
                else:
                    self.mood = "discontent"
            elif self.mood == "hopeful":
                if utility > self.utility + self.threshold:
                    self.mood = "content"
                    self.utility = utility
                elif np.abs(utility - self.utility) < self.threshold:
                    self.mood = "content"
                else:
                    self.mood = "watchful"

            if self.mood == "content":
                self.count_best[k] += 1
            # if self.rank == 0:
            #     print(self.mood, self.action, self.utility)


class SelfishRobustMMAB:
    def __init__(self, N, K, T, rank, loop, beta, seed=0):
        set_seed(seed)
        self.N = N
        self.K = K
        self.T = T
        self.rank = rank
        self.rewards = np.zeros(self.K)
        self.count = np.zeros(self.K)
        self.beta = beta
        self.pull_list = []
        self.loop = loop

        self.coin = np.random.rand(T) > 0.5
        self.coin_count = 0

        self.KK = int(np.ceil(self.K / self.N)) * self.N

    def pull(self, t):
        if t < self.K:
            k = t
        elif t < self.KK:
            k = np.random.randint(0, self.K)
        else:
            if t % self.N == 0:
                mu = self.rewards / self.count
                idx = np.argsort(mu)[::-1]
                z = mu[idx[self.N - 1]]

                self.candidate = []
                tmp = self.count * kl_divergence(mu, z)
                target = self.beta * (np.log(t + 1) + 4 * np.log(np.log(t + 1)))
                for i in range(self.K):
                    if mu[i] < z and tmp[i] <= target:
                        self.candidate.append(i)

                self.pne_list = idx[: self.N]
                self.explore_idx = self.N - 1

            idx = (self.rank + t) % self.N
            k = self.pne_list[idx]
            if idx == self.explore_idx and len(self.candidate) > 0:
                if self.coin[self.coin_count]:
                    k = np.random.choice(self.candidate)
                self.coin_count += 1
        self.pull_list.append(k)
        return k

    def update(self, arm_reward, personal_reward, choices):
        k = self.pull_list[-1]
        self.rewards[k] += arm_reward
        self.count[k] += 1
