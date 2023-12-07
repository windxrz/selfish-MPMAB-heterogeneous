import random
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from multiprocessing import Pool

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def best_response_dynamics(N, K, mu, weights, threshold=1000):
    w = np.zeros(K)
    w[0] = np.sum(weights[:, 0])
    res = [0] * N
    for _ in range(threshold):
        flag = False
        for i in range(N):
            k = res[i]
            w[k] -= weights[i][k]
            r = mu * weights[i] / (w + weights[i] + 1e-6)
            t = np.argmax(r)
            if np.abs(r[t] - r[k]) < 1e-5:
                w[k] += weights[i][k]
            else:
                res[i] = t
                w[t] += weights[i][t]
                flag = True
                print(i, k, t, r[k], r[t])
                break
        if not flag:
            print(N, K, res)
            return True
    w = np.zeros(K)
    for i in range(N):
        if i not in [0, 5, 4, 10]:
            w[res[i]] += weights[i][res[i]]

    print(N, K, res, w)
    return False


def best_best_response_dynamics(N, K, mu, weights, threshold=1000, res=None):
    if res is None:
        res = np.array([0] * N).astype(np.int32)
    w = np.zeros(K)
    for i in range(N):
        w[res[i]] += weights[i][res[i]]
    for _ in range(threshold):
        flag = False
        ii = -1
        kk = 0
        tt = 0
        max_inc = 0
        for i in range(N):
            k = res[i]
            w[k] -= weights[i][k]
            r = mu * weights[i] / (w + weights[i] + 1e-9)
            t = np.argmax(r)
            if np.abs(r[t] - r[k]) >= 1e-5 and r[t] - r[k] > max_inc:
                max_inc = r[t] - r[k]
                ii = i
                kk = k
                tt = t
            w[k] += weights[i][k]

        if ii == -1:
            reward = []
            for i in range(N):
                k = res[i]
                reward.append(
                    mu[k] * weights[i][k] / (np.sum(weights[:, k] * (res == k)) + 1e-9)
                )
            # print(N, K, res + 1, reward)
            return True
        else:
            w[kk] -= weights[ii][kk]
            r = mu * weights[ii] / (w + weights[ii] + 1e-9)
            w[tt] += weights[ii][tt]
            res[ii] = tt
            # print(res, ii, kk, tt, max_inc)
    return False


def dfs(i, res, N, K, mu, weights):
    if i == N:
        if best_best_response_dynamics(N, K, mu, weights, threshold=10, res=res):
            print(N, K, res)
            return True
        return False
    for j in range(K):
        if weights[i][j] > 0:
            res[i] = j
            if dfs(i + 1, res, N, K, mu, weights):
                return True
    return False


def count_PNE_ratio(dis):
    trial = 0
    non_exist = 0
    while True:
        K = np.random.randint(10, 100)
        N = np.random.randint(10, 100)
        if dis == "uniform":
            mu = np.random.uniform(0, 1, K)
            weights = np.random.uniform(0, 1, (N, K))
        elif dis == "gaussian":
            mu = np.random.normal(0.5, 0.1, K)
            weights = np.random.uniform(0, 1, (N, K))
        elif dis == "t-distribution":
            mu = np.random.standard_t(5, K) + 0.5
            weights = np.random.standard_t(5, (N, K)) + 0.5
        elif dis == "beta":
            mu = np.random.beta(0.5, 0.5, K)
            weights = np.random.beta(0.5, 0.5, (N, K))
        mu = mu.clip(0, 1)
        weights = weights.clip(0, 1)
        trial += 1
        if not best_best_response_dynamics(N, K, mu, weights):
            non_exist += 1
        if trial % 100 == 0:
            print("dis = {}".format(dis), non_exist, trial, non_exist / trial)


def generate_matrix(N, K):
    possible_values = np.linspace(0, 1, 11)
    return [np.array(p).reshape(N, K) for p in product(possible_values, repeat=N * K)]


def count_PNE_ratio_NK(N, K):
    weights = generate_matrix(N, K)
    mus = generate_matrix(K, 1)
    trial = 0
    non_exist = 0
    for weight in tqdm(weights):
        for mu in mus:
            mu = mu.reshape(-1)
            trial += 1
            if not best_best_response_dynamics(N, K, mu, weight):
                non_exist += 1
                print(N, K, mu, weight)
    print(N, K, trial, non_exist)


def main():
    # z = 0
    # K = 5
    # N = 14
    # mu = np.array([0.84538588, 0.08993399, 0.02328485, 0.41458564, 0.20688417])
    # weights = np.array(
    #     [
    #         [0.85953784, 0.1472667, 0.23333599, 0.80106269, 0.25534665],
    #         [0.9662989, 0.18805253, 0.19130415, 0.73760676, 0.96716714],
    #         [0.31660732, 0.61450541, 0.57473792, 0.82624625, 0.80292198],
    #         [0.01589213, 0.16420416, 0.37400423, 0.56574185, 0.12780276],
    #         [0.39903661, 0.88823095, 0.84845857, 0.58696091, 0.92798576],
    #         [0.01584441, 0.01420995, 0.93271437, 0.13611637, 0.54831471],
    #         [0.76849256, 0.31070712, 0.9855136, 0.00873075, 0.23346988],
    #         [0.60553975, 0.23878048, 0.74958383, 0.25162698, 0.5263092],
    #         [0.89518715, 0.5396467, 0.07764075, 0.56965229, 0.42519371],
    #         [0.48343897, 0.34573245, 0.13526474, 0.13658451, 0.97509145],
    #         [0.35926791, 0.45314008, 0.37322034, 0.56469558, 0.80148231],
    #         [0.83973002, 0.33033441, 0.94436208, 0.83280573, 0.21922857],
    #         [0.31522223, 0.98746007, 0.60471945, 0.88926758, 0.62860021],
    #         [0.79551123, 0.9280455, 0.37032116, 0.4726352, 0.7264403],
    #     ]
    # )

    # N, K = 4, 4
    # mu = np.array([0.02, 1, 1.1, 0.8])
    # weights = np.array(
    #     [[1, 0.2, 0, 0], [0, 0.8, 0, 0.9], [0, 0, 0.2, 0.1], [0, 10, 0.8, 0]]
    # )

    # N, K = 7, 5

    # mu = np.array([0.84538588, 0.08993399, 0.02328485, 0.41458564, 0.20688417])
    # weights = np.array(
    #     [
    #         [5.18598184, 0, 0, 0, 0],
    #         [0, 0, 0, 1.3919881, 0],
    #         [0, 0, 0, 0, 0.97509145],
    #         [0.85953784, 0, 0, 0.80106269, 0],
    #         [0, 0.01420995, 0, 0, 0.54831471],
    #         [0, 0, 0, 0.58696091, 0.92798576],
    #         [0, 0.45314008, 0, 0.56469558, 0],
    #     ]
    # )

    # res = [0] * N
    # dfs(0, res, N, K, mu, weights)
    # print(best_best_response_dynamics(N, K, mu, weights))
    # return

    # N, K = 7, 4
    # mu = np.array([0.84538588, 0.09, 0.41458564, 0.20688417])
    # tmp = 1e-6
    # weights = np.array(
    #     [
    #         [5.18598184, tmp, tmp, tmp],
    #         [tmp, tmp, 1.3919881, tmp],
    #         [tmp, tmp, tmp, 0.97509145],
    #         [0.85953784, tmp, 0.80106269, tmp],
    #         [tmp, 0.01420995, tmp, 0.54831471],
    #         [tmp, tmp, 0.58696091, 0.92798576],
    #         [tmp, 0.45314008, 0.56469558, tmp],
    #     ]
    # )
    # mu = mu / np.max(mu)
    # mu = mu.round(4)
    # weights = weights / np.max(weights, axis=0, keepdims=True)
    # weights = np.round(weights, 4)
    # weights[weights == 0] = weights[weights == 0] + 0.001
    # weights = weights.round(4)
    # print(mu)
    # print(weights)
    # res = [0] * N
    # print(dfs(0, res, N, K, mu, weights))
    # print(best_best_response_dynamics(N, K, mu, weights))
    # count_PNE_ratio()
    # print(N, K, mu, weights)

    # count_PNE_ratio_NK(3, 2)
    # count_PNE_ratio("gaussian")
    # count_PNE_ratio("t-distribution")
    count_PNE_ratio("beta")


if __name__ == "__main__":
    main()

# if __name__ == '__main__':
#     N, K = 2, 3  # Replace with your own values
#     weights = generate_matrix(N, K)
#     mus = generate_matrix(K, 1)

#     args_list = [(N, K, weight, mu) for weight in weights for mu in mus]

#     trial = 0
#     non_exist = 0

#     with Pool() as pool:
#         results = list(tqdm(pool.imap_unordered(worker, args_list), total=len(args_list)))

#     for t, ne in results:
#         trial += t
#         non_exist += ne

#     print(N, K, trial, non_exist)
