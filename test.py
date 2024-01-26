import numpy as np

from model.data import Loop
from utils.delta import calculate_SMAA
from utils.delta import calculate_delta


def generate_data():
    Ns = [3]
    Ks = [5]
    T = 10000
    dises = ["beta"]
    cates = ["normal"]
    for N in Ns:
        for K in Ks:
            for dis in dises:
                for cate in cates:
                    for seed in range(50, 100):
                        loop = Loop(N, K, T, dis=dis, cate=cate, seed=seed)


def analyze_data():
    N = 10
    K = 3
    T = 3000000
    dis = "beta"
    cates = ["normal", "same"]
    s = []
    for seed in range(0, 50):
        for cate in cates:
            loop = Loop(N, K, T, dis=dis, cate=cate, seed=seed, seed_reward=0)
            tmp = calculate_SMAA(loop.weights, loop.mu, isprint=True)
            delta_pne, delta_nopne, delta, welfare = calculate_delta(
                        loop.weights, loop.mu
                    )
            print(tmp, delta)
            s.append([tmp, delta])

    print(np.mean(s, axis=1))


if __name__ == "__main__":
    # loop = Loop(3, 2, 10, cate="same")
    # generate_data()
    analyze_data()
    # choices = [1, 0, 1]
    # t = 0
    # print(loop.pull(choices, t))
