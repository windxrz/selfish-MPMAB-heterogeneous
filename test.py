import numpy as np

from model.data import Loop
from utils.delta import calculate_SMAA


def generate_data():
    Ns = [2]
    Ks = [10]
    T = 10000
    dises = ["beta"]
    cates = ["normal"]
    for N in Ns:
        for K in Ks:
            for dis in dises:
                for cate in cates:
                    for seed in range(50):
                        loop = Loop(N, K, T, dis=dis, cate=cate, seed=seed)


def analyze_smaa():
    N = 10
    K = 2
    T = 3000000
    dis = "beta"
    cate = "normal"
    s = []
    for seed in range(50):
        loop = Loop(N, K, T, dis=dis, cate=cate, seed=seed)
        tmp = calculate_SMAA(loop.weights, loop.mu, isprint=False)
        print(tmp)
        s.append(tmp)

    print(np.mean(s))


if __name__ == "__main__":
    # loop = Loop(3, 2, 10, cate="same")
    # generate_data()
    analyze_smaa()
    # choices = [1, 0, 1]
    # t = 0
    # print(loop.pull(choices, t))
