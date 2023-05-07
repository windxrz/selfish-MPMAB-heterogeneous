import numpy as np


def best_response_dynamics(N, K, mu, weights, threshold=1000):
    w = np.zeros(K)
    w[0] = np.sum(weights[:, 0])
    res = [0] * N
    for _ in range(threshold):
        flag = False
        for i in range(N):
            k = res[i]
            w[k] -= weights[i][k]
            r = mu * weights[i] / (w + weights[i])
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
    return False

def best_best_response_dynamics(N, K, mu, weights, threshold=1000):
    w = np.zeros(K)
    w[0] = np.sum(weights[:, 0])
    res = np.array([0] * N).astype(np.int32)
    for _ in range(threshold):
        flag = False
        ii = -1
        kk = 0
        tt = 0
        max_inc = 0
        for i in range(N):
            k = res[i]
            w[k] -= weights[i][k]
            r = mu * weights[i] / (w + weights[i])
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
                reward.append(mu[k] * weights[i][k] / np.sum(weights[:, k] * (res == k)))
            print(N, K, res, reward)
            return True
        else:
            w[kk] -= weights[ii][kk]
            r = mu * weights[ii] / (w + weights[ii])
            w[tt] += weights[ii][tt]
            res[ii] = tt
    return False

def main():
    z = 0
    K = 5
    N = 14
    mu = np.array([0.84538588, 0.08993399, 0.02328485, 0.41458564, 0.20688417])
    weights = np.array(
        [
            [0.85953784, 0.1472667, 0.23333599, 0.80106269, 0.25534665],
            [0.9662989, 0.18805253, 0.19130415, 0.73760676, 0.96716714],
            [0.31660732, 0.61450541, 0.57473792, 0.82624625, 0.80292198],
            [0.01589213, 0.16420416, 0.37400423, 0.56574185, 0.12780276],
            [0.39903661, 0.88823095, 0.84845857, 0.58696091, 0.92798576],
            [0.01584441, 0.01420995, 0.93271437, 0.13611637, 0.54831471],
            [0.76849256, 0.31070712, 0.9855136, 0.00873075, 0.23346988],
            [0.60553975, 0.23878048, 0.74958383, 0.25162698, 0.5263092],
            [0.89518715, 0.5396467, 0.07764075, 0.56965229, 0.42519371],
            [0.48343897, 0.34573245, 0.13526474, 0.13658451, 0.97509145],
            [0.35926791, 0.45314008, 0.37322034, 0.56469558, 0.80148231],
            [0.83973002, 0.33033441, 0.94436208, 0.83280573, 0.21922857],
            [0.31522223, 0.98746007, 0.60471945, 0.88926758, 0.62860021],
            [0.79551123, 0.9280455, 0.37032116, 0.4726352, 0.7264403],
        ]
    )

    print(best_best_response_dynamics(N, K, mu, weights))
    # return
    while True:
        K = np.random.randint(2, 20)
        N = np.random.randint(K // 2, K * 3)
        mu = np.random.uniform(0, 1, K)
        weights = np.random.uniform(1e-3, 1, (N, K))
        if not best_best_response_dynamics(N, K, mu, weights):
            break
        z += 1
        print(z)
    print(N, K, mu, weights)


if __name__ == "__main__":
    main()
