from model.data import Loop

def generate_data():
    Ns = [4, 7]
    K = 5
    T = 10000
    dises = ["beta", "bernoulli"]
    cates = ["normal", "same"]
    for N in Ns:
        for dis in dises:
            for cate in cates:
                for seed in range(100):
                    loop = Loop(N, K, T, dis=dis, cate=cate, seed=seed)

if __name__ == "__main__":
    # loop = Loop(3, 2, 10, cate="same")
    generate_data()
    # choices = [1, 0, 1]
    # t = 0
    # print(loop.pull(choices, t))
