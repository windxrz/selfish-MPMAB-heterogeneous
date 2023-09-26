from model.data import Loop


if __name__ == "__main__":
    loop = Loop(3, 2, 10, cate="same")
    choices = [1, 0, 1]
    t = 0
    print(loop.pull(choices, t))
