import argparse
import os
import pickle as pkl

import nni
import numpy as np
from tqdm import tqdm

from model.data import Loop
from model.method import (
    SMAA,
    TotalReward,
    Ours,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", type=int, default=4)
    parser.add_argument("-K", type=int, default=5)
    parser.add_argument("-T", type=int, default=500000)
    parser.add_argument(
        "--dis", type=str, choices=["bernoulli", "beta"], default="beta"
    )
    parser.add_argument(
        "--cate", type=str, choices=["normal", "same", "smaa"], default="normal"
    )

    parser.add_argument(
        "--method",
        choices=[
            "SMAA",
            "TotalReward",
            "Ours",
        ],
        default="Ours",
    )

    # TotalReward
    parser.add_argument("--alpha", type=int, default=500)

    # SMAA
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--tolerance", type=float, default=1e-6)

    # Ours
    parser.add_argument("--c1", type=float, default=1e-3)
    parser.add_argument("--c2", type=float, default=100)
    parser.add_argument("--c3", type=float, default=1)
    parser.add_argument("--eta", type=float, default=1.5)
    parser.add_argument("--epsilon", type=float, default=1e-4)

    parser.add_argument("--nni", action="store_true")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.nni:
        my_dict = vars(args)
        optimized_params = nni.get_next_parameter()
        my_dict.update(optimized_params)
        args = argparse.Namespace(**my_dict)

    print(args)

    N, K, T, dis, method, cate = (
        args.N,
        args.K,
        args.T,
        args.dis,
        args.method,
        args.cate,
    )
    if method == "TotalReward":
        method_name = "{}_alpha_{}".format(method, args.alpha)
    elif method == "SMAA":
        method_name = "{}_beta_{}_tolerance_{}".format(
            method, args.beta, args.tolerance
        )
    elif method == "Ours":
        method_name = "{}_c1_{}_c2_{}_c3_{}_eta_{}_epsilon_{}".format(
            method, args.c1, args.c2, args.c3, args.eta, args.epsilon
        )

    res_path_base = os.path.join(
        "results",
        "N_{}_K_{}_T_{}_dis_{}_cate_{}".format(N, K, T, dis, cate),
        method_name,
    )
    if not os.path.exists(res_path_base):
        os.makedirs(res_path_base)

    total_runs = 20
    pne_nums = []
    regrets_sum = []
    for seed_data in range(total_runs):
        print("Running {}/{}".format(seed_data + 1, total_runs))

        res_file = os.path.join(res_path_base, "{}.pkl".format(seed_data))
        count_tmp = np.zeros(8)

        if not os.path.exists(res_file) or args.method == "Ours":
            loop = Loop(N, K, T, dis=dis, cate=cate, seed=seed_data)
            print(loop.mu)
            print(loop.delta)
            players = []
            for i in range(args.N):
                if method == "TotalReward":
                    player = TotalReward(N, K, T, i, loop, alpha=args.alpha, seed=i)
                elif method == "SMAA":
                    player = SMAA(
                        N,
                        K,
                        T,
                        i,
                        loop,
                        beta=args.beta,
                        tolerance=args.tolerance,
                        seed=i,
                    )
                elif method == "Ours":
                    player = Ours(
                        N,
                        K,
                        T,
                        i,
                        loop,
                        c1=args.c1,
                        c2=args.c2,
                        c3=args.c3,
                        eta=args.eta,
                        epsilon=args.epsilon,
                        seed=i,
                    )

                players.append(player)

            res_arm_rewards = []
            res_personal_rewards = []
            res_is_pne = []
            res_regrets = []
            print("here before t")
            for t in tqdm(range(T)):
                choices = []
                for i in range(N):
                    choices.append(players[i].pull(t))

                arm_rewards, personal_rewards, is_pne, regrets = loop.pull(choices, t)
                # if args.method == "Ours":
                #     choices_new = []
                #     for i in range(N):
                #         choices_new.append(np.argmax(players[0].count_best))
                #     _, _, is_pne_new, regrets_new = loop.pull(choices_new, t)
                #     tmp = choices[0] * 4 + choices[1] * 2 + choices[2]
                #     if players[0].mood == "content" and players[1].mood == "content" and players[2].mood == "content":
                #         count_tmp[tmp] += 1
                #         print(choices, is_pne, count_tmp / np.sum(count_tmp), loop.mu)
                    # print(choices, regrets, is_pne, choices_new, is_pne_new, regrets_new, loop.mu)

                res_arm_rewards.append(arm_rewards.reshape(1, -1))
                res_personal_rewards.append(personal_rewards.reshape(1, -1))
                res_is_pne.append(is_pne)
                res_regrets.append(regrets.reshape(1, -1))
                for i in range(N):
                    players[i].update(arm_rewards[i], personal_rewards[i], choices)

            res_arm_rewards = np.concatenate(res_arm_rewards, axis=0)

            res_personal_rewards = np.concatenate(res_personal_rewards, axis=0)
            res_personal_rewards = np.cumsum(res_personal_rewards, axis=0)
            res_personal_rewards = res_personal_rewards[::100, :]

            res_is_pne = np.array(res_is_pne)
            res_is_pne = np.cumsum(res_is_pne)
            res_is_pne = res_is_pne[::100]

            res_regrets = np.concatenate(res_regrets, axis=0)
            res_regrets = np.cumsum(res_regrets, axis=0)
            res_regrets = res_regrets[::100]

            res = {
                "personal_rewards": res_personal_rewards,
                "is_pne": res_is_pne,
                "regrets": res_regrets,
            }

            with open(res_file, "wb") as f:
                f.write(pkl.dumps(res))
                f.close()

            for player in players:
                del player
            del loop
        else:
            with open(res_file, "rb") as f:
                res = pkl.loads(f.read())
                f.close()

        print("Is PNE: ", res["is_pne"][-1], "Regrets: ", res["regrets"][-1])

        pne_nums.append(res["is_pne"][-1])
        regrets_sum.append(np.mean(res["regrets"][-1]))
        del res

        # if args.method == "Ours":
        #     break

    report = {"default": np.mean(pne_nums), "regret": np.mean(regrets_sum)}
    print(report)

    if args.nni:
        nni.report_final_result(report)


if __name__ == "__main__":
    main()
