import argparse
import os
import pickle as pkl

import nni
import numpy as np
from tqdm import tqdm

from model.data import Loop
from model.method import SMAA, Ours, SelfishRobustMMAB, TotalReward
from utils.delta import calculate_delta


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", type=int, default=10)
    parser.add_argument("-K", type=int, default=2)
    parser.add_argument("-T", type=int, default=3000000)
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
            "SelfishRobustMMAB",
        ],
        default="Ours",
    )

    # TotalReward
    parser.add_argument("--alpha", type=int, default=500)

    # SMAA and SelfishRobustMMAb
    parser.add_argument("--beta", type=float, default=0.1)

    # SMAA
    parser.add_argument("--tolerance", type=float, default=1e-6)

    # Ours
    parser.add_argument("--c1", type=float, default=0.0001)
    parser.add_argument("--c2", type=float, default=100000000)
    parser.add_argument("--c3", type=float, default=100)
    parser.add_argument("--eta", type=float, default=0)
    parser.add_argument("--epsilon", type=float, default=1e-2)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--no-gamma", action="store_true", default=False)

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
        method_name = "{}{}_c1_{}_c2_{}_c3_{}_eta_{}_epsilon_{}".format(
            method,
            "nogamma" if args.no_gamma else "",
            args.c1,
            args.c2,
            args.c3,
            args.eta,
            args.epsilon,
        )
    elif method == "SelfishRobustMMAB":
        method_name = "{}_beta_{}".format(method, args.beta)

    res_path_base = os.path.join(
        "results",
        "N_{}_K_{}_T_{}_dis_{}_cate_{}".format(N, K, T, dis, cate),
        method_name,
    )
    if not os.path.exists(res_path_base):
        os.makedirs(res_path_base)

    total_runs = 50
    pne_nums = []
    regrets_sum = []
    if args.debug:
        last_mood = ["discontent"] * args.N
        last_choice = [0] * args.N
        last_utility = [0] * args.N
    for seed_data in range(0, total_runs):
        if args.debug and seed_data != 1:
            continue

        print("Running {}/{}".format(seed_data + 1, total_runs))

        res_file = os.path.join(res_path_base, "{}.pkl".format(seed_data))
        count_tmp = np.zeros(8)

        if (not os.path.exists(res_file)) or args.debug:
            if args.debug:
                loop = Loop(N, K, T, dis=dis, cate=cate, seed=seed_data)
            else:
                loop = Loop(N, K, T, dis=dis, cate=cate, seed=seed_data)
            print(loop.mu)
            print(loop.delta)
            print(loop.weights)
            if args.debug:
                res_delta = calculate_delta(loop.weights, loop.mu, isprint=True)
                print(res_delta)
                exit()
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
                    print("debug: ", args.debug)
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
                        debug=args.debug,
                        no_gamma=args.no_gamma,
                    )
                elif method == "SelfishRobustMMAB":
                    player = SelfishRobustMMAB(N, K, T, i, loop, beta=args.beta, seed=i)

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

                res_arm_rewards.append(arm_rewards.reshape(1, -1))
                res_personal_rewards.append(personal_rewards.reshape(1, -1))
                res_is_pne.append(is_pne)
                res_regrets.append(regrets.reshape(1, -1))
                for i in range(N):
                    players[i].update(arm_rewards[i], personal_rewards[i], choices)

                if args.method == "Ours" and args.debug:
                    flag = False
                    for i in range(args.N):
                        if players[i].mood != last_mood[i] or players[i].action != last_choice[i]:
                            flag = True
                            break
                    if flag:
                        print("=" * 15 + "t={}".format(t) + "=" * 15)
                        print(choices)
                        print(
                            [
                                (
                                    last_mood[i][0],
                                    last_choice[i],
                                    round(last_utility[i], 6),
                                )
                                for i in range(args.N)
                            ],
                        )
                        print(
                            [
                                (
                                    player.mood[0],
                                    player.action,
                                    round(player.utility, 6),
                                )
                                for player in players
                            ],
                        )
                        print([player.true_utility.round(6) for player in players])
                        print( is_pne,
                            regrets.round(2).tolist(),
                            loop.mu.round(2))
                        print(
                            "counting best: ",
                            [player.count_best.argmax() for player in players],
                        )
                        print("delta:", res_delta)
                    # if choices == [3, 0, 1, 2]:
                    #     exit()
                    # exit()

                    last_mood = [player.mood for player in players]
                    last_choice = [player.action for player in players]
                    last_utility = [player.utility for player in players]

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
