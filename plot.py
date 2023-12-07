import os
import pickle as pkl

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc

rc("font", **{"family": "sans-serif", "sans-serif": ["Times New Roman"]})

rc("text", usetex=True)

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42

COUNT = 50
THRESHOLD = 0.9

LINEWIDTH = 3
MARKEREDGEWIDTH = 2
MS = 20
FONTSIZE = 22
LEGEND_FONSIZE = 20
LABEL_FONTSIZE = 25
MARKERSIZE = 10
TICKFONTSIZE = 18

COLOR_LIST = [
    "#C9625E",
    "#CBBD62",
    "#8ECB62",
    "#62CBC7",
    "#CD6ACA",
    "#000000",
    "#AAAAAA",
    "#9F10A3",
    "#BD2C14",
    "#0E5AD3",
    "#22AB16",
]


COLOR = {
    "TotalReward": "#605CB8",
    "SelfishRobustMMAB": "#53C292",
    "SMAA": "#FB9649",
    # "Oursnogamma": "#CBBD62",
    "Ours": "#E64640",
}
MARKER = {
    "TotalReward": "p",
    "SelfishRobustMMAB": "s",
    "SMAA": "^",
    # "Oursnogamma": "H",
    "Ours": "o",
}


def setting_path(N, K, T, dis, cate):
    return "N_{}_K_{}_T_{}_dis_{}_cate_{}".format(N, K, T, dis, cate)


def analyze_method_run(setting, method):
    res_path = os.path.join("results", setting, method)

    if os.path.exists(os.path.join(res_path, "res_{}.pkl".format(COUNT))):
        with open(os.path.join(res_path, "res_{}.pkl".format(COUNT)), "rb") as f:
            final = pkl.loads(f.read())
            f.close()
        return COUNT, final
    else:
        final = {"personal_rewards": None, "is_pne": None, "regrets": None}
        count = 0
        if len(os.listdir(res_path)) < COUNT * THRESHOLD:
            return 0, None
        for filename in sorted(os.listdir(res_path)):
            with open(os.path.join(res_path, filename), "rb") as f:
                res = pkl.loads(f.read())
                f.close()
            count += 1
            if final["personal_rewards"] is None:
                final["personal_rewards"] = res["personal_rewards"][np.newaxis, :]
                final["is_pne"] = res["is_pne"][np.newaxis, :]
                final["regrets"] = res["regrets"][np.newaxis, :]
            else:
                final["personal_rewards"] = np.concatenate(
                    [final["personal_rewards"], res["personal_rewards"][np.newaxis, :]],
                    axis=0,
                )
                final["is_pne"] = np.concatenate(
                    [final["is_pne"], res["is_pne"][np.newaxis, :]],
                    axis=0,
                )
                final["regrets"] = np.concatenate(
                    [final["regrets"], res["regrets"][np.newaxis, :]],
                    axis=0,
                )
            if count == COUNT:
                break

        if count == 0:
            return 0, None

        final["regrets"] = np.mean(final["regrets"], axis=2)

        final["regrets_std"] = np.std(final["regrets"], axis=0)
        final["is_pne_std"] = np.std(final["is_pne"], axis=0)

        final["regrets"] = np.mean(final["regrets"], axis=0)
        final["is_pne"] = np.mean(final["is_pne"], axis=0)

        with open(os.path.join(res_path, "res_{}.pkl".format(COUNT)), "wb") as f:
            f.write(pkl.dumps(final))
            f.close()

        # print(setting, method, count, final["is_pne"][-1], final["regrets"][-1])

        return count, final


def analyze_method(setting, method):
    is_pne_max = 0
    regret_min = 1e9
    final = None
    run_setting = ""
    for run in sorted(os.listdir(os.path.join("results", setting))):
        if method + "_" not in run:
            continue
        count, res = analyze_method_run(setting, run)
        if count < COUNT * THRESHOLD:
            continue
        # if res["is_pne"][-1] > is_pne_max:
        #     is_pne_max = res["is_pne"][-1]
        #     final = res.copy()
        #     run_setting = run
        if res["regrets"][-1] < regret_min:
            regret_min = res["regrets"][-1]
            final = res.copy()
            run_setting = run

    if final is not None:
        print(
            setting,
            method,
            "best",
            run_setting,
            int(round(final["is_pne"][-1], 0)),
            int(round(final["is_pne_std"][-1], 0)),
            int(round(final["regrets"][-1], 0)),
            int(round(final["regrets_std"][-1], 0)),
        )
    return final


def plot_part(N, K, T, dis, cate, ax1, ax2):
    setting_name = setting_path(N, K, T, dis, cate)

    step = 100

    print(setting_name)
    if not os.path.exists(os.path.join("results", setting_name)):
        return

    for method in COLOR.keys():
        if "Relaxed" in method:
            continue
        res = analyze_method(setting_name, method)
        if res is None:
            continue

        method_name = method.split("_")[0]
        label_name = method_name

        ax1.plot(
            range(1, T + 1, step),
            res["regrets"],
            label=label_name,
            color=COLOR[method_name],
            linewidth=LINEWIDTH,
            marker=MARKER[method_name],
            markevery=T // (5 * step) - 1,
            markersize=MARKERSIZE,
            markerfacecolor="None",
            markeredgewidth=MARKEREDGEWIDTH,
        )
        ax1.fill_between(
            range(1, T + 1, step),
            res["regrets"] - res["regrets_std"],
            res["regrets"] + res["regrets_std"],
            color=COLOR[method_name],
            alpha=0.1,
        )

        ax2.plot(
            range(1, T + 1, step),
            (np.arange(1, T + 1, step) - res["is_pne"]),  # / np.arange(1, T + 1, step),
            label=label_name,
            color=COLOR[method_name],
            linewidth=LINEWIDTH,
            marker=MARKER[method_name],
            markevery=T // (5 * step) - 1,
            markersize=MARKERSIZE,
            markerfacecolor="None",
            markeredgewidth=MARKEREDGEWIDTH,
        )
        ax2.fill_between(
            range(1, T + 1, step),
            (np.arange(1, T + 1, step) - res["is_pne"]) - res["is_pne_std"],
            (np.arange(1, T + 1, step) - res["is_pne"]) + res["is_pne_std"],
            color=COLOR[method_name],
            alpha=0.1,
        )

        ax1.ticklabel_format(style="sci", scilimits=(-3, 3), axis="both")
        ax2.ticklabel_format(style="sci", scilimits=(-3, 3), axis="both")

        ax1.set_xticks(np.arange(0, T + 1, T // 5))
        ax2.set_xticks(np.arange(0, T + 1, T // 5))
        ax1.tick_params(axis="both", which="major", labelsize=TICKFONTSIZE)
        ax2.tick_params(axis="both", which="major", labelsize=TICKFONTSIZE)

        ax2.set_title(
            "N={}, K={}, {}".format(
                N,
                K,
                "Equal weights" if cate == "same" else "Different weights",
            ),
            size=FONTSIZE,
            pad=15,
        )
        ax1.set_xlabel("Round", size=FONTSIZE)


def plot_all():
    plt.clf()
    fig, axes = plt.subplots(2, 4, figsize=(17, 6.5))

    dis = "beta"
    T = 1000000
    plot_part(2, 3, T, dis, "same", axes[1][0], axes[0][0])
    plot_part(2, 3, T, dis, "normal", axes[1][1], axes[0][1])
    plot_part(4, 4, T * 3, dis, "same", axes[1][2], axes[0][2])
    plot_part(4, 4, T * 3, dis, "normal", axes[1][3], axes[0][3])
    # axes[1][2].set_ylim(-5e3, 55e3)
    # axes[1][3].set_ylim(-5e3, 55e3)
    # plot_part(5, 4, T, dis, "normal", axes[0][4], axes[1][4])
    # plot_part(5, 4, T, dis, "same", axes[0][5], axes[1][5])

    axes[1][0].set_ylabel("Average Regret", size=FONTSIZE)
    axes[0][0].set_ylabel("\# of Non-PNE Rounds", size=FONTSIZE)

    lines, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        lines,
        labels,
        prop={"size": LEGEND_FONSIZE},
        loc="lower center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=5,
    )
    plt.tight_layout()
    plt.savefig("figs/all.png", bbox_inches="tight")
    plt.savefig("figs/all.pdf", bbox_inches="tight")


def plot_rebuttal():
    plt.clf()
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))

    dis = "beta"
    T = 3000000
    plot_part(2, 10, T, dis, "normal", axes[1][0], axes[0][0])
    plot_part(10, 2, T, dis, "normal", axes[1][1], axes[0][1])
    # axes[1][2].set_ylim(-5e3, 55e3)
    # axes[1][3].set_ylim(-5e3, 55e3)
    # plot_part(5, 4, T, dis, "normal", axes[0][4], axes[1][4])
    # plot_part(5, 4, T, dis, "same", axes[0][5], axes[1][5])

    axes[1][0].set_ylabel("Average Regret", size=FONTSIZE)
    axes[0][0].set_ylabel("\# of Non-PNE Rounds", size=FONTSIZE)

    lines, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        lines,
        labels,
        prop={"size": LEGEND_FONSIZE},
        loc="lower center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=5,
    )
    plt.tight_layout()
    plt.savefig("figs/all.png", bbox_inches="tight")
    plt.savefig("figs/all.pdf", bbox_inches="tight")


def main():
    if not os.path.exists("figs"):
        os.mkdir("figs")
    plot_rebuttal()


if __name__ == "__main__":
    main()
