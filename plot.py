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

COUNT = 100

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

COLOR_DARK = {
    "SelfishRobustMMAB": "#DBAF63",
    "TotalReward": "#543466",
    "ExploreThenCommit": "#22613D",
    "SMAA": "#B03E3A",
}
COLOR = {
    "SelfishRobustMMAB": "#FB9649",
    "TotalReward": "#605CB8",
    "ExploreThenCommit": "#53C292",
    # "ExploreThenCommitPlus": "#FFE680",
    "SMAA": "#E64640",
}
MARKER = {
    "SelfishRobustMMAB": "^",
    "TotalReward": "p",
    "ExploreThenCommit": "s",
    # "ExploreThenCommitPlus": "P",
    "SMAA": "o",
}

# COLOR_RELAXED = [, "#F52E00", "#FF6003", "#FE7701"]
COLOR_RELAXED = ["#4A0F0F", "#870300", "#D61010", "#E3784D", "#F6BDC0"]
MARKER_RELAXED = ["^", "p", "s", "o"]


def setting_path(N, K, T, dis, cate):
    return "N_{}_K_{}_T_{}_dis_{}_cate_{}".format(N, K, T, dis, cate)


def analyze_method_run(setting, method):
    res_path = os.path.join("results", setting, method)

    final = {"personal_rewards": None, "is_pne": None, "regrets": None}
    count = 0
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

    print(setting, method, count, final["is_pne"][-1], final["regrets"][-1])

    return count, final


def analyze_method(setting, method):
    is_pne_max = 0
    regret_min = 1e9
    final = None
    for run in sorted(os.listdir(os.path.join("results", setting))):
        if method + "_" not in run:
            continue
        count, res = analyze_method_run(setting, run)
        if count < 2:
            continue
        # if res["is_pne"][-1] > is_pne_max:
        #     is_pne_max = res["is_pne"][-1]
        #     final = res.copy()
        if res["regrets"][-1] < regret_min:
            regret_min = res["regrets"][-1]
            final = res.copy()
    if final is not None:
        print(setting, method, "best", final["is_pne"][-1], final["regrets"][-1])
    return final


def plot_part(N, K, T, dis, cate, ax1):
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

        ax1.ticklabel_format(style="sci", scilimits=(-3, 4), axis="both")

        ax1.set_xticks(np.arange(0, T + 1, T // 5))
        ax1.tick_params(axis="both", which="major", labelsize=TICKFONTSIZE)

        ax1.set_title(
            "$N={}$, {} distribution".format(N, dis.capitalize()), size=FONTSIZE, pad=15
        )


def plot_all():
    plt.clf()
    fig, axes = plt.subplots(2, 4, figsize=(17, 6.5))

    cates = ["normal", "same"]
    for i, cate in enumerate(cates):
        plot_part(4, 5, 500000, "beta", cate, axes[i][0])
        plot_part(4, 5, 500000, "bernoulli", cate, axes[i][1])
        plot_part(7, 5, 500000, "beta", cate, axes[i][2])
        plot_part(7, 5, 500000, "bernoulli", cate, axes[i][3])

    axes[0][0].set_ylabel("Normal", size=FONTSIZE)
    axes[1][0].set_ylabel("Same", size=FONTSIZE)

    lines, labels = axes[0, 1].get_legend_handles_labels()
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
    plot_all()


if __name__ == "__main__":
    main()
