import os

import torch
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from scipy import spatial as sci_spatial
from rdkit import Chem, DataStructs

from utils.evaluation import eval_bond_length


# region Metrics Loading
class Globals:
    variants = [
        "crossdocked",
        "liGAN",
        "AR",
        "Pocket2Mol",
        "TargetDiff",
        "TargetDiff-r",
        "TargetDiff+Ours",  # only used for value evaluation
        "BindDM-r",  # BindDM reproduction (seems useless)
        "BindDM+Ours",
    ]  # General Configs
    paths = [
        os.path.join("..", "metrics", f"{p.lower()}_vina_docked.pt") for p in variants
    ]
    # results = [torch.load(p) for p in paths]
    results = [[]]
    atom_dist = []
    c_c_dist = []


# group the reference for common usage
Globals.results[0] = [[v] for v in Globals.results[0]]
# endregion


def vina_energy_plot(figname: str, legend_fontsize=20, show=True):
    OFFSET = 2
    LIGAN = 0
    try:
        vina = [
            np.array(
                [  # liGAN
                    (
                        np.median(
                            [
                                v["vina"][0]["affinity"] if v["vina"] != None else 0.0
                                for v in pocket
                            ]
                        )
                        if len(pocket) > 0
                        else 0.0
                    )
                    for pocket in Globals.results[1]
                ]
            )
        ]
        LIGAN += len(vina)
        for idx, val in enumerate(Globals.results[OFFSET:]):  # AR, P2M, TD, GD, BGD
            value = np.array(
                [
                    (
                        np.median([v["vina"]["dock"][0]["affinity"] for v in pocket])
                        if len(pocket) > 0
                        else 0.0
                    )
                    for pocket in val
                ]
            )
            vina.append(value)
            print(f"processing {Globals.variants[idx+OFFSET]}")
    except:
        breakpoint()

    all_vina = np.stack(vina, axis=0)
    best_vina_idx = np.argmin(all_vina, axis=0)

    plt.figure(figsize=(24, 6), dpi=100)

    ax = plt.subplot(1, 1, 1)
    n_data = len(vina[-1])
    fig1_idx = np.argsort(vina[-1])
    ALPHA = 0.75
    POINT_SIZE = 128
    SCALER = 0.75

    for idx, vn in enumerate(vina):
        plt.scatter(
            np.arange(n_data),
            vn[fig1_idx],
            label=f"{Globals.variants[idx+OFFSET-LIGAN]}({np.mean(best_vina_idx==idx)*100:.0f}%)",
            alpha=ALPHA,
            s=POINT_SIZE * SCALER,
        )

    plt.yticks(fontsize=16)
    for i in range(n_data):
        plt.axvline(i, c="0.1", lw=0.2)
    plt.xlim(-1, 100)
    plt.ylim(-13, -1.5)
    yticks = [-14, -12, -10, -8, -6, -4, -2]
    plt.yticks(yticks, [str(i) for i in yticks], fontsize=20)
    plt.ylabel("Median Vina Dock", fontsize=30)
    plt.legend(
        fontsize=legend_fontsize,
        handletextpad=0.01,
        loc="lower center",
        ncol=len(Globals.variants) - OFFSET + LIGAN,
        frameon=False,
        bbox_to_anchor=(0.5, -0.25),
    )
    plt.xticks(
        np.arange(0, 100, 10),
        [f"target {v}" for v in np.arange(0, 100, 10)],
        fontsize=20,
    )

    plt.tight_layout()
    try:
        plt.savefig(figname)
    except:
        breakpoint()
    if show:
        plt.show()


# region High Affinity
def compute_high_affinity(vina_ref, results, ret: bool = False):
    percentage_good = []
    num_docked = []
    qed_good, sa_good = [], []
    for i in range(len(results)):
        score_ref = vina_ref[i]
        pocket_results = [r for r in results[i] if r["vina"] is not None]
        if len(pocket_results) < 50:
            continue
        num_docked.append(len(pocket_results))

        scores_gen = []
        for docked in pocket_results:
            aff = docked["vina"]["dock"][0]["affinity"]
            scores_gen.append(aff)
            if isinstance(score_ref, list):
                score_ref = score_ref[0]["vina"]["dock"][0]["affinity"]
            if aff <= score_ref:
                qed_good.append(docked["chem_results"]["qed"])
                sa_good.append(docked["chem_results"]["sa"])
        scores_gen = np.array(scores_gen)
        percentage_good.append((scores_gen <= score_ref).mean())

    percentage_good = np.array(percentage_good)
    num_docked = np.array(num_docked)

    if ret:
        return (np.mean(percentage_good), np.median(percentage_good))
    print(
        "[HF%%]  Avg: %.2f%% | Med: %.2f%% "
        % (np.mean(percentage_good) * 100, np.median(percentage_good) * 100)
    )
    print(
        "[HF-QED]  Avg: %.4f | Med: %.4f "
        % (np.mean(qed_good) * 100, np.median(qed_good) * 100)
    )
    print(
        "[HF-SA]   Avg: %.4f | Med: %.4f "
        % (np.mean(sa_good) * 100, np.median(sa_good) * 100)
    )
    print("[Success%%] %.2f%% " % (np.mean(percentage_good > 0) * 100,))


# endregion


# region Diversity
def pocket_similarity(mol_list):
    """
    Calculate the similarity between the generated molecules in the same pocket.
    """
    if not mol_list:
        return [0]
    mol_fp_list = [Chem.RDKFingerprint(mol) for mol in mol_list]
    sim_list = [
        DataStructs.TanimotoSimilarity(mol, ref)
        for i, ref in enumerate(mol_fp_list)
        for mol in mol_fp_list[i + 1 :]
    ]
    return sim_list


def compute_diversity(metrics) -> tuple:
    """
    Compute the diversity of the generated molecules based on metrics.pt.
    """
    pocket_sim = [
        np.mean(pocket_similarity([rel["mol"] for rel in pocket])) for pocket in metrics
    ]
    sim_mean, sim_med = np.mean(pocket_sim), np.median(pocket_sim)
    return 1 - sim_mean, 1 - sim_med


# endregion

# region All-atom distribution
LABEL_FONTSIZE = 18
LW = 2
ALPHA = 0.75
BINS = np.linspace(0, 12, 100)


def get_all_atom_distance(results):
    atom_distance_list = []
    for pocket in results:
        for ligand in pocket:
            mol = ligand["mol"]
            mol = Chem.RemoveAllHs(mol)
            pos = mol.GetConformers()[0].GetPositions()
            dist = sci_spatial.distance.pdist(pos, metric="euclidean")
            atom_distance_list += dist.tolist()
    return np.array(atom_distance_list)


def _compute_jsd(atom_dist_list, reference_profile, bin):
    profile = eval_bond_length.get_distribution(atom_dist_list, bins=bin)
    return sci_spatial.distance.jensenshannon(reference_profile, profile)


def _plot_label(plot_ylabel=False):
    if plot_ylabel:
        plt.ylabel("Density", fontsize=LABEL_FONTSIZE)
    plt.xlabel("Distance of all atom pairs ($\AA$)", fontsize=LABEL_FONTSIZE)
    plt.ylim(0, 0.5)
    plt.xlim(0, 12)


def _plot_hist(
    ax: Axes,
    dists: list,
    rng: tuple,
    colors: list,
    reference_profile: np.ndarray,
    plot_ylabel=False,
):
    assert len(colors) == rng[1] - rng[0]
    assert len(rng) == 2

    ax.hist(
        dists[0],
        bins=BINS,
        histtype="step",
        density=True,
        lw=LW,
        color="grey",
        alpha=ALPHA,
    )  # type: ignore
    jsds = []
    for idx, color in zip(range(rng[0], rng[1]), colors):
        jsd = _compute_jsd(dists[idx], reference_profile, BINS)
        ax.hist(
            dists[idx],
            bins=BINS,
            histtype="step",
            density=True,
            lw=LW,
            color=color,
            alpha=ALPHA,
            label=f"{Globals.variants[idx]} JSD: {jsd:.3f}",
        )  # type: ignore
        jsds.append(jsd)

    labels = [
        f"{Globals.variants[idx]}: {jsd:.3f}"
        for idx, jsd in zip(range(rng[0], rng[1]), jsds)
    ]
    from matplotlib.lines import Line2D

    custom_lines = [
        Line2D([0], [0], color=c, marker="o", linestyle="None", markersize=10)
        for c in colors
    ]
    ax.legend(
        custom_lines, labels, fontsize=15, frameon=False, loc="upper right", ncol=1
    )
    _plot_label(plot_ylabel)


def jsd_all_atom_plot(figname: str, show: bool = True):
    bins = np.linspace(0, 12, 100)
    reference_profile = eval_bond_length.get_distribution(
        get_all_atom_distance(Globals.results[0]), bins=bins
    )
    dists = [get_all_atom_distance(r) for r in Globals.results]

    last_start = len(Globals.variants) - 2
    indexes = [(1, 4), (4, last_start), (last_start, len(Globals.variants))]
    spx, spy = 1, len(indexes)
    SUBFIG_SIZE = 5
    plt.figure(figsize=(SUBFIG_SIZE * spy, SUBFIG_SIZE * spx))

    colors = ["red", "green", "blue", "pink", "black", "red", "green", "blue"]
    for subplot_idx in range(spy):
        _plot_hist(
            plt.subplot(spx, spy, subplot_idx + 1),
            dists,
            indexes[subplot_idx],
            colors[indexes[subplot_idx][0] - 1 : indexes[subplot_idx][1] - 1],
            reference_profile,
            plot_ylabel=subplot_idx == 0,
        )

    plt.tight_layout()
    plt.savefig(figname)
    if show:
        plt.show()


# endregion All-atom distribution


def annealing_plot(figname: str, show: bool = True):
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Define epochs
    epochs = np.linspace(0, 200, 100)
    xlim, ylim = 210, 1.05
    titles = ["(a)", "(b)", "(c)"]

    # Panel (a) - Different μ values
    colors_a = ["red", "green", "blue", "orange", "cyan"]
    mu_values = [1, 12, 25, 40, 50]

    for i, mu in enumerate(mu_values):
        p = mu / (mu + np.exp(epochs / mu))
        ax1.plot(epochs, p, linewidth=2, label=f"μ={mu}")

    # Panel (b) - Different r values
    colors_b = ["red", "black", "green", "blue", "orange", "cyan"]
    r_values = [1.5, 2, 3, 4, 8, 250]
    for i, r in enumerate(r_values):
        p = np.sqrt((r**2 - (epochs / 100) ** 2).clip(0)) / r
        if r > 8:
            r = r"$\infty$"
        ax2.plot(epochs, p, linewidth=2, label=f"r={r}")

    # Panel (c) - Three different conditions
    colors_c = ["red", "green", "blue"]
    # μ=25 condition (similar to panel a)
    mu = 25
    c0 = mu / (mu + np.exp(epochs / mu))
    # slope=-0.005 condition (linear decay)
    slope = -0.005
    c1 = 1 + slope * epochs
    # r=2 condition (circle decay)
    r = 2
    c2 = np.sqrt((r**2 - (epochs / 100) ** 2).clip(0)) / r
    ax3.plot(epochs, c0, linewidth=2, label=f"μ={mu}")
    ax3.plot(epochs, c1, linewidth=2, label=f"slope={slope}")
    ax3.plot(epochs, c2, linewidth=2, label=f"r={r}")

    for i, ax in enumerate([ax1, ax2, ax3]):
        ax.set_xlabel("epoch")
        if i == 0:
            ax.set_ylabel("p", rotation='vertical')
            
        ax.set_title(titles[i])
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, xlim)
        ax.legend()
        ax.set_ylim(0, ylim)

    fig.tight_layout()
    fig.savefig(figname)
    if show:
        fig.show()


if __name__ == "__main__":
    # offset = 2
    # for i in range(offset, len(Globals.variants)):
    #     div = compute_diversity(Globals.results[i])
    #     ha = compute_high_affinity(Globals.results[0], Globals.results[i])
    #     print(f"{Globals.variants[i]} diversity: {div}")
    #     print(f"{Globals.variants[i]} high affinity: {ha}")

    root, suff = "paper", "_esf"
    filer = lambda x: os.path.join(root, f"{x}{suff}.pdf")
    # vina_energy_plot(filer("mve"), 16, show=False) # figure 2
    # jsd_all_atom_plot(filer("dist_12a"), show=False) # figure 3
    annealing_plot(filer("ablation"), show=False)  # figure 4
