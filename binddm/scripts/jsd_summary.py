#!/usr/bin/env python
# coding: utf-8

# In[20]:


import os
from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
# from scipy import spatial as sci_spatial
import torch

# In[2]:

from rdkit import Chem

from utils.evaluation import eval_bond_length

# In[4]:
# # Load Data

# In[43]:
# easydict = dict

class Globals:
    # variants = ["crossdocked", "liGAN", "AR", "Pocket2Mol", "TargetDiff", "TargetDiff+Ours", "BindDM-r", "BindDM+Ours"] # General Configs
    # variants = ["crossdocked", "liGAN", "AR", "Pocket2Mol", "TargetDiff", "BindDM-r", "GapDiff"] # Old Paper Configs
    # variants = ["crossdocked", "liGAN", "AR", "BindDM-r", "GapDiff"] # Old Paper Configs
    variants = ["TargetDiff-r_pdbbind", "targetdiff+ours_pdbbind"] # Old Paper Configs
    # variants = ["crossdocked", "molcraft"] # Old Paper Configs
    # variants = ["pdbbind", "targetdiff+ours_pdbbind", "BindDM-r_pdbbind", "BindDM+ours_pdbbind"] # PDBbind Configs
    # variants = ["pdbbind", "targetdiff-r_pdbbind", "targetdiff+ours_pdbbind", "BindDM-r", "BindDM+ours_pdbbind"] # PDBbind Configs
    # variants = ["crossdocked", "liGAN", "AR", "Pocket2Mol", "TargetDiff", "GapDiff"] # Paper Configs
    # variants = ["crossdocked", "liGAN", "AR", "Pocket2Mol", "TargetDiff", "TargetDiff-r", "TargetDiff+Ours", "BindDM-r", "BindDM+Ours"] # General Configs
    # variants = ["crossdocked", "liGAN", "AR", "Pocket2Mol", "TargetDiff", "TargetDiff+Ours", "BindDM-r", "BindDM+Ours"] # General Configs w/o TD-r
    paths = [os.path.join("..","metrics", f"{p.lower()}_vina_docked.pt") for p in variants]
    results = [torch.load(p, weights_only=False) for p in paths]
    atom_dist = []
    c_c_dist = []


# In[44]:
if Globals.variants[0] == "crossdocked":
    Globals.results[0] = [[v] for v in Globals.results[0]]
# # Metrics Summary

# # Vina Score

# In[40]:

def vina_energy_plot(figname:str="paper/mve.jpg", legend_fontsize=20, show=True):
    OFFSET = 2
    LIGAN = 0
    try:
        vina = [
            np.array([ # liGAN
                np.median([v['vina'][0]['affinity'] if v['vina'] != None else 0. for v in pocket]) if len(pocket) > 0 else 0.
                for pocket in Globals.results[1]
            ])
        ]
        LIGAN += len(vina)
        for idx, val in enumerate(Globals.results[OFFSET:]): # AR, P2M, TD, GD, BGD
            value = np.array([
                np.median([v['vina']['dock'][0]['affinity'] for v in pocket]) if len(pocket) > 0 else 0.
                for pocket in val
            ])
            vina.append(value)
            print(f"processing {Globals.variants[idx+OFFSET]}")
    except:
        breakpoint()

    # In[41]:

    all_vina = np.stack(vina, axis=0)
    best_vina_idx = np.argmin(all_vina, axis=0)


    plt.figure(figsize=(24, 6), dpi=100)

    ax = plt.subplot(1, 1, 1)
    # ax.set_prop_cycle('color', plt.cm.Set1.colors)
    n_data = len(vina[-1])
    fig1_idx = np.argsort(vina[-1])
    ALPHA = 0.75
    POINT_SIZE = 128
    SCALER = 0.75
    
    for idx, vn in enumerate(vina):
        plt.scatter(np.arange(n_data), vn[fig1_idx], label=f'{Globals.variants[idx+OFFSET-LIGAN]}({np.mean(best_vina_idx==idx)*100:.0f}%)', alpha=ALPHA, s=POINT_SIZE * SCALER)


    plt.yticks(fontsize=16)
    for i in range(n_data):
        plt.axvline(i, c='0.1', lw=0.2)
    plt.xlim(-1, 100)
    plt.ylim(-13, -1.5)
    # plt.yticks([-10, -8, -6, -4, -2], [-10, -8, -6, -4, '$\geq$-2'], fontsize=25)
    yticks = [-14, -12, -10, -8, -6, -4, -2]
    plt.yticks(yticks, [str(i) for i in yticks], fontsize=20)
    plt.ylabel('Median Vina Dock', fontsize=30)
    plt.legend(fontsize=legend_fontsize, handletextpad=0.01, loc='lower center', ncol=len(Globals.variants)-OFFSET+LIGAN, frameon=False, bbox_to_anchor=(0.5, -0.25))
    plt.xticks(np.arange(0, 100, 10), [f'target {v}' for v in np.arange(0, 100, 10)], fontsize=20)

    plt.tight_layout()
    try:
        plt.savefig(figname)
    except:
        breakpoint()
    if show:
        plt.show()

# In[52]:

def print_results(results, show_vina=True):
    qed = [r['chem_results']['qed'] for r in results]
    sa = [r['chem_results']['sa'] for r in results]
    mol_size = [r['mol'].GetNumAtoms() for r in results]
    print('Num results: %d' % len(results))
    if show_vina:
        vina_score_only = [x['vina']['score_only'][0]['affinity'] for x in results]
        vina_min = [x['vina']['minimize'][0]['affinity'] for x in results]
        vina_dock = [r['vina']['dock'][0]['affinity'] for r in results]
        print('[Vina Score] Avg: %.2f | Med: %.2f' % (np.mean(vina_score_only), np.median(vina_score_only)))
        print('[Vina Min]   Avg: %.2f | Med: %.2f' % (np.mean(vina_min), np.median(vina_min)))
        print('[Vina Dock]  Avg: %.4f | Med: %.4f' % (np.mean(vina_dock), np.median(vina_dock)))
        
    print('[QED]  Avg: %.4f | Med: %.4f' % (np.mean(qed), np.median(qed)))
    print('[SA]   Avg: %.4f | Med: %.4f' % (np.mean(sa), np.median(sa)))
    print('[Size] Avg: %.4f | Med: %.4f' % (np.mean(mol_size), np.median(mol_size)))

def compute_high_affinity(vina_ref, results, thres=50):
    percentage_good = []
    num_docked = []
    qed_good, sa_good = [], []
    for i in range(len(results)):
        score_ref = vina_ref[i]
        pocket_results = [r for r in results[i] if r['vina'] is not None]
        if len(pocket_results) < thres:
            continue
        num_docked.append(len(pocket_results))

        scores_gen = []
        for docked in pocket_results:
            aff = docked['vina']['dock'][0]['affinity']
            scores_gen.append(aff)
            # breakpoint()
            # print(f"aff: {type(aff)}, score_ref: {type(score_ref)}")
            if isinstance(score_ref, list):
                score_ref = score_ref[0]['vina']['dock'][0]['affinity']
            if aff <= score_ref:
                qed_good.append(docked['chem_results']['qed'])
                sa_good.append(docked['chem_results']['sa'])
        scores_gen = np.array(scores_gen)
        percentage_good.append((scores_gen <= score_ref).mean())

    percentage_good = np.array(percentage_good)
    num_docked = np.array(num_docked)

    print('[HF%%]  Avg: %.2f%% | Med: %.2f%% ' % (np.mean(percentage_good)*100, np.median(percentage_good)*100))
    print('[HF-QED]  Avg: %.4f | Med: %.4f ' % (np.mean(qed_good)*100, np.median(qed_good)*100))
    print('[HF-SA]   Avg: %.4f | Med: %.4f ' % (np.mean(sa_good)*100, np.median(sa_good)*100))
    print('[Success%%] %.2f%% ' % (np.mean(percentage_good > 0)*100, ))

def compute_success_rate(results)->float:
    QED, SA, VinaDock = 0.25, 0.59, -8.18
    count, success = 0, 0
    for pocket in results:
        count += len(pocket)
        for ligand in pocket:
            if ligand["chem_results"]["qed"] > QED and ligand["chem_results"]["sa"] > SA and ligand["vina"]["dock"][0]["affinity"] < VinaDock:
                success += 1
    return success / count

# region atom distance distribution

def get_all_atom_distance(results):
    atom_distance_list = []
    for pocket in results:
        for ligand in pocket:
            mol = ligand['mol']
            mol = Chem.RemoveAllHs(mol)
            pos = mol.GetConformers()[0].GetPositions()
            dist = sci_spatial.distance.pdist(pos, metric='euclidean')
            atom_distance_list += dist.tolist()
    return np.array(atom_distance_list)


def get_c_c_distance(results):
    c_c_distance_list = []
    for pocket in results:
        for ligand in pocket:
            mol = ligand['mol']
            mol = Chem.RemoveAllHs(mol)
            for bond_type, dist in eval_bond_length.bond_distance_from_mol(mol):
                if bond_type[:2] == (6, 6):
                    c_c_distance_list.append(dist)
    return np.array(c_c_distance_list)

def get_all_c_c_distance(results):
    c_c_distance_list = []
    for pocket in results:
        for ligand in pocket:
            mol = ligand['mol']
            mol = Chem.RemoveAllHs(mol)
            for bond_type, dist in eval_bond_length.bond_distance_from_mol(mol):
                if bond_type[0] == 6:
                    c_c_distance_list.append(dist)
    return np.array(c_c_distance_list)

MAP_ATOM_TYPE_AROMATIC_TO_INDEX = {
    (1, False): 0,
    (6, False): 1,
    (6, True): 2,
    (7, False): 3,
    (7, True): 4,
    (8, False): 5,
    (8, True): 6,
    (9, False): 7,
    (15, False): 8,
    (15, True): 9,
    (16, False): 10,
    (16, True): 11,
    (17, False): 12
}

MAP_INDEX_TO_ATOM_TYPE_AROMATIC = {v: k for k, v in MAP_ATOM_TYPE_AROMATIC_TO_INDEX.items()}

def get_all_dist_from_mol(metrics):
    if "all_results" in metrics:
        result = metrics["all_results"]
    else:
        result = (mol for group in metrics for mol in group)

    all_pair_dist = []
    for res in result:
        pred_pos, pred_v = res["pred_pos"], res["pred_v"]
        pred_atom_type = [MAP_INDEX_TO_ATOM_TYPE_AROMATIC[i][0] for i in pred_v.tolist()]
        pair_dist = eval_bond_length.pair_distance_from_pos_v(pred_pos, pred_atom_type)
        all_pair_dist += pair_dist

    return np.array([d[1] for d in all_pair_dist if d[1] < 12])

def get_all_dist(metrics):
    try:
        return get_all_dist_from_mol(metrics)
    except:
        return get_all_atom_distance(metrics)
# In[10]:

LABEL_FONTSIZE = 18
def _plot_other_all_12a(plot_ylabel=False):
    if plot_ylabel:
        plt.ylabel('Density', fontsize=LABEL_FONTSIZE)
    plt.xlabel('Distance of all atom pairs ($\AA$)', fontsize=LABEL_FONTSIZE)
    plt.ylim(0, 0.5)
    plt.xlim(0, 12)

def _plot_other_cc_2a(plot_ylabel=False):
    if plot_ylabel:
        plt.ylabel('Density', fontsize=LABEL_FONTSIZE)
    plt.xlabel('Distance of carbon carbon bond ($\AA$)', fontsize=LABEL_FONTSIZE)
    plt.ylim(0, 12)
    plt.xlim(0, 2)

def _compute_jsd(atom_dist_list, reference_profile, bin):
    profile = eval_bond_length.get_distribution(atom_dist_list, bins=bin)
    return sci_spatial.distance.jensenshannon(reference_profile, profile)

LW = 2
ALPHA = 0.75
def _inline_ploter(ax:Axes, dists:list, rng:tuple, colors:list, reference_profile:np.ndarray, dist_type:str, plot_ylabel=False):
    assert len(colors) == rng[1]-rng[0]
    assert len(rng) == 2
    if dist_type == "All_12A":
        bins = np.linspace(0, 12, 100)
        plot_other = _plot_other_all_12a
        legend_loc = 'upper right'
    else: # "CC_2A" or "CCs_2A"
        bins = np.linspace(0, 2, 100)
        plot_other = _plot_other_cc_2a
        legend_loc = 'upper left'

    ax.hist(dists[0], bins=bins, histtype='step', density=True, lw=LW, color='grey', alpha=ALPHA) # type: ignore
    jsds = []
    for idx, color in zip(range(rng[0], rng[1]), colors):
        jsd = _compute_jsd(dists[idx], reference_profile, bins)
        ax.hist(
            dists[idx], bins=bins, histtype='step', density=True, lw=LW, color=color, alpha=ALPHA,
            label=f"{Globals.variants[idx]} JSD: {jsd:.3f}"
        ) # type: ignore
        jsds.append(jsd)

    labels = [ f"{Globals.variants[idx]}: {jsd:.3f}" for idx, jsd in zip(range(rng[0], rng[1]), jsds) ]
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color=c, marker='o', linestyle='None', markersize=10)
        for c in colors
    ]
    ax.legend(custom_lines, labels, fontsize=15, frameon=False, loc=legend_loc, ncol=1)
    plot_other(plot_ylabel)

DIST_DIST = {
    # "CC_2A": eval_bond_length.get_distribution(get_c_c_distance(Globals.results[0]), bins=np.linspace(0, 2, 100)),
    # "CCs_2A": eval_bond_length.get_distribution(get_all_c_c_distance(Globals.results[0]), bins=np.linspace(0, 2, 100)),
    # "All_12A": eval_bond_length.get_distribution(get_all_atom_distance(Globals.results[0]), bins=np.linspace(0, 12, 100)),
}

def jsd_plot(dist_type:Literal["CCs_2A","CC_2A", "All_12A"] ,figname:str, show:bool=True):
    if dist_type == "CC_2A":
        dist_func = get_c_c_distance
        name_x_list = [(0, 0.75), (0, 1.25), (0, 1.)]
    elif dist_type == "CCs_2A":
        dist_func = get_all_c_c_distance
    elif dist_type == "All_12A":
        dist_func = get_all_atom_distance
        name_x_list = [(3.0, 7.5), (3.0, 8.5), (3.0, 8.)]
    else:
        raise ValueError(f"Invalid dist_type: {dist_type}")
    reference_profile = np.array(DIST_DIST[dist_type])
    dists = [dist_func(r) for r in Globals.results]

    spx, spy = 1, 3
    SUBFIG_SIZE = 5
    plt.figure(figsize=(SUBFIG_SIZE*spy, SUBFIG_SIZE*spx))

    last_start = len(Globals.variants) - 2
    indexes = [(1, 4), (4, last_start), (last_start, len(Globals.variants))]
    colors = ['blue','red', 'orange', 'green', 'red', 'blue', 'pink', "black"]
    for subplot_idx in range(len(indexes)):
        _inline_ploter(
            plt.subplot(spx, spy, subplot_idx+1),
            dists, indexes[subplot_idx], colors[indexes[subplot_idx][0]-1:indexes[subplot_idx][1]-1],
            reference_profile, dist_type, plot_ylabel=subplot_idx==0,
        )

    plt.tight_layout()
    plt.savefig(figname)
    if show:
        plt.show()

# endregion atom distance distribution

# region plot compact

def _plot_label(plot_ylabel=False):
    if plot_ylabel:
        plt.ylabel('Density', fontsize=LABEL_FONTSIZE)
    plt.xlabel('Distance of all atom pairs ($\AA$)', fontsize=LABEL_FONTSIZE)
    plt.ylim(0, 0.5)
    plt.xlim(0, 12)

LW = 2
ALPHA = 0.75
BINS = np.linspace(0, 12, 100)
def _plot_hist(ax:Axes, dists:list, rng:tuple, colors:list, reference_profile:np.ndarray, plot_ylabel=False):
    assert len(colors) == rng[1]-rng[0]
    assert len(rng) == 2

    ax.hist(dists[0], bins=BINS, histtype='step', density=True, lw=LW, color='grey', alpha=ALPHA) # type: ignore
    jsds = []
    for idx, color in zip(range(rng[0], rng[1]), colors):
        jsd = _compute_jsd(dists[idx], reference_profile, BINS)
        ax.hist(
            dists[idx], bins=BINS, histtype='step', density=True, lw=LW, color=color, alpha=ALPHA,
            label=f"{Globals.variants[idx]} JSD: {jsd:.3f}"
        ) # type: ignore
        jsds.append(jsd)

    labels = [ f"{Globals.variants[idx]}: {jsd:.3f}" for idx, jsd in zip(range(rng[0], rng[1]), jsds) ]
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color=c, marker='o', linestyle='None', markersize=10)
        for c in colors
    ]
    ax.legend(custom_lines, labels, fontsize=15, frameon=False, loc="upper right", ncol=1)
    _plot_label(plot_ylabel)


def jsd_plot_compact(figname:str, show:bool=True):
    bins = np.linspace(0, 12, 100)
    reference_profile = eval_bond_length.get_distribution(get_all_atom_distance(Globals.results[0]), bins=bins)
    dists = [get_all_atom_distance(r) for r in Globals.results]

    last_start = len(Globals.variants) - 1
    # indexes = [(3, last_start), (last_start, len(Globals.variants))]
    indexes = [(1, 4), (4, last_start), (last_start, len(Globals.variants))]
    spx, spy = 1, len(indexes)
    SUBFIG_SIZE = 5
    plt.figure(figsize=(SUBFIG_SIZE*spy, SUBFIG_SIZE*spx))
    
    colors = ['red','green', 'blue', 'pink', 'black']
    for subplot_idx in range(spy):
        _plot_hist(
            plt.subplot(spx, spy, subplot_idx+1),
            dists, indexes[subplot_idx], colors[indexes[subplot_idx][0]-1:indexes[subplot_idx][1]-1],
            reference_profile, plot_ylabel=subplot_idx==0,
        )

    plt.tight_layout()
    plt.savefig(figname)
    if show:
        plt.show()

# endregion plot compact

from scripts.diversity import compute_diversity

def div_it(offset=2):
    for idx in range(offset, len(Globals.variants)):
        print(f"{Globals.variants[idx]} diversity: {compute_diversity(Globals.results[idx])}")

def compute_bond_dist(meta):
    bond_counter = {}
    rest = 0
    for group in meta:
        for result in group:
            mol = result['mol']
            bonds = eval_bond_length.bond_distance_from_mol(mol)
            for bond, dist in bonds:
                if bond[0] > bond[1]:
                    bond = (bond[1], bond[0], bond[2])
                if bond[0] != 6 or bond[1] > 8: # C . CNO
                    rest += 1
                    continue
                if bond not in bond_counter:
                    bond_counter[bond] = 0
                bond_counter[bond] += 1
    return bond_counter, rest
def compute_bond_dist_all(offset=2):
    for idx in range(offset, len(Globals.variants)):
        bond_dist, rest_count = compute_bond_dist(Globals.results[idx])
        bond_count = sum(bond_dist.values()) + rest_count
        bond_dist_ratio = {k: v / bond_count for k, v in bond_dist.items()}
        print(f"{Globals.variants[idx]} bond distance: {bond_dist_ratio}")

if __name__ == "__main__":
    for i in range(1, len(Globals.variants)):
        print(f"{Globals.variants[i]}:")
        compute_high_affinity(Globals.results[0], Globals.results[i], 5)
    # for i in range(3, len(Globals.variants)):
    #     print(f"{Globals.variants[i]} success: {compute_success_rate(Globals.results[i])*100:.2f}%")
    # suffix = "_r"
    # vina_energy_plot(f"paper/mvde{suffix}.jpg", 20, show=False)
    # jsd_plot_compact(f"paper/dist_12a_compact{suffix}.jpg", show=False)
    # div_it(1)
    # compute_bond_dist_all(0)
    # jsd_plot("CCs_2A", f"paper/dist_ccs{suffix}.jpg", show=False)
    # jsd_plot("CC_2A", f"paper/dist_cc{suffix}.jpg", show=False)
    # jsd_plot("All_12A", f"paper/dist_12a{suffix}.jpg", show=False)
# filter=80 UP
"""high affinity:
AR:
[HF%]  Avg: 37.94% | Med: 31.00% 
[HF-QED]  Avg: 52.1360 | Med: 51.9444 
[HF-SA]   Avg: 59.7250 | Med: 59.0000 
[Success%] 91.58% 
Pocket2Mol:
[HF%]  Avg: 48.36% | Med: 51.00% 
[HF-QED]  Avg: 56.5935 | Med: 57.5578 
[HF-SA]   Avg: 72.3812 | Med: 72.0000 
[Success%] 88.78% 
TargetDiff:
[HF%]  Avg: 58.11% | Med: 59.09% 
[HF-QED]  Avg: 49.8408 | Med: 49.9754 
[HF-SA]   Avg: 56.3709 | Med: 56.0000 
[Success%] 98.99% 
TargetDiff-r:
[HF%]  Avg: 50.80% | Med: 46.84% 
[HF-QED]  Avg: 43.5305 | Med: 42.9693 
[HF-SA]   Avg: 53.2019 | Med: 53.0000 
[Success%] 100.00% 
TargetDiff+Ours:
[HF%]  Avg: 64.11% | Med: 65.12% 
[HF-QED]  Avg: 45.2801 | Med: 44.8286 
[HF-SA]   Avg: 51.3252 | Med: 51.0000 
[Success%] 100.00% 
BindDM-r:
[HF%]  Avg: 56.74% | Med: 54.30% 
[HF-QED]  Avg: 48.7430 | Med: 49.0874 
[HF-SA]   Avg: 55.7977 | Med: 56.0000 
[Success%] 98.98% 
BindDM+Ours:
[HF%]  Avg: 68.89% | Med: 72.24% 
[HF-QED]  Avg: 52.1775 | Med: 52.7275 
[HF-SA]   Avg: 56.3768 | Med: 56.0000 
[Success%] 100.00% 
"""

"""diversity:
Pocket2Mol diversity: (0.6922656877683115, 0.7108300411926559)
TargetDiff diversity: (0.7165301534965167, 0.7143136579901482)
TargetDiff+Ours diversity: (0.7923381861449451, 0.7741767160826978)
BindDM-r diversity: (0.7327078982594131, 0.7257319618054732)
GapDiff diversity: (0.7481572097058511, 0.745135285501431)
"""
