#!/usr/bin/env python
# coding: utf-8

# In[20]:


import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import spatial as sci_spatial
import torch
from tqdm.auto import tqdm
from rdkit import Chem
import seaborn as sns

# In[2]:


from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  

from utils.evaluation import eval_bond_length


# In[4]:
# # Load Data

# In[43]:
# easydict = dict

class Globals:
    variants = ["crossdocked", "liGAN", "AR", "Pocket2Mol", "TargetDiff", "Ours"]
    # variants = ["crossdocked", "liGAN", "AR", "Pocket2Mol", "TargetDiff", "GapDiff"]
    paths = [os.path.join("pretrained_models", f"{p.lower()}_vina_docked.pt") for p in variants]
    results = [torch.load(p) for p in paths]
    atom_dist = []
    c_c_dist = []


# In[44]:
Globals.results[0] = [[v] for v in Globals.results[0]]
# # Metrics Summary

# # Vina Score

# In[40]:

def vina_score_plot():
    OFFSET = 1
    try:
        # breakpoint()
        vina = [
            np.array([ # liGAN
                np.median([v['vina'][0]['affinity'] if v['vina'] != None else 0. for v in pocket]) if len(pocket) > 0 else 0.
                for pocket in Globals.results[1]
            ])
        ]
        for idx, val in enumerate(Globals.results[OFFSET+1:]): # AR, P2M, TD, GD
            value = np.array([
                np.median([v['vina']['dock'][0]['affinity'] for v in pocket]) if len(pocket) > 0 else 0.
                for pocket in val
            ])
            vina.append(value)
    except:
        breakpoint()

    # In[41]:


    # all_vina = np.stack([our_vina, ar_vina, pocket2mol_vina], axis=0)
    all_vina = np.stack(vina, axis=0)
    best_vina_idx = np.argmin(all_vina, axis=0)


    # In[42]:
    import matplotlib.text


    plt.figure(figsize=(25, 6), dpi=100)

    ax = plt.subplot(1, 1, 1)
    # ax.set_prop_cycle('color', plt.cm.Set1.colors)
    n_data = len(vina[-1])
    fig1_idx = np.argsort(vina[-1])
    ALPHA = 0.75
    POINT_SIZE = 128
    SCALER = [0.75, 0.75, 0.75, 0.75, 1]
    
    for idx, vn in enumerate(vina):
        plt.scatter(np.arange(n_data), vn[fig1_idx], label=f'{Globals.variants[idx+OFFSET]}/{np.mean(best_vina_idx==idx)*100:.0f}% lowest', alpha=ALPHA, s=POINT_SIZE * SCALER[idx])

    # for artist in plt.gca().get_children():
    #     if isinstance(artist, matplotlib.text.Text):
    #         artist.set_fontsize(8)

    # plt.scatter(np.arange(n_data), ar_vina[fig1_idx], label=f'AR (lowest in {np.mean(best_vina_idx==1)*100:.0f}%)', alpha=ALPHA, s=POINT_SIZE * 0.75)
    # plt.scatter(np.arange(n_data), pocket2mol_vina[fig1_idx], label=f'Pocket2Mol (lowest in {np.mean(best_vina_idx==2)*100:.0f}%)', alpha=ALPHA, s=POINT_SIZE * 0.75)
    # plt.scatter(np.arange(n_data), our_vina[fig1_idx], label=f'{MODEL_NAME} (lowest in {np.mean(best_vina_idx==0)*100:.0f}%)', alpha=ALPHA, s=POINT_SIZE)

    # plt.xticks([])
    plt.yticks(fontsize=16)
    for i in range(n_data):
        plt.axvline(i, c='0.1', lw=0.2)
    plt.xlim(-1, 100)
    plt.ylim(-13, -1.5)
    # plt.yticks([-10, -8, -6, -4, -2], [-10, -8, -6, -4, '$\geq$-2'], fontsize=25)
    yticks = [-12, -10, -8, -6, -4, -2]
    plt.yticks(yticks, [str(i) for i in yticks], fontsize=25)
    plt.ylabel('Median Vina Energy', fontsize=30)
    plt.legend(fontsize=24, handletextpad=0.1, loc='lower center', ncol=len(Globals.variants)-OFFSET, frameon=False, bbox_to_anchor=(0.5, -0.3))
    plt.xticks(np.arange(0, 100, 10), [f'target {v}' for v in np.arange(0, 100, 10)], fontsize=20)

    plt.tight_layout()
    plt.savefig('paper/mve.png')
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

def compute_high_affinity(vina_ref, results):
    percentage_good = []
    num_docked = []
    qed_good, sa_good = [], []
    for i in range(100):
        score_ref = vina_ref[i]
        pocket_results = [r for r in results[i] if r['vina'] is not None]
        if len(pocket_results) < 50:
            continue
        num_docked.append(len(pocket_results))

        scores_gen = []
        for docked in pocket_results:
            aff = docked['vina']['dock'][0]['affinity']
            scores_gen.append(aff)
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


# ## Reference

# In[51]:
# # Atom Distance

# In[7]:


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



# In[8]:


def get_all_c_c_distance(results):
    c_c_distance_list = []
    for pocket in results:
        for ligand in pocket:
            mol = ligand['mol']
            mol = Chem.RemoveAllHs(mol)
            for bond_type, dist in eval_bond_length.bond_distance_from_mol(mol):
                if bond_type[:2] == (6, 6):
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

def get_all_dist_from_bonds(metrics):
    # metrics = torch.load(path)
    result = metrics["all_results"]

    all_pair_dist = []
    for res in result:
        pred_pos, pred_v = res["pred_pos"], res["pred_v"]
        pred_atom_type = [MAP_INDEX_TO_ATOM_TYPE_AROMATIC[i][0] for i in pred_v.tolist()]
        pair_dist = eval_bond_length.pair_distance_from_pos_v(pred_pos, pred_atom_type)
        all_pair_dist += pair_dist
    
    # return [d[1] for d in all_pair_dist]
    # return np.array([d[1] for d in all_pair_dist])
    return np.array([d[1] for d in all_pair_dist if d[1] < 12])
    all_pair_length_profile = eval_bond_length.get_pair_length_profile(all_pair_dist)
    all_js_metrics = eval_bond_length.eval_pair_length_profile(all_pair_length_profile)
    return all_js_metrics["JSD_All_12A"]
    for k, v in all_js_metrics.items():
        if v is not None:
            print(f'{k}:\t{v:.4f}')
        else:
            print(f'{k}:\tNone')

# In[10]:

def jsd_atom_plot():
    Globals.atom_dist = [get_all_atom_distance(r) for r in Globals.results[:-1]]
    Globals.atom_dist.append(get_all_dist_from_bonds(Globals.results[-1]))
    # Globals.c_c_dist = [get_all_c_c_distance(r) for r in Globals.results]

    # for idx, ad in enumerate(Globals.atom_dist):
    #     print(f"{Globals.variants[idx]}:{ad.shape}")
    # return

    LW = 2
    LABEL_FONTSIZE = 18
    ALPHA = 0.75
    spx, spy, spi = 1, 3, 1
    SUBFIG_SIZE = 5
    plt.figure(figsize=(SUBFIG_SIZE*spy, SUBFIG_SIZE*spx))

    BINS = np.linspace(0, 12, 100)
    def _plot_other(plot_ylabel=False):
        if plot_ylabel:
            plt.ylabel('Density', fontsize=LABEL_FONTSIZE)
        plt.xlabel('Distance of all atom pairs ($\AA$)', fontsize=LABEL_FONTSIZE)
        plt.ylim(0, 0.5)
        plt.xlim(0, 12)
        
    def _compute_jsd(atom_dist_list, reference_profile):
        # clamp_list = [ad for ad in atom_dist_list if ad < 12]
        # profile = eval_bond_length.get_distribution(clamp_list, bins=BINS)
        profile = eval_bond_length.get_distribution(atom_dist_list, bins=BINS)
        return sci_spatial.distance.jensenshannon(reference_profile, profile)

    from utils.evaluation import eval_bond_length_config
    reference_atom_profile = np.array(eval_bond_length_config.PAIR_EMPIRICAL_DISTRIBUTIONS["All_12A"])


    ax = plt.subplot(spx, spy, spi)
    plt.hist(Globals.atom_dist[0], bins=BINS, histtype='step', density=True, lw=LW, color='gray', alpha=ALPHA, label=Globals.variants[0]) # type: ignore
    plt.hist(Globals.atom_dist[1], bins=BINS, histtype='step', density=True, lw=LW, color='blue', alpha=ALPHA, label=Globals.variants[1]) # type: ignore
    plt.hist(Globals.atom_dist[2], bins=BINS, histtype='step', density=True, lw=LW, color='red', alpha=ALPHA, label=Globals.variants[2]) # type: ignore
    plt.hist(Globals.atom_dist[3], bins=BINS, histtype='step', density=True, lw=LW, color='orange', alpha=ALPHA, label=Globals.variants[3]) # type: ignore
    jsd = _compute_jsd(Globals.atom_dist[1], reference_atom_profile)
    ar_jsd = _compute_jsd(Globals.atom_dist[2], reference_atom_profile)
    p2m_jsd = _compute_jsd(Globals.atom_dist[3], reference_atom_profile)
    jsds = [jsd, ar_jsd, p2m_jsd]
    # ax.text(5, 0.4, f'liGAN JSD: {jsd:.3f}', fontsize=15, weight='bold')
    ax.text(3.0, 0.4, '\n'.join(Globals.variants[1:4]), fontsize=15, weight='bold')
    ax.text(7.5, 0.4, '\n'.join([f"JSD: {jsd:.3f}" for jsd in jsds]), fontsize=15, weight='bold')
    # plt.title(f'liGAN (JSD={jsd:.3f})')
    _plot_other(plot_ylabel=True)
    spi+=1

    ax = plt.subplot(spx, spy, spi)
    plt.hist(Globals.atom_dist[0], bins=BINS, histtype='step', density=True, lw=LW, color='gray', alpha=ALPHA, label=Globals.variants[0]) # type: ignore
    plt.hist(Globals.atom_dist[4], bins=BINS, histtype='step', density=True, lw=LW, color='yellow', alpha=ALPHA, label=Globals.variants[4]) # type: ignore
    jsd = _compute_jsd(Globals.atom_dist[4], reference_atom_profile)
    ax.text(5, 0.4, f'{Globals.variants[4]} JSD:{jsd:.3f}', fontsize=15, weight='bold')
    _plot_other()
    spi+=1

    ax = plt.subplot(spx, spy, spi)
    plt.hist(Globals.atom_dist[0], bins=BINS, histtype='step', density=True, lw=LW, color='gray', alpha=ALPHA, label=Globals.variants[0]) # type: ignore
    plt.hist(Globals.atom_dist[5], bins=BINS, histtype='step', density=True, lw=LW, color='green', alpha=ALPHA, label=Globals.variants[5]) # type: ignore
    jsd = _compute_jsd(Globals.atom_dist[5], reference_atom_profile)
    # jsd = Globals.atom_dist[-1]
    """
    JSD_CC_2A:    0.3956
    JSD_All_12A:  0.0777
    """
    ax.text(5, 0.4, f'{Globals.variants[5]} JSD:{jsd:.3f}', fontsize=15, weight='bold')
    _plot_other()
    spi+=1

    #region old code
    # BINS = np.linspace(0, 2, 100)

    # def _plot_other(plot_ylabel=False):
    #     if plot_ylabel:
    #         plt.ylabel('Density', fontsize=LABEL_FONTSIZE)
    #     plt.xlabel('Distance of carbon carbon bond ($\AA$)', fontsize=LABEL_FONTSIZE)
    #     plt.ylim(0, 12)
    #     plt.xlim(0, 2)

    # reference_cc_profile = eval_bond_length.get_distribution(Globals.c_c_dist[0], bins=BINS)

    # ax = plt.subplot(spx, spy, spi)
    # plt.hist(Globals.c_c_dist[0], bins=BINS, histtype='step', density=True, lw=LW, color='gray', alpha=ALPHA, label=Globals.variants[0])
    # plt.hist(Globals.c_c_dist[1], bins=BINS, histtype='step', density=True, lw=LW, color='blue', alpha=ALPHA, label=Globals.variants[1])
    # plt.hist(Globals.c_c_dist[2], bins=BINS, histtype='step', density=True, lw=LW, color='red', alpha=ALPHA, label=Globals.variants[2])
    # plt.hist(Globals.c_c_dist[3], bins=BINS, histtype='step', density=True, lw=LW, color='orange', alpha=ALPHA, label=Globals.variants[3])
    # ligand_jsd = _compute_jsd(Globals.c_c_dist[1], reference_cc_profile)
    # ar_jsd = _compute_jsd(Globals.c_c_dist[2], reference_cc_profile)
    # p2m_jsd = _compute_jsd(Globals.c_c_dist[3], reference_cc_profile)
    # jsds = [ligand_jsd, ar_jsd, p2m_jsd]
    # ax.text(0.1, 10, '\n'.join(Globals.variants[1:4]), fontsize=15, weight='bold')
    # ax.text(0.8, 10, '\n'.join([f"JSD: {jsd:.3f}" for jsd in jsds]), fontsize=15, weight='bold')
    # _plot_other(plot_ylabel=True)
    # spi+=1

    # ax = plt.subplot(spx, spy, spi)
    # plt.hist(Globals.c_c_dist[0], bins=BINS, histtype='step', density=True, lw=LW, color='gray', alpha=ALPHA, label=Globals.variants[0])
    # plt.hist(Globals.c_c_dist[4], bins=BINS, histtype='step', density=True, lw=LW, color='yellow', alpha=ALPHA, label=Globals.variants[4])
    # jsd = _compute_jsd(Globals.c_c_dist[4], reference_cc_profile)
    # ax.text(0.1, 10, f'{Globals.variants[4]} JSD: {jsd:.3f}', fontsize=15, weight='bold')
    # _plot_other()
    # spi+=1

    # ax = plt.subplot(spx, spy, spi)
    # plt.hist(Globals.c_c_dist[0], bins=BINS, histtype='step', density=True, lw=LW, color='gray', alpha=ALPHA, label=Globals.variants[0])
    # plt.hist(Globals.c_c_dist[5], bins=BINS, histtype='step', density=True, lw=LW, color='green', alpha=ALPHA, label=Globals.variants[5])
    # jsd = _compute_jsd(Globals.c_c_dist[5], reference_cc_profile)
    # ax.text(0.1, 10, f'{Globals.variants[5]} JSD: {jsd:.3f}', fontsize=15, weight='bold')
    # # plt.title(f'{MODEL_NAME} (JSD={jsd:.3f})')
    # _plot_other()
    #endregion old code

    plt.tight_layout()
    plt.savefig('paper/dist_noord.png')
    plt.show()


if __name__ == "__main__":
    jsd_atom_plot()
exit()

# # Bond Distance

# In[11]:


def get_bond_length_profile(results):
    bond_distances = []
    for pocket in results:
        for ligand in pocket:
            mol = ligand['mol']
            mol = Chem.RemoveAllHs(mol)
            bond_distances += eval_bond_length.bond_distance_from_mol(mol)
    return eval_bond_length.get_bond_length_profile(bond_distances)

Globals.reference_bond_length_profile = get_bond_length_profile(Globals.reference_results)
Globals.our_bond_length_profile = get_bond_length_profile(Globals.our_results)
Globals.ar_bond_length_profile = get_bond_length_profile(Globals.ar_results)
Globals.pocket2mol_bond_length_profile = get_bond_length_profile(Globals.pocket2mol_results)
Globals.cvae_bond_length_profile = get_bond_length_profile(Globals.cvae_results)


# In[12]:


REPORT_TYPE = (
    (6,6,1),
    (6,6,2),
    (6,6,4),
    (6,7,1),
    (6,7,2),
    (6,7,4),
    (6,8,1),
    (6,8,2),
    (6,8,4),
)

def _bond_type_str(bond_type) -> str:
    atom1, atom2, bond_category = bond_type
    return f'{atom1}-{atom2}|{bond_category}'

def eval_bond_length_profile(model_profile):
    metrics = {}

    for bond_type in REPORT_TYPE:
        metrics[f'JSD_{_bond_type_str(bond_type)}'] = sci_spatial.distance.jensenshannon(Globals.reference_bond_length_profile[bond_type],model_profile[bond_type])
    return metrics

eval_bond_length_profile(Globals.our_bond_length_profile)


# In[13]:


eval_bond_length_profile(Globals.ar_bond_length_profile)


# In[14]:


eval_bond_length_profile(Globals.pocket2mol_bond_length_profile)


# In[15]:


eval_bond_length_profile(Globals.cvae_bond_length_profile)



# In[16]:


import networkx as nx
from rdkit.Chem.rdchem import BondType
from copy import deepcopy
from collections import OrderedDict


# In[23]:


class RotBondFragmentizer():
    def __init__(self, only_single_bond=True):
        self.type = 'RotBondFragmentizer'
        self.only_single_bond = only_single_bond

    # code adapt from Torsion Diffusion
    def get_bonds(self, mol):
        bonds = []
        G = nx.Graph()
        for i, atom in enumerate(mol.GetAtoms()):
            G.add_node(i)
        # nodes = set(G.nodes())
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            G.add_edge(start, end)
        for e in G.edges():
            G2 = copy.deepcopy(G)
            G2.remove_edge(*e)
            if nx.is_connected(G2): continue
            l = list(sorted(nx.connected_components(G2), key=len)[0])
            if len(l) < 2: continue
            # n0 = list(G2.neighbors(e[0]))
            # n1 = list(G2.neighbors(e[1]))
            if self.only_single_bond:
                bond_type = mol.GetBondBetweenAtoms(e[0], e[1]).GetBondType()
                if bond_type != BondType.SINGLE:
                    continue
            bonds.append((e[0], e[1]))
        return bonds

    def fragmentize(self, mol, dummyStart=1, bond_list=None):
        if bond_list is None:
            # get bonds need to be break
            bonds = self.get_bonds(mol)
        else:
            bonds = bond_list
        # whether the molecule can really be break
        if len(bonds) != 0:
            bond_ids = [mol.GetBondBetweenAtoms(x, y).GetIdx() for x, y in bonds]
            bond_ids = list(set(bond_ids))
            # break the bonds & set the dummy labels for the bonds
            dummyLabels = [(i + dummyStart, i + dummyStart) for i in range(len(bond_ids))]
            break_mol = Chem.FragmentOnBonds(mol, bond_ids, dummyLabels=dummyLabels)
            dummyEnd = dummyStart + len(dummyLabels) - 1
        else:
            break_mol = mol
            bond_ids = []
            dummyEnd = dummyStart - 1

        return break_mol, bond_ids, dummyEnd
    

def get_clean_mol(mol):
    rdmol = deepcopy(mol)
    for at in rdmol.GetAtoms():
        at.SetAtomMapNum(0)
        at.SetIsotope(0)
    Chem.RemoveStereochemistry(rdmol)
    return rdmol


def replace_atom_in_mol(ori_mol, src_atom, dst_atom):
    mol = deepcopy(ori_mol)
    m_mol = Chem.RWMol(mol)
    for atom in m_mol.GetAtoms():
        if atom.GetAtomicNum() == src_atom:
            atom_idx = atom.GetIdx()
            m_mol.ReplaceAtom(atom_idx, Chem.Atom(dst_atom))
    return m_mol.GetMol()


def ff_optimize(ori_mol, addHs=False, enable_torsion=False):
    mol = deepcopy(ori_mol)
    Chem.GetSymmSSSR(mol)
    if addHs:
        mol = Chem.AddHs(mol, addCoords=True)
    mp = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94s')
    if mp is None:
        return (None, )

    # turn off angle-related terms
    mp.SetMMFFOopTerm(enable_torsion)
    mp.SetMMFFAngleTerm(True)
    mp.SetMMFFTorsionTerm(enable_torsion)

    # optimize unrelated to angles
    mp.SetMMFFStretchBendTerm(True)
    mp.SetMMFFBondTerm(True)
    mp.SetMMFFVdWTerm(True)
    mp.SetMMFFEleTerm(True)
    
#     try:
    ff = AllChem.MMFFGetMoleculeForceField(mol, mp)
    energy_before_ff = ff.CalcEnergy()
    ff.Minimize()
    energy_after_ff = ff.CalcEnergy()
    # print(f'Energy: {energy_before_ff} --> {energy_after_ff}')
    energy_change = energy_before_ff - energy_after_ff
    Chem.SanitizeMol(ori_mol)
    Chem.SanitizeMol(mol)
    rmsd = rdMolAlign.GetBestRMS(ori_mol, mol)
#     except:
#         return (None, )
    return energy_change, rmsd, mol


def frag_analysis_from_mol_list(input_mol_list):
    all_frags_dict = {}
    sg = RotBondFragmentizer()
    for mol in tqdm(input_mol_list):
        frags, _, _ = sg.fragmentize(mol)
        frags = [get_clean_mol(f) for f in Chem.GetMolFrags(frags, asMols=True)]

        for frag in frags:
            num_atoms = frag.GetNumAtoms() - Chem.MolToSmiles(frag).count('*')
            if 2 < num_atoms < 10:
                if num_atoms not in all_frags_dict:
                    all_frags_dict[num_atoms] = []

                mol = deepcopy(frag)
                mol_hs = replace_atom_in_mol(mol, src_atom=0, dst_atom=1)
                mol_hs = Chem.RemoveAllHs(mol_hs)
                all_frags_dict[num_atoms].append(mol_hs)
    
    all_frags_dict = OrderedDict(sorted(all_frags_dict.items()))
    all_rmsd_by_frag_size = {}
    for k, mol_list in all_frags_dict.items():
        n_fail = 0
        all_energy_diff, all_rmsd = [], []
        for mol in mol_list:
            ff_results = ff_optimize(mol, addHs=True, enable_torsion=False)
            if ff_results[0] is None:
                n_fail += 1
                continue
            energy_diff, rmsd, _, = ff_results
            all_energy_diff.append(energy_diff)
            all_rmsd.append(rmsd)
        print(f'Num of atoms: {k} ({n_fail} of {len(mol_list)} fail):   '
              f'\tEnergy {np.mean(all_energy_diff):.2f} / {np.median(all_energy_diff):.2f}' 
              f'\tRMSD   {np.mean(all_rmsd):.2f} / {np.median(all_rmsd):.2f}'
             )
        all_rmsd_by_frag_size[k] = all_rmsd
    return all_frags_dict, all_rmsd_by_frag_size


# In[24]:


targetdiff_mols = [r['mol'] for pr in Globals.our_results for r in pr]
ar_mols = [r['mol'] for pr in Globals.ar_results for r in pr]
pocket2mol_mols = [r['mol'] for pr in Globals.pocket2mol_results for r in pr]
cvae_mols = [r['mol'] for pr in Globals.cvae_results for r in pr]


# In[25]:


_, ours_rigid_rmsd = frag_analysis_from_mol_list(targetdiff_mols)
_, ar_rigid_rmsd = frag_analysis_from_mol_list(ar_mols)
_, pocket2mol_rigid_rmsd = frag_analysis_from_mol_list(pocket2mol_mols)
_, cvae_rigid_rmsd = frag_analysis_from_mol_list(cvae_mols)


# In[26]:


def construct_df(rigid_dict):
    df = []
    for k, all_v in rigid_dict.items():
        for v in all_v:
            df.append({'f_size': k, 'rmsd': v})
    return pd.DataFrame(df)

# sns.set(style="darkgrid")
sns.set_style("white")
sns.set_palette("muted")

tmp_1 = construct_df(ours_rigid_rmsd)
tmp_1['model'] = MODEL_NAME
tmp_2 = construct_df(ar_rigid_rmsd)
tmp_2['model'] = 'AR'
tmp_3 = construct_df(pocket2mol_rigid_rmsd)
tmp_3['model'] = 'Pocket2Mol'
tmp_4 = construct_df(cvae_rigid_rmsd)
tmp_4['model'] = 'liGAN'

viz_df = pd.concat([tmp_1, tmp_2, tmp_3, tmp_4]).reset_index()
viz_df = viz_df.query('3<=f_size<=9')

LABEL_FONTSIZE = 24
TICK_FONTSIZE = 16
LEGEND_FONTSIZE = 20
plt.figure(figsize=(12, 8))
sns.boxplot(x='f_size', y='rmsd', hue='model', data=viz_df, hue_order=('liGAN', 'AR', 'Pocket2Mol', MODEL_NAME), showfliers = False)
plt.xlabel('Fragment Size', fontsize=LABEL_FONTSIZE)
plt.ylabel('Median RMSD ($\AA{}$)', fontsize=LABEL_FONTSIZE)
plt.xticks(fontsize=TICK_FONTSIZE)
plt.yticks(fontsize=TICK_FONTSIZE)
plt.legend(frameon=False, fontsize=LEGEND_FONTSIZE)
# plt.savefig('output_figures/rigid_rmsd.pdf')
plt.show()
