from rdkit import Chem, DataStructs
import numpy as np

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
        for mol in mol_fp_list[i+1:]
    ]
    return sim_list

def compute_diversity(metrics) -> tuple:
    """
    Compute the diversity of the generated molecules based on metrics.pt.
    """
    pocket_sim = [
        np.mean(pocket_similarity([rel['mol'] for rel in pocket]))
        for pocket in metrics
    ]
    sim_mean, sim_med = np.mean(pocket_sim), np.median(pocket_sim)
    return 1 - sim_mean, 1 - sim_med

