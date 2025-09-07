import os
import pickle
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import random


from tqdm.auto import tqdm
import torch
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from utils.data import PDBProtein, parse_sdf_file
def parse_pdb_file(file):
    with open(file, 'r') as fd:
        pdb_block = fd.read()
    protein = PDBProtein(pdb_block)
    return protein

def process_item(key: str, source: str, radius:int):
    pdb_file = os.path.join(source, key, f'{key}_protein.pdb')
    sdf_file = os.path.join(source, key, f'{key}_ligand.sdf')
    if get_pocket(pdb_file, sdf_file, radius):
        return (
            os.path.join(key, f'{key}_pocket{radius}.pdb'), # pocket
            os.path.join(key, f'{key}_ligand.sdf'), # ligand
            os.path.join(key, f'{key}_protein.pdb') # protein
        )
    else:
        return key


def get_pocket(pdb_file: str, sdf_file:str, radius: int = 10):
    """Just write inplace

    :return: whether to parse the pair and get the pocket
    """
    try:
        protein = parse_pdb_file(pdb_file)
        ligand = parse_sdf_file(sdf_file)
    except Exception:
        return False

    pdb_block_pocket = protein.residues_to_pdb_block(
        protein.query_residues_ligand(ligand, radius)
    )
    pocket_file = pdb_file.replace("protein", f"pocket{radius}")

    with open(pocket_file, 'w') as f:
        f.write(pdb_block_pocket)
    return True

def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def split_by_name(index_data, source:str, val:int=0, test:int=128):
    ids = list(range(len(index_data)))
    random.seed(42)
    test_ids = random.sample(ids, test)
    train_ids = list(set(ids) - set(test_ids))
    split = {
        "train": [index_data[i] for i in train_ids],
        "test": [index_data[i] for i in test_ids],
    }
    torch.save(split, os.path.join(source, 'split_by_name.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=str)
    parser.add_argument('--radius', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=32)
    args = parser.parse_args()

    index = []
    fail = []

    mapper = partial(process_item, source=args.source, radius=args.radius)
    keys = os.listdir(args.source)
    pbar = tqdm(desc='Extracting pockets', total=len(keys))
    with ProcessPoolExecutor(args.num_workers) as pool:
        for result in pool.map(mapper, keys):
            if isinstance(result, tuple):
                index.append(result)
            else:
                fail.append(result)
            pbar.update(1)

    print(f"Fail: {len(fail)}\nPass: {len(index)}")
    save_pickle(index, os.path.join(args.source, 'index.pkl'))
    save_pickle(fail, os.path.join(args.source, 'fail.pkl'))
    split_by_name(index, args.source)


