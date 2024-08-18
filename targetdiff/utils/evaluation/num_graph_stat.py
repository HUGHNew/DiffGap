import argparse
import os

from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose

import utils.train as utils_train
import utils.misc as misc
import utils.transforms as trans
from datasets import get_dataset
from datasets.pl_data import FOLLOW_BATCH

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs_diffusion')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--train_report_iter', type=int, default=200)
    args = parser.parse_args()

    # Load configs
    config = misc.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    misc.seed_all(config.train.seed)

    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(config.data.transform.ligand_atom_mode)
    transform_list = [
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ]
    if config.data.transform.random_rot:
        transform_list.append(trans.RandomRotation())
    transform = Compose(transform_list)

    # Datasets and loaders
    dataset, subsets = get_dataset(
        config=config.data,
        transform=transform
    )
    train_set, val_set = subsets['train'], subsets['test']

    collate_exclude_keys = ['ligand_nbh_list']
    train_loader = DataLoader(
        train_set,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=collate_exclude_keys
    )
    val_loader = DataLoader(val_set, config.train.batch_size, shuffle=False,
                            follow_batch=FOLLOW_BATCH, exclude_keys=collate_exclude_keys)

    train_num_graphs, val_num_graphs = [], []
    print(f"train_size: {len(train_set)}, val_size: {len(val_set)}")
    train_log = len(train_set) // 100
    for idx, batch in enumerate(train_loader):
        p,r = divmod(idx, train_log)
        if r == 0:
            print(f"train_loader: {p}%")
        train_num_graphs.append(batch.protein_element_batch.max().item() + 1)
    
    val_log = len(val_set) // 100
    for idx, batch in enumerate(val_loader):
        p,r = divmod(idx, val_log)
        if r == 0:
            print(f"val_loader: {p}%")
        val_num_graphs.append(batch.protein_element_batch.max().item() + 1)

    print(f'[Train] mean: {(sum(train_num_graphs) / len(train_num_graphs)):.3f},\t max: {max(train_num_graphs)},\t min: {min(train_num_graphs)},\t num: {len(train_num_graphs)}')
    print(f'[Val] mean: {(sum(val_num_graphs) / len(val_num_graphs)):.3f},\t max: {max(val_num_graphs)},\t min: {min(val_num_graphs)},\t num: {len(val_num_graphs)}')
    # [Train] mean: 3.999919993599488,         max: 4,         min: 2
    # [Val] mean: 4.0,         max: 4,         min: 4


    # .3f with BS=4
    # [Train] mean: 4.000,     max: 4,         min: 2,         num: 24998
    # [Val] mean: 4.000,       max: 4,         min: 4,         num: 25
