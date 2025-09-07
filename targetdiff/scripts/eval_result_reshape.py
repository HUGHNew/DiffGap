import argparse

import torch as th

def extract_order(meta_path:str) -> list:
    meta = th.load(meta_path)
    return [m[0]['ligand_filename'] for m in meta]

def metrics2meta(metrics_path:str, meta_path:str, order_path:str='') -> list:
    results = th.load(metrics_path)['all_results']
    # [{'mol', 'smiles', 'ligand_filename', 'pred_pos', 'pred_v', 'chem_results', 'vina'}]
    meta, cache, flag = [], [], None
    for data in results:
        if not flag:
            flag = data['ligand_filename']
        if flag != data['ligand_filename']:
            meta.append(cache)
            cache, flag = [], None
        cache.append(data)
    if cache:
        meta.append(cache)

    if order_path:
        mapper = {val[0]['ligand_filename']:idx for idx, val in enumerate(meta)}
        meta.append([])
        order_list = extract_order(order_path)
        ordered_meta = [meta[mapper.get(ord, -1)] for ord in order_list]
    else:
        ordered_meta = meta
    th.save(ordered_meta, meta_path)
    return ordered_meta

def meta_check(meta: list) -> bool:
    for outer in meta:
        if len(outer) == 0: continue
        name = outer[0]['ligand_filename']
        for inner in outer:
            assert name == inner['ligand_filename']
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('eval_file', type=str) # 'sampling_results/gapdiff/eval_results/metrics_-1.pt'
    parser.add_argument('meta_file', type=str) # 'sampling_results/gapdiff_test_docked.pt'
    parser.add_argument('meta_ord', type=str, default='')
    args = parser.parse_args()
    meta_check(metrics2meta(args.eval_file, args.meta_file, args.meta_ord))