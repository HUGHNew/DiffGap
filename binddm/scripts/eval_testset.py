import argparse
import os
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
from rdkit import Chem, RDLogger
import torch
from tqdm.auto import tqdm

from utils.evaluation import scoring_func, eval_bond_length
from utils import misc
from utils.evaluation.docking_qvina import QVinaDockingTask
from utils.evaluation.docking_vina import VinaDockingTask


def print_dict(d, logger):
    for k, v in d.items():
        if v is not None:
            logger.info(f"{k}:\t{v:.4f}")
        else:
            logger.info(f"{k}:\tNone")


def print_ring_ratio(all_ring_sizes, logger):
    for ring_size in range(3, 10):
        n_mol = 0
        for counter in all_ring_sizes:
            if ring_size in counter:
                n_mol += 1
        logger.info(f"ring size: {ring_size} ratio: {n_mol / len(all_ring_sizes):.3f}")


@dataclass
class EvalArgs:
    sample_path: str
    verbose: bool = False
    eval_step: int = -1
    eval_num_examples: Optional[int] = None
    save: bool = True
    protein_root: str = "test_set"
    atom_enc_mode: str = "add_aromatic"
    docking_mode: Literal["qvina", "vina_score", "vina_dock", "none"] = "vina_dock"
    exhaustiveness: int = 16
    source: Literal["crossdocked", "pdbbind"] = "pdbbind"


def resolve_path(path: str, data_source: str) -> Tuple[str, str]:
    """resolve ligand and protein filename by given path and specific data source

    :return: ligand_filename, protein_filename
    """
    if data_source == "crossdocked":
        sdfs = [f for f in os.listdir(path) if f.endswith(".sdf")]
        for sdf in sdfs:
            pdb_file = os.path.join(path, sdf[:10] + ".pdb")
            if os.path.exists(pdb_file):
                return os.path.join(path, sdf), pdb_file
        # it should not happen
        raise ValueError(f"cannot find ligand file for {path}")
    elif data_source == "pdbbind":
        key = path.split(os.sep)[-1]
        return os.path.join(path, f"{key}_ligand.sdf"), os.path.join(
            path, f"{key}_pocket10.pdb"
        )
    else:
        raise ValueError(f"Unsupported data source: {data_source}")


def main(args: EvalArgs):
    logger = misc.get_logger("evaluate")
    if not args.verbose:
        RDLogger.DisableLog("rdApp.*")

    eval_pt = os.path.join(args.sample_path, "eval_results.pt")

    # Load generated data
    results_fn_list = [
        os.path.join(args.sample_path, fn)
        for fn in os.listdir(args.sample_path)
        if os.path.isdir(os.path.join(args.sample_path, fn))
    ]
    if args.eval_num_examples is not None:
        results_fn_list = results_fn_list[: args.eval_num_examples]
    num_examples = len(results_fn_list)
    logger.info(f"Load generated data done! {num_examples} examples in total.")

    num_samples = 0
    if os.path.exists(eval_pt):
        logger.info(f"Load evaluation results from {eval_pt}")
        metrics = torch.load(eval_pt)
        results = metrics['all_results']
        all_bond_dist = metrics['bond_length']
    else:
        results = []
        all_bond_dist = []
        for example_idx, entry in enumerate(tqdm(results_fn_list, desc="Eval")):
            # ['data', 'pred_ligand_pos', 'pred_ligand_v', 'pred_ligand_pos_traj', 'pred_ligand_v_traj']
            # ['pred_ligand_pos_traj']  # [num_samples, num_steps, num_atoms, 3]
            # PDBbind_refined_2020_test/1a69: 1a69_ligand.mol2  1a69_ligand.sdf  1a69_pocket.pdb  1a69_protein.pdb
            ligand_fn, protein_fn = resolve_path(entry, args.source)
            mol = Chem.MolFromMolFile(ligand_fn)
            smiles = Chem.MolToSmiles(mol)

            # chemical and docking check
            try:
                chem_results = scoring_func.get_chem(mol)
                if args.docking_mode == "qvina":
                    with open(protein_fn) as fd:
                        pdb_block = fd.read()
                    vina_task = QVinaDockingTask(pdb_block, mol)
                    vina_results = vina_task.run_sync()
                elif args.docking_mode in ["vina_score", "vina_dock"]:
                    vina_task = VinaDockingTask(protein_fn, mol)
                    score_only_results = vina_task.run(
                        mode="score_only", exhaustiveness=args.exhaustiveness
                    )
                    minimize_results = vina_task.run(
                        mode="minimize", exhaustiveness=args.exhaustiveness
                    )
                    vina_results = {
                        "score_only": score_only_results,
                        "minimize": minimize_results,
                    }
                    if args.docking_mode == "vina_dock":
                        docking_results = vina_task.run(
                            mode="dock", exhaustiveness=args.exhaustiveness
                        )
                        vina_results["dock"] = docking_results
                else:
                    vina_results = None
            except Exception as e:
                if args.verbose:
                    logger.warning(f"Evaluation failed [{example_idx}] {ligand_fn} for {e}")
                continue

            # now we only consider complete molecules as success
            bond_dist = eval_bond_length.bond_distance_from_mol(mol)
            all_bond_dist += bond_dist

            results.append(
                {
                    "mol": mol,
                    "smiles": smiles,
                    "ligand_filename": ligand_fn,
                    "chem_results": chem_results,
                    "vina": vina_results,
                }
            )
        logger.info(f"Evaluate done! {num_samples} samples in total.")

    qed = [r["chem_results"]["qed"] for r in results]
    sa = [r["chem_results"]["sa"] for r in results]
    logger.info("QED:   Mean: %.3f Median: %.3f" % (np.mean(qed), np.median(qed)))
    logger.info("SA:    Mean: %.3f Median: %.3f" % (np.mean(sa), np.median(sa)))
    if args.docking_mode == "qvina":
        vina = [r["vina"][0]["affinity"] for r in results]
        logger.info("Vina:  Mean: %.3f Median: %.3f" % (np.mean(vina), np.median(vina)))
    elif args.docking_mode in ["vina_dock", "vina_score"]:
        vina_score_only = [r["vina"]["score_only"][0]["affinity"] for r in results]
        vina_min = [r["vina"]["minimize"][0]["affinity"] for r in results]
        logger.info(
            "Vina Score:  Mean: %.3f Median: %.3f"
            % (np.mean(vina_score_only), np.median(vina_score_only))
        )
        logger.info(
            "Vina Min  :  Mean: %.3f Median: %.3f"
            % (np.mean(vina_min), np.median(vina_min))
        )
        if args.docking_mode == "vina_dock":
            vina_dock = [r["vina"]["dock"][0]["affinity"] for r in results]
            logger.info(
                "Vina Dock :  Mean: %.3f Median: %.3f"
                % (np.mean(vina_dock), np.median(vina_dock))
            )

    # check ring distribution
    print_ring_ratio([r["chem_results"]["ring_size"] for r in results], logger)

    if args.save:
        print(f"saving result into {args.sample_path}/eval_results.pt")
        torch.save({"bond_length": all_bond_dist, "all_results": results}, eval_pt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_path", type=str)
    parser.add_argument("--verbose", type=eval, default=False)
    parser.add_argument("--eval_step", type=int, default=-1)
    parser.add_argument("--eval_num_examples", type=int, default=None)
    parser.add_argument("--save", type=eval, default=True)
    parser.add_argument("--atom_enc_mode", type=str, default="add_aromatic")
    parser.add_argument(
        "--docking_mode",
        type=str,
        choices=["qvina", "vina_score", "vina_dock", "none"],
        default="vina_dock",
    )
    parser.add_argument("--exhaustiveness", type=int, default=16)
    args = parser.parse_args()

    main(EvalArgs(**dict(args._get_kwargs())))
