from datetime import datetime
from typing import Literal

from scripts.train_diffusion import main as train_main, TrainArgs
from scripts.sample_diffusion import main as sample_main, SampleArgs
from scripts.evaluate_diffusion import main as eval_main, EvalArgs
import utils.misc as misc


import fire
import yaml

# def save_sample_config
def get_sample_config_name(train_config:str, sample_root="./configs/") -> str:
    train_cfg = misc.load_config(train_config)
    date = datetime.now().strftime('%b%d')
    if "bias" in train_cfg:
        bias_cfg = train_cfg.bias
        if isinstance(bias_cfg.method, list):
            method = "_".join(bias_cfg.method)
        else:
            method = bias_cfg.method
        return f"{sample_root}/{method}_min{bias_cfg.min_p}_{bias_cfg.update_method}_{date}.yml"
    else:
        return f"{sample_root}/reproduce_{date}.yml"

def main(train_config:str, resule_path:str, launch_stage:Literal["train", "sample", "eval"]="train", split:str=None, checkpoint:str='', protein_root:str="test_set"):
    pipeline_start = datetime.now()
    stage_mapper = {
        "train": 0,
        "sample": 10,
        "eval": 20,
    }
    stage = stage_mapper[launch_stage]
    print(f"pipeline start at stage: {launch_stage}. Using config file: {train_config}")

    if stage == stage_mapper["train"]:
        train_start = datetime.now()
        train_args = TrainArgs(
            config=train_config,
            checkpoint=checkpoint
        )
        ckpt = train_main(train_args)
        train_endup = datetime.now()
        print(f"Training: [{train_start}-{train_endup}]={misc.convert_seconds((train_endup-train_start).total_seconds())}")


        with open("./configs/sampling.yml") as spl:
            sample_config = yaml.safe_load(spl)
        sample_config["model"]["checkpoint"] = ckpt
        sample_save = get_sample_config_name(train_config)
        # get train
        with open(sample_save, "w") as sf:
            yaml.safe_dump(sample_config, sf)
    else:
        sample_save = train_config

    if stage < stage_mapper["eval"]:
        sample_start = datetime.now()
        sample_args = SampleArgs(
            config=sample_save,
            result_path=resule_path
        )
        if split is not None:
            sample_args.split = split
        sample_main(sample_args)
        sample_endup = datetime.now()
        print(f"Sampling: [{sample_start}-{sample_endup}]={misc.convert_seconds((sample_endup-sample_start).total_seconds())}")


    eval_start = datetime.now()
    eval_args = EvalArgs(resule_path, protein_root=protein_root, verbose=True)
    eval_main(eval_args)
    eval_endup = datetime.now()
    print(f"Evaluation: [{eval_start}-{eval_endup}]={misc.convert_seconds((eval_endup-eval_start).total_seconds())}")

    print(f"Pipeline: [{pipeline_start}-{eval_endup}]= {misc.convert_seconds((eval_endup-pipeline_start).total_seconds())}")


if __name__ == "__main__":
    fire.Fire(main)