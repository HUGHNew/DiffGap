# GapDiff

Official implementation for "[Bridging the Gap between Learning and Inference for Diffusion-Based Molecule Generation]" **(AAAI 2025)**.

## Data

The data preparation follows [TargetDiff](https://arxiv.org/abs/2303.03543). 
For more details, please refer to [the repository of TargetDiff](https://github.com/guanjq/targetdiff#data).

## Usage

We use `pipeline.py` to wrap the whole pipeline of training, sampling, and evaluation for both projects.

```bash
python -m pipeline <configs> <sampling_results> [train|sample|eval] [-c resume_from_checkpoint_for_training]
# python -m pipeline configs/training.yml sampling_results/reproduce # for whole pipeline
# python -m pipeline configs/sampling.yml sampling_results/reproduce sample # for pipeline starts from sampling
# python -m pipeline "no matter" sampling_results/reproduce eval # for pipeline for evaluation
```
or you can manually run the script for each stage like TargetDiff or BindDM.

> We remove the `{train,sample,evaluate}.py` in **BindDM**
> because they are just the copies of the `{train,sample,evaluate}_diffusion.py` in `scripts`.

It is worth noting that we provide [script](binddm/scripts/male-es.py) 
for plotting and metrics calculation like High Affinity and Diversity
which is just based on the metrics_-1.pt (meta file) generated by evaluation.

These meta files and checkpoints are coming soon after published.

## Citation

Coming soon.
