### 0. install mamba ###
curl -L micro.mamba.pm/install.sh -o mamba.install.sh
$SHELL mamba.install.sh
######

### 1. create env ###
# mamba refers to micromamba
mamba env create -n vina.9 python=3.9
pip install torch torchvision tensorboard
pip install pyyaml easydict
mamba install rdkit openbabel python-lmdb -c conda-forge

# For pyg
# the url depends on your torch version. See more: https://github.com/pyg-team/pytorch_geometric?tab=readme-ov-file#installation
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html # for torch==2.1.2
pip install torch_geometric

# For Vina Docking
mamba install swig boost-cpp numpy -c conda-forge
pip install meeko==0.1.dev3 scipy pdb2pqr vina  git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3  # vina==1.2.2 is incompatible with higher version numpy (>=1.20)
# error when compile wheel for vina
# Vina only supports Python<3.10. issue: https://github.com/ccsb-scripps/AutoDock-Vina/issues/255
######


### 2. train model ###
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python -m scripts.train_diffusion configs/training.yml
######


### 3. sample model ###
python -m scripts.sample_diffusion configs/sampling.yml --data_id {i} --result_path {SAMPLING_OUTPUT} # Replace {i} with the index of the data. i should be between 0 and 99 for the testset.
python -m scripts.sample_diffusion configs/sampling.yml [-s new_split_file] # Sample all test_sets [with new_split_file]


CUDA_VISIBLE_DEVICES=0 bash scripts/batch_sample_diffusion.sh configs/sampling.yml outputs 2 0 0 [new_split_file]
CUDA_VISIBLE_DEVICES=1 bash scripts/batch_sample_diffusion.sh configs/sampling.yml outputs 2 1 0 [new_split_file]
######

### 4. evaluate model ###
# vina_dock contains all vina metrics
python -m scripts.evaluate_diffusion {OUTPUT_DIR} --docking_mode vina_dock --protein_root {test_set}
######

