# Download **Protein-ligand complexes: The refined set** from https://pdbbind-plus.org.cn/download
# Then you will get *PDBbind_v2020_refined.tar.gz*

import os
import random
import shutil
import pickle

import fire

tar_file = "PDBbind_v2020_refined.tar.gz"
data_path = "PDBbind_refined_2020"
data_test_path = "PDBbind_refined_2020_test"
cmd_untar = f"tar xf {tar_file}"
cmd_rename = "mv refined_set PDBbind_refined_2020"


def unzip_file(file=tar_file):
    assert os.path.exists(file), f"Please download {tar_file} first"
    if os.system(cmd_untar) != 0:
        raise Exception(f"Failed to unzip {file}")
    if os.system(cmd_rename) != 0:
        raise Exception(f"Failed to rename {file}")


def roll_testset(folder: str = data_path, size: int = 128):
    index_pkl = os.path.join(folder, "index.pkl")
    if os.path.exists(index_pkl):
        with open(index_pkl, "rb") as f:
            index = pickle.load(f)
        files = [item[0].split(os.sep)[0] for item in index]
        print("index.pkl exists. Use it to sample testset")
    else:
        files = os.listdir(folder)
    choices = random.sample(files, k=size)

    test_path = os.path.join(folder, "..", data_test_path)
    if not os.path.exists(test_path):
        os.mkdir(test_path)
    assert len(os.listdir(test_path)) == 0, f"{test_path} is not a empty directory"
    for f in choices:
        shutil.copytree(os.path.join(folder, f), os.path.join(test_path, f))


def process_pdbbind(tarball: str, test_size: int):
    if not os.path.exists(data_path):
        unzip_file(tarball)
    random.seed(42)
    roll_testset(size=test_size)

if __name__ == "__main__":
    fire.Fire({
        "pdbbind": process_pdbbind,
    })
