"""
Fix the issue that file lost in the test set.
"""
import glob
import shutil
import os.path as osp

def search_same_prefix(test_set:str) -> list[str]:
    """
    Search the ${__PREFIX__}_{0,1} in the test_set and return the list of __PREFIX__.
    """
    assert osp.exists(test_set)
    _1_files = glob.glob(osp.join(test_set, f"*_1"))
    prefixes = set(osp.basename(f)[:-2] for f in _1_files)
    exists_prefixes = [p for p in prefixes if osp.exists(osp.join(test_set, f"{p}_0"))]
    return exists_prefixes

def One2Zero(test_set:str, prefix:str, dry_run:bool=False) -> None:
    """
    Copy the ${prefix}_1 files to ${prefix}_0.
    """
    assert osp.exists(test_set)
    assert osp.exists(osp.join(test_set, f"{prefix}_0"))
    assert osp.exists(osp.join(test_set, f"{prefix}_1"))
    # for f in os.walk(osp.join(test_set, f"{prefix}_1")):
    for f in glob.glob(osp.join(test_set, f"{prefix}_1", "*")):
        if dry_run:
            print(f"Copying {f} to {osp.join(test_set, f'{prefix}_0')}")
        else:
            shutil.copy(f, osp.join(test_set, f"{prefix}_0"))

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("test_set", type=str, help="The test set directory.")
    parser.add_argument("-d", "--dry_run", action="store_true", help="Dry run.")
    return parser.parse_args()

def main():
    args = get_args()
    prefixes = search_same_prefix(args.test_set)
    for prefix in prefixes:
        One2Zero(args.test_set, prefix, args.dry_run)

if __name__ == "__main__":
    main()