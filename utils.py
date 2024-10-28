import os
import shutil
import json
from argparse import Namespace

def save_gin_cfg(cfg_path: str, dest_path: str):
    assert os.path.isfile(cfg_path), f"{cfg_path} is not a file"
    shutil.copy(cfg_path, dest_path)


def save_args(args: Namespace, dest_path: str):
    with open(os.path.join(dest_path, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)