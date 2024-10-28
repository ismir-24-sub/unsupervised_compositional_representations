from models import iadb
from networks import unet, encoder
import torch
import dataset
import gin
import json
import os
from utils import save_gin_cfg, save_args

from argparse import ArgumentParser

gin.external_configurable(torch.utils.data.DataLoader)

parser = ArgumentParser()
parser.add_argument("--config", "-c", type=str, required=True)
parser.add_argument("--variant", "-v", type=str, required=True, choices=["c", "d"], default="c")
parser.add_argument("--epochs", "-e", type=int, required=True)
parser.add_argument("--device", "-d", type=str, required=True)
parser.add_argument("--restart", "-r", type=str, default=None)
parser.add_argument("--pretrain", "-p", type=str, required=True)
args = parser.parse_args()

cfg_pretrain = [f for f in os.listdir(args.pretrain) if f.endswith(".gin")][0]
cfg_pretrain = os.path.join(args.pretrain, cfg_pretrain)
gin.parse_config_file(cfg_pretrain)
with open(os.path.join(args.pretrain, "args.json"), "r") as f:
    args_pretrain = json.load(f)
pretrain = iadb.DecompLatentIADB(val_dl=None,
                                 checkpoint=False,
                                 log=None, 
                                 device=args.device,
                                 version=args_pretrain["variant"])
pretrain.load_checkpoint(os.path.join(args.pretrain, "checkpoints.pt"))

if args.restart is not None:
    gin_cfg = [f for f in os.listdir(args.restart) if f.endswith(".gin")][0]
    gin_cfg = os.path.join(args.restart, gin_cfg)
    print("Running ", gin_cfg)
    gin.parse_config_file(gin_cfg)
else:
    gin_cfg = args.config
    print("Running ", gin_cfg)
    gin.parse_config_file(gin_cfg)

*_, topdir, middir, fname = gin_cfg.split("/")
model = iadb.MaskingLatentIADB(pre_train_decomp=pretrain,
                               log=f"_iadb_masking_{args.variant}_{topdir}_{middir}_{fname}", 
                               device=args.device, 
                               version=args.variant)

if args.restart is not None:
    print("Restart from", args.restart)
    model.load_checkpoint(os.path.join(args.restart, "checkpoints.pt"))

if model.logging:
    print(f"Logging to {model.writer.log_dir}")
    save_gin_cfg(gin_cfg, model.writer.log_dir)
    save_args(args, model.writer.log_dir)
    model.writer.add_text("gin", gin.operative_config_str())

model.train(epochs=args.epochs)

