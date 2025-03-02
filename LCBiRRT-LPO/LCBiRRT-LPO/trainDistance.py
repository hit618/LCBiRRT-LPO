import os
import yaml
import pickle
from tqdm import tqdm
import numpy as np
import argparse

import torch
import torch.utils.data
from torch.utils.data import TensorDataset

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from train_voxel_distance_net import VoxelValidityNetModule

from ljcmp.models.validity_network import VoxelValidityNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_float32_matmul_precision('high')

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=60)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.003)
parser.add_argument('--dataset_size', type=int, default=500)
parser.add_argument('--max_config_len', type=int, default=200)
parser.add_argument('--voxel_latent_dim', type=int, default=16)
parser.add_argument('--exp_name', '-E', type=str, default='panda_orientation', help='panda_orientation, panda_dual_orientation, panda_dual, or panda_triple')

args = parser.parse_args()

exp_info = yaml.load(open('model/{exp_name}/model_info.yaml'.format(exp_name=args.exp_name), 'r'), Loader=yaml.FullLoader)

epochs = args.epochs

constraint_model_path = 'models/{exp_name}/{model_path}'.format(exp_name=args.exp_name, model_path=exp_info['constraint_model']['path'])
ckpt_name = constraint_model_path.split('/')[-1].split('.ckpt')[0]

"""check directory exist"""
run_index = 1
dir_base = f'wandb/checkpoints/{args.exp_name}/voxel_distance/'
while True:
    run_name = "H_{h_dim}_{dsz}_{tag}_{ckpt_name}_{run_index}".format(h_dim=exp_info['voxel_validity_model']['h_dim'],
                                                                      dsz=args.dataset_size, tag=exp_info['constraint_model']['tag'],
                                                                      ckpt_name=ckpt_name, run_index=run_index)
    run_path = os.path.join(dir_base, run_name)
    if not os.path.exists(run_path):
        break
    run_index += 1

os.makedirs(run_path, exist_ok=True)

wandb_logger = WandbLogger(project=args.exp_name+'_voxel_distance', name=run_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wandb_logger.experiment.config.update(args)
z_dim = exp_info['z_dim']
# z_dim = 14
model = VoxelValidityNetModule(h_dim=exp_info['voxel_validity_model']['h_dim'], z_dim = z_dim, c_dim=exp_info['c_dim'], x_dim=exp_info['x_dim'],
                               scene_range=range(args.dataset_size), max_config_len=args.max_config_len, tag_name=exp_info['constraint_model']['tag'],
                               voxel_latent_dim=args.voxel_latent_dim,
                               batch_size=args.batch_size, exp_name=args.exp_name, lr=args.lr).to(device)

checkpoint_callback = ModelCheckpoint(dirpath=run_path,
                                      filename='{epoch}-{val_loss_estimation:.2f}',
                                      monitor="val_loss_estimation", mode="min", save_top_k=1, save_last=True)
trainer = Trainer(max_epochs=epochs, logger=wandb_logger, callbacks=[checkpoint_callback], val_check_interval=1.0)
# trainer = Trainer(max_epochs=epochs, logger=None, callbacks=[checkpoint_callback], val_check_interval=1.0)

trainer.fit(model)