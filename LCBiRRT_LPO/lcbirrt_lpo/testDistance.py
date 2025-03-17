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
from train_SDF_distance_net import SDFDistanceNetModule
from ljcmp.utils.model_utils_zjw import generate_environment, load_model
from display import qposShow

# from ljcmp.models.validity_network import VoxelValidityNet
from lcbirrt_lpo.network.Distance_network import SDFDistanceNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_float32_matmul_precision('high')

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--dataset_size', type=int, default=500)
parser.add_argument('--max_config_len', type=int, default=200)
parser.add_argument('--voxel_latent_dim', type=int, default=16)
parser.add_argument('--exp_name', '-E', type=str, default='panda_orientation', help='panda_orientation, panda_dual, or panda_triple')

args = parser.parse_args()

exp_info = yaml.load(open('model/{exp_name}/model_info.yaml'.format(exp_name=args.exp_name), 'r'), Loader=yaml.FullLoader)

scene_dir = f'dataset/{args.exp_name}/scene_data'
scene_dir_local = '{}/scene_{:04d}'.format(scene_dir, 1)
scene_data = yaml.load(open(f'{scene_dir_local}/scene.yaml', 'r'), Loader=yaml.FullLoader)
constraint, model_info, condition, update_scene_from_yaml, set_constraint, _ = generate_environment(args.exp_name)
update_scene_from_yaml(scene_data)

constraint_model, _, _, _ = load_model(args.exp_name, exp_info,
                                              load_validity_model=False)

z_dim = exp_info['z_dim']
# z_dim = 14
model = SDFDistanceNetModule(h_dim=exp_info['voxel_validity_model']['h_dim'], z_dim=z_dim, c_dim=exp_info['c_dim'], x_dim=exp_info['x_dim'],
                               scene_range=range(args.dataset_size), max_config_len=args.max_config_len, tag_name=exp_info['constraint_model']['tag'],
                               voxel_latent_dim=args.voxel_latent_dim,
                               batch_size=args.batch_size, exp_name=args.exp_name, test=True, lr=args.lr).to(device)

checkpoint=f'model/{args.exp_name}/distance/last.ckpt'
qlist_sum = model.test(checkpoint, constraint_model)
for qlist in qlist_sum:
    qposShow(qlist, args.exp_name, pc=constraint.planning_scene)


