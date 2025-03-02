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
from ljcmp.utils.model_utils_zjw import generate_environment, load_model
from display import qposShow


from ljcmp.models.validity_network import VoxelValidityNet

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
# constraint_model_path = 'models/{exp_name}/{model_path}'.format(exp_name=args.exp_name, model_path=exp_info['constraint_model']['path'])
# ckpt_name = constraint_model_path.split('/')[-1].split('.ckpt')[0]

# """check directory exist"""
# run_index = 1
# dir_base = f'wandb/checkpoints/{args.exp_name}/voxel_distance/'
# while True:
#     run_name = "H_{h_dim}_{dsz}_{tag}_{ckpt_name}_{run_index}".format(h_dim=exp_info['voxel_validity_model']['h_dim'],
#                                                                       dsz=args.dataset_size, tag=exp_info['constraint_model']['tag'],
#                                                                       ckpt_name=ckpt_name, run_index=run_index)
#     run_path = os.path.join(dir_base, run_name)
#     if not os.path.exists(run_path):
#         break
#     run_index += 1

# os.makedirs(run_path, exist_ok=True)

# wandb_logger = WandbLogger(project=args.exp_name+'_voxel_distance', name=run_name)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# wandb_logger.experiment.config.update(args)
z_dim = exp_info['z_dim']
# z_dim = 14
model = VoxelValidityNetModule(h_dim=exp_info['voxel_validity_model']['h_dim'], z_dim=z_dim, c_dim=exp_info['c_dim'], x_dim=exp_info['x_dim'],
                               scene_range=range(args.dataset_size), max_config_len=args.max_config_len, tag_name=exp_info['constraint_model']['tag'],
                               voxel_latent_dim=args.voxel_latent_dim,
                               batch_size=args.batch_size, exp_name=args.exp_name, test=True, lr=args.lr).to(device)

# data = np.load(f'dataset/{args.exp_name}/manifold/random_10000.npy')
# data = np.load(f'dataset/{args.exp_name}/manifold/aug_10000.npy')
# null = np.load(f'dataset/{args.exp_name}/manifold/null_10000.npy')
# checkpoint='model/panda_dual/V0.5_H512_B0.1_TYH_TSAFalse_D_10000_ML-1_0_random/last.ckpt'
# checkpoint='model/panda_dual/V0.5_H512_B0.1_TYH_TSATrue_D_10000_ML-1_1_random_tsa/last.ckpt'

# checkpoint='model/panda_dual/V0.5_H512_B0.1_TYH_TSAFalse_D_10000_ML-1_2_random/last.ckpt'
checkpoint=f'model/{args.exp_name}/distance/last.ckpt'
qlist_sum = model.test(checkpoint, constraint_model)
for qlist in qlist_sum:
    qposShow(qlist, args.exp_name, pc=constraint.planning_scene)
# model = VoxelValidityNet(h_dim=exp_info['voxel_validity_model']['h_dim'], z_dim=z_dim, c_dim=exp_info['c_dim'],
#                          voxel_latent_dim=args.voxel_latent_dim).to(device)
# z = np.array([[ 2.071266  ,  1.1856269 ,  0.159054  ,  1.4850245 , -0.13115384,
#         0.62590414,  1.7003075 , -0.3206837 ],
#      [-1.2599807 ,  0.6807824 , -0.1247533 ,  1.8425336 , -1.1016347 ,
#        -0.17676242,  0.0065361 , -1.2823322 ],#false
#      [ 0.9057754 ,  0.5764822 , -0.11475699,  0.31811246, -0.46026295,
#         0.9917694 ,  0.6406807 , -0.9000406 ],
#      [-0.4123424 ,  0.2599481 ,  0.53360075, -1.8194113 ,  1.3091234 ,
#        -1.3447282 ,  0.12250197, -1.0963074 ],
#      [-1.1273451 ,  1.4513611 , -0.49460572, -0.71521026, -0.32529044,
#         0.5482336 , -1.2067385 , -2.0669248 ],
#      [-0.41362256,  0.8839944 ,  0.13365479,  1.787329  , -1.2627608 ,
#         0.33601794, -1.9604917 ,  0.536235  ],
#      [-0.68312407, -1.9808791 ,  1.1939871 ,  1.2982494 ,  0.22780308,
#         1.8667268 , -0.50745016, -0.6650223 ],
#      [ 1.3049575 ,  0.03011544, -1.3219297 , -1.2981503 , -0.8218893 ,
#        -0.48073265, -1.031524  , -0.34740528],
#      [ 0.7706614 , -0.50106055,  1.1220274 , -1.3109318 ,  1.3429586 ,
#        -0.5172223 ,  1.9085027 , -1.3397676 ],
#      [-1.3385066 , -1.4475207 , -0.89365494,  0.98738635, -0.1569792 ,
#         0.38626385, -0.6975405 ,  0.18558459]#false
#      ])
# z_input = torch.from_numpy(z)
# g = torch.zeros(10, 1)
# output, voxel_latent = model(z_input)
# for i in range(10):
#     g[:, i] = torch.autograd.grad(output[:, i].sum(), z_input, retain_graph=True)[0].data
#
# print("GRADIENT: {}")
# print(g[0])

