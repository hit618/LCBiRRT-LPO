import os
import yaml
import pickle
from tqdm import tqdm
import numpy as np
import argparse

import torch
import torch.utils.data
from torch.utils.data import TensorDataset
from torch.autograd import Variable

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from ljcmp.models.validity_network import VoxelValidityNet
from voxel_utils import plotSdf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_float32_matmul_precision('high')

"""pytorch lightning training code for ModelVAE"""
class VoxelValidityNetModule(pl.LightningModule):
    def __init__(self, h_dim, z_dim, x_dim=None, c_dim=0, scene_range=range(500), n_grid=32, max_config_len=1000, tag_name='no_tag',
                voxel_latent_dim=4,
                batch_size=128, lr=1e-3, exp_name='panda_dual_arm_with_fixed_orientation_condition',test=False):
        super(VoxelValidityNetModule, self).__init__()

        """model parameters"""
        self.batch_size = batch_size
        self.lr = lr
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.c_dim = c_dim
        print('z_dim', z_dim)
        print('c_dim', c_dim)
        print('h_dim', h_dim)

        self.exp_name = exp_name

        self.model = VoxelValidityNet(h_dim=h_dim, z_dim=z_dim, c_dim=c_dim, voxel_latent_dim=voxel_latent_dim).to(device)

        if not test:
            """load dataset"""
            data = pickle.load(open(f'dataset/{exp_name}/scene_data/config_distace_sdf.pkl', 'rb'))
            # data = pickle.load(open(f'dataset/{exp_name}/scene_data/config_distace_vg.pkl', 'rb'))
            data_c = data['data_c']
            data_q = data['data_q']
            data_z = data['data_z']
            data_sdf = data['data_sdf']
            # data_vg = data['data_vg']
            data_dis = data['data_dis']
            data_dis_n = []
            for i in range(len(data_dis)):
                data_dis_n.append([data_dis[i]])
                # if data_dis[i] > 0:
                #     data_dis_n.append([1])
                # else:
                #     data_dis_n.append([0])
            if self.c_dim>0:
                C0 = torch.from_numpy(np.array(data_c)).float()
            Y0 = torch.from_numpy(np.array(data_dis_n)).float()
            D0 = torch.from_numpy(np.array(data_z)).float()
            V0 = torch.from_numpy(np.array([data_sdf[k].flatten() for k in range(len(data_sdf))])).float()
            # V0 = torch.from_numpy(np.array([data_vg[k].flatten() for k in range(len(data_vg))])).float()

            # V0 = torch.zeros((len(scene_range), n_grid**3), dtype=torch.float32)
            # D0 = torch.zeros((len(scene_range), max_config_len, z_dim), dtype=torch.float32)
            # Y0 = torch.zeros((len(scene_range), max_config_len, 1), dtype=torch.float32)
            # C0 = torch.zeros((len(scene_range), max_config_len, c_dim), dtype=torch.float32)
            #
            # scene_data_path = f'dataset/{exp_name}/scene_data/'
            # for i in tqdm(scene_range, desc='loading scene data'):
            #     scene_name = 'scene_{:04d}'.format(i)
            #     scene = yaml.load(open(os.path.join(scene_data_path, scene_name, 'scene.yaml'), 'r'), Loader=yaml.FullLoader)
            #     if c_dim > 0:
            #         c = scene['c']
            #     joint_configs = pickle.load(open(os.path.join(scene_data_path, scene_name, tag_name, 'config.pkl'), 'rb'))
            #     config_len = min(len(joint_configs['valid_set']), len(joint_configs['invalid_set']), max_config_len//2)
            #     assert (len(joint_configs['valid_set']) >= config_len) and (len(joint_configs['invalid_set']) >= config_len)
            #
            #     valid_set = joint_configs['valid_set'][:config_len]
            #     invalid_set = joint_configs['invalid_set'][:config_len]
            #     for j in range(config_len):
            #         D0[i, j, :] = torch.from_numpy(valid_set[j][0])
            #         Y0[i, j] = 1
            #         D0[i, j+config_len, :] = torch.from_numpy(invalid_set[j][0])
            #         Y0[i, j+config_len] = 0
            #
            #         if c_dim > 0:
            #             C0[i, j, :] = torch.from_numpy(valid_set[j][2])
            #             C0[i, j+config_len, :] = torch.from_numpy(invalid_set[j][2])
            #
            #     voxel = np.load(os.path.join(scene_data_path, scene_name, 'voxel.npy')).flatten()
            #     # numpy float to bool
            #     # voxel = voxel > 0.5
            #     V0[i, :] = torch.from_numpy(voxel)

            V0 = V0.to(device)
            D0 = D0.to(device)
            Y0 = Y0.to(device)
            if self.c_dim > 0:
                C0 = C0.to(device)

            len_total = len(V0)
            train_set_len = int(len_total*0.9)

            """dataset with device"""
            if self.c_dim > 0:
                train_dataset = TensorDataset(D0[:train_set_len],
                                              Y0[:train_set_len],
                                              C0[:train_set_len],
                                               V0[:train_set_len])
                validation_dataset = TensorDataset(D0[train_set_len:],
                                                   Y0[train_set_len:],
                                                   C0[train_set_len:],
                                                   V0[train_set_len:])
            else:
                train_dataset = TensorDataset(D0[:train_set_len],
                                              Y0[:train_set_len],
                                              V0[:train_set_len])
                validation_dataset = TensorDataset(D0[train_set_len:],
                                                   Y0[train_set_len:],
                                                   V0[train_set_len:])
            """dataloader with device"""
            self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            self.validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size)

            print('train', len(self.train_loader))
            print('validation', len(self.validation_loader))


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # v = v.repeat_interleave(x.shape[1], dim=0)
        #
        # x = x.reshape(-1, self.z_dim)
        # y = y.reshape(-1, 1)
        # if self.c_dim > 0:
        #     c = c.reshape(-1, self.c_dim)

        if self.c_dim > 0:
            x = batch[0]
            y = batch[1]
            c = batch[2]
            v = batch[3]
            y_hat, voxel_latent = self.model(x, voxel=v, c=c)
        else:
            x = batch[0]
            y = batch[1]
            v = batch[2]
            y_hat, voxel_latent = self.model(x, voxel=v)

        loss_estimation = self.model.disLoss(y, y_hat)
        # loss_estimation = self.model.loss(x, c, y, y_hat)

        self.log('train_loss_estimation', loss_estimation)

        return loss_estimation

    def validation_step(self, batch, batch_idx):
        # v = v.repeat_interleave(x.shape[1], dim=0)
        #
        # x = x.reshape(-1, self.z_dim)
        # y = y.reshape(-1, 1)
        # if self.c_dim > 0:
        #     c = c.reshape(-1, self.c_dim)

        if self.c_dim > 0:
            x = batch[0]
            y = batch[1]
            c = batch[2]
            v = batch[3]
            y_hat, voxel_latent = self.model(x, voxel=v, c=c)
        else:
            x = batch[0]
            y = batch[1]
            v = batch[2]
            y_hat, voxel_latent = self.model(x, voxel=v)

        loss_estimation = self.model.disLoss(y, y_hat)
        # loss_estimation = self.model.loss(x, c, y, y_hat)

        # accuracy = torch.sum(torch.round(y_hat) == y).float() / y.shape[0]

        self.log('val_loss_estimation', loss_estimation)
        # self.log('val_accuracy', accuracy)

        return loss_estimation

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        sheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)
        return [optimizer], [sheduler]

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.validation_loader

    def test(self, ckpt, constraint_model=None):
        # if type == 'aug':
        #     checkpoint = torch.load('model/threeLink/V0.5_H256_B0.1_TYH_TSAFalse_D_10000_ML-1_5_aug/last.ckpt')
        # elif type == 'random':
        #     checkpoint = torch.load('model/threeLink/V0.5_H256_B0.1_TYH_TSAFalse_D_10000_ML-1_6_random/last.ckpt')
        # else:
        #     raise ValueError

        checkpoint = torch.load(ckpt)

        self.load_state_dict(checkpoint['state_dict'])
        self.eval()

        data = pickle.load(open(f'dataset/{self.exp_name}/scene_data/config_distace_sdf.pkl', 'rb'))
        # data = pickle.load(open(f'dataset/{self.exp_name}/scene_data/config_distace_vg.pkl', 'rb'))
        # print(data['data_z'][0:10])
        data_c = data['data_c']
        data_q = data['data_q']
        data_z = data['data_z']
        data_sdf = data['data_sdf']
        # data_vg = data['data_vg']
        data_dis = data['data_dis']
        data_dis_n = []
        for i in range(len(data_dis)):
            data_dis_n.append([data_dis[i]])
            # if data_dis[i] > 0:
            #     data_dis_n.append([1])
            # else:
            #     data_dis_n.append([0])

        if self.c_dim > 0:
            C0 = torch.from_numpy(np.array(data_c)).float()
        Y0 = torch.from_numpy(np.array(data_dis_n)).float()
        D0 = torch.from_numpy(np.array(data_z)).float()
        V0 = torch.from_numpy(np.array([data_sdf[k].flatten() for k in range(len(data_sdf))])).float()
        # V0 = torch.from_numpy(np.array([data_vg[k].flatten() for k in range(len(data_vg))])).float()
        V0 = V0.to(device)
        D0 = D0.to(device)
        Y0 = Y0.to(device)
        if self.c_dim > 0:
            C0 = C0.to(device)

        len_total = len(V0)
        train_set_len = int(len_total * 0.9)

        # if self.exp_name=='threeLink':
        #     C0 = np.array(self.data[:, -c_dim:], dtype=np.float32)
        #     D0 = np.array(self.data[:, :-c_dim], dtype=np.float32)
        # else:
        #     C0 = np.array(self.data[:, :c_dim], dtype=np.float32)
        #     D0 = np.array(self.data[:, c_dim:], dtype=np.float32)
        # xIn = torch.from_numpy(D0[self.train_set_len:]).to(device)
        # cIn = torch.from_numpy(C0[self.train_set_len:]).to(device)
        # (z_mean, var), (q_z, p_z), z, x, x_recon = self.model(xIn, c=cIn, eval=True)
        if self.c_dim > 0:
            y_hat, voxel_latent = self.model(D0[train_set_len:], voxel=V0[train_set_len:], c=C0[train_set_len:])
        else:
            y_hat, voxel_latent = self.model(D0[train_set_len:], voxel=V0[train_set_len:])
        recon = y_hat.detach().cpu().numpy()
        true = Y0.detach().cpu().numpy()[train_set_len:]
        # self.qcListError(D0[self.train_set_len:], recon)
        # losses = self.model.loss(x, x_recon, q_z, p_z)
        # qc_recon = np.concatenate((recon,C0[self.train_set_len:]),axis=1)
        # print(np.mean(true - recon))
        # np.save(f'dataset/{self.exp_name}/manifold/dis_recon.npy', recon)

        # g = torch.zeros(10, 8)
        # z = torch.tensor([[ 2.071266  ,  1.1856269 ,  0.159054  ,  1.4850245 , -0.13115384,
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
        #      ]).cuda()
        # z = z0.to(device)
        # dataNum = 0
        qlist_sum = []
        for dataNum in range(201,220):
            print('-------------------------------')
            print('true dis:', data_dis[dataNum])
            # plotSdf(data_sdf[dataNum])
            qlist=[]
            z = D0[dataNum]
            z.requires_grad_()
            # z.retain_grad()
            # z = Variable(D0[0:10], requires_grad=True)
            v = Variable(V0[dataNum], requires_grad=True)
            if self.c_dim > 0:
                c = Variable(C0[dataNum], requires_grad=True)
            for k in range(50):
                if self.c_dim > 0:
                    output, voxel_latent = self.model(z.unsqueeze(dim=0), voxel=v.unsqueeze(dim=0), c=c.unsqueeze(dim=0))
                else:
                    output, voxel_latent = self.model(z.unsqueeze(dim=0), voxel=v.unsqueeze(dim=0))
                # output, voxel_latent = self.model(D0[0], voxel=V0[0], c=C0[0])
                output[0].sum().backward()
                print('distance:',output)
                # print('grad:',z.grad)
                z = z + 0.4 * z.grad
                z.requires_grad_()
                z.retain_grad()
                # print('z:',z)
                if self.c_dim > 0:
                    constraint_model.set_condition(c.squeeze(dim=0).cpu().data.numpy())
                q = constraint_model.to_state(z.cpu().data.numpy())
                # print('q:',q)
                qlist.append(q)
                if output.cpu().data.numpy()[0][0]>0.1:
                    break
            qlist_sum.append(qlist)
                # np.save(f'dataset/{self.exp_name}/manifold/dis_recon.npy', recon)
        # qposShow(qlist, args.exp_name, pc=constraint.planning_scene)
        return qlist_sum

        # for i in range(8):
        #     g[:, i] = torch.autograd.grad(output[:,i], D0[0], retain_graph=True)[0].data
        #
        # print("GRADIENT: {}")
        # print(g[0])


# parser = argparse.ArgumentParser()
# parser.add_argument('--epochs', type=int, default=100)
# parser.add_argument('--batch_size', type=int, default=32)
# parser.add_argument('--lr', type=float, default=0.005)
# parser.add_argument('--dataset_size', type=int, default=500)
# parser.add_argument('--max_config_len', type=int, default=200)
# parser.add_argument('--voxel_latent_dim', type=int, default=16)
# parser.add_argument('--exp_name', '-E', type=str, default='panda_dual', help='panda_orientation, panda_dual, or panda_triple')
#
# args = parser.parse_args()
#
# exp_info = yaml.load(open('model/{exp_name}/model_info.yaml'.format(exp_name=args.exp_name), 'r'), Loader=yaml.FullLoader)
#
# epochs = args.epochs
#
# constraint_model_path = 'models/{exp_name}/{model_path}'.format(exp_name=args.exp_name, model_path=exp_info['constraint_model']['path'])
# ckpt_name = constraint_model_path.split('/')[-1].split('.ckpt')[0]
#
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
#
# os.makedirs(run_path, exist_ok=True)
#
# wandb_logger = WandbLogger(project=args.exp_name+'_voxel_distance', name=run_name)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# wandb_logger.experiment.config.update(args)
# # z_dim = exp_info['z_dim']
# z_dim = 14
# model = VoxelValidityNetModule(h_dim=exp_info['voxel_validity_model']['h_dim'], z_dim = z_dim, c_dim=exp_info['c_dim'], x_dim=exp_info['x_dim'],
#                                scene_range=range(args.dataset_size), max_config_len=args.max_config_len, tag_name=exp_info['constraint_model']['tag'],
#                                voxel_latent_dim=args.voxel_latent_dim,
#                                batch_size=args.batch_size, exp_name=args.exp_name, lr=args.lr).to(device)
#
# checkpoint_callback = ModelCheckpoint(dirpath=run_path,
#                                       filename='{epoch}-{val_accuracy:.2f}',
#                                       monitor="val_accuracy", mode="max", save_top_k=1, save_last=True)
# trainer = Trainer(max_epochs=epochs, logger=wandb_logger, callbacks=[checkpoint_callback], val_check_interval=1.0)
#
# trainer.fit(model)

