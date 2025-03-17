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

from lcbirrt_lpo.network.Distance_network import SDFDistanceNet
from lcbirrt_lpo.utils.voxel_utils import plotSdf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_float32_matmul_precision('high')

"""pytorch lightning training code for ModelVAE"""
class SDFDistanceNetModule(pl.LightningModule):
    def __init__(self, h_dim, z_dim, x_dim=None, c_dim=0, scene_range=range(500), n_grid=32, max_config_len=1000, tag_name='no_tag',
                voxel_latent_dim=4,
                batch_size=128, lr=1e-3, exp_name='panda_dual_arm_with_fixed_orientation_condition',test=False):
        super(SDFDistanceNetModule, self).__init__()

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

        self.model = SDFDistanceNet(h_dim=h_dim, z_dim=z_dim, c_dim=c_dim, voxel_latent_dim=voxel_latent_dim).to(device)

        if not test:
            """load dataset"""
            data = pickle.load(open(f'dataset/{exp_name}/distance_data/config_distace_sdf.pkl', 'rb'))
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
        if self.c_dim > 0:
            y_hat, voxel_latent = self.model(D0[train_set_len:], voxel=V0[train_set_len:], c=C0[train_set_len:])
        else:
            y_hat, voxel_latent = self.model(D0[train_set_len:], voxel=V0[train_set_len:])
        recon = y_hat.detach().cpu().numpy()
        true = Y0.detach().cpu().numpy()[train_set_len:]

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




