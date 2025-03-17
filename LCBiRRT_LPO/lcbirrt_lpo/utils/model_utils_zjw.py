
import torch
import os
import yaml
import tqdm
import copy
import time
import numpy as np
import pickle
import networkx as nx
import math
import concurrent.futures
import random

from ljcmp.models import TSVAE
from ljcmp.models.validity_network import VoxelValidityNet
from lcbirrt_lpo.network.Distance_network import SDFDistanceNet
from ljcmp.models.sampler import Sampler

from ljcmp.planning.sample_region import RegionSampler, LatentRegionSampler

from ljcmp.planning.constrained_bi_rrt import SampleBiasedConstrainedBiRRT,ConstrainedBiRRT
from ljcmp.planning.precomputed_roadmap import PrecomputedRoadmap, PrecomputedGraph
from lcbirrt_lpo.planning.constrained_bi_rrt_latent_jump import ConstrainedLatentBiRRT
from ljcmp.utils.time_parameterization import time_parameterize
from ljcmp.planning.latent_prm import LatentPRM

from scipy.linalg import null_space

from termcolor import colored
from sklearn.neighbors import NearestNeighbors
from ljcmp.utils.generate_environment import generate_environment
from sdf_tools.utils_3d import compute_sdf_and_gradient

import multiprocessing as mp

def load_model(exp_name, model_info, load_validity_model=True,
               load_sample_model = False, load_distance_model = False):
    constraint_model_path = 'model/{exp_name}/{model_path}'.format(exp_name=exp_name, 
                                                                    model_path=model_info['constraint_model']['path'])
    
    model_type = constraint_model_path.split('.')[-1]
    tag = model_info['constraint_model']['tag']

    if model_type == 'pt':
        constraint_model = torch.load(constraint_model_path)

    elif model_type == 'ckpt':
        constraint_model_checkpoint = torch.load(constraint_model_path)
        constraint_model_state_dict = constraint_model_checkpoint['state_dict']
        constraint_model = TSVAE(x_dim=model_info['x_dim'], 
                                 h_dim=model_info['constraint_model']['h_dim'], 
                                 z_dim=model_info['z_dim'], 
                                 c_dim=model_info['c_dim'], 
                                 null_augment=False)
        
        for key in list(constraint_model_state_dict):
            constraint_model_state_dict[key.replace("model.", "")] = constraint_model_state_dict.pop(key)

        constraint_model.load_state_dict(constraint_model_state_dict)
        # save pt
        os.makedirs('model/{exp_name}/weights/{tag}'.format(exp_name=exp_name, tag=tag), exist_ok=True)
        torch.save(constraint_model, 'model/{exp_name}/weights/{tag}/constraint_model.pt'.format(exp_name=exp_name, tag=tag))

    else:
        raise NotImplementedError
    
    constraint_model.eval()

    # tag = model_info['voxel_validity_model']['tag']
    # validity_model_path = 'model/{exp_name}/{model_path}'.format(exp_name=exp_name,
    #                                                               model_path=model_info['voxel_validity_model']['path'])
    # validity_model = VoxelValidityNet(z_dim=model_info['z_dim'],
    #                                   c_dim=model_info['c_dim'],
    #                                   h_dim=model_info['voxel_validity_model']['h_dim'],
    #                                   voxel_latent_dim=model_info['voxel_validity_model']['voxel_latent_dim'])

    if load_validity_model:
        tag = model_info['voxel_validity_model']['tag']
        validity_model_path = 'model/{exp_name}/{model_path}'.format(exp_name=exp_name,
                                                                     model_path=model_info['voxel_validity_model'][
                                                                         'path'])
        validity_model = VoxelValidityNet(z_dim=model_info['z_dim'],
                                          c_dim=model_info['c_dim'],
                                          h_dim=model_info['voxel_validity_model']['h_dim'],
                                          voxel_latent_dim=model_info['voxel_validity_model']['voxel_latent_dim'])

        validity_model_type = validity_model_path.split('.')[-1]
        if validity_model_type == 'pt':
            validity_model = torch.load(validity_model_path)

        elif validity_model_type == 'ckpt':
            validity_model_checkpoint = torch.load(validity_model_path)
            validity_model_state_dict = validity_model_checkpoint['state_dict']
            validity_model_state_dict_z_model = {}
            for key in list(validity_model_state_dict):
                if key.startswith('model.'):
                    validity_model_state_dict_z_model[key.replace("model.", "")] = validity_model_state_dict.pop(key)

            validity_model.load_state_dict(validity_model_state_dict_z_model)
            # save pt
            os.makedirs('model/{exp_name}/weights/{tag}'.format(exp_name=exp_name, tag=tag), exist_ok=True)
            torch.save(validity_model, 'model/{exp_name}/weights/{tag}/voxel_validity_model.pt'.format(exp_name=exp_name, tag=tag))

        else:
            raise NotImplementedError

        validity_model.threshold = model_info['voxel_validity_model']['threshold']
    # else:
    #     validity_model.threshold = 0.0
        validity_model.eval()
    else:
        validity_model = None

    if load_sample_model:
        tag = model_info['sample_model']['tag']
        sample_model_path = 'model/{exp_name}/{model_path}'.format(exp_name=exp_name,
                                                                   model_path=model_info['sample_model']['path'])
        sample_model_type = sample_model_path.split('.')[-1]
        if sample_model_type == 'pt':
            sample_model = torch.load(sample_model_path)

        elif sample_model_type == 'ckpt':
            sample_model_checkpoint = torch.load(sample_model_path)
            sample_model_state_dict = sample_model_checkpoint['state_dict']
            sample_model_state_dict_z_model = {}
            for key in list(sample_model_state_dict):
                if key.startswith('model.'):
                    sample_model_state_dict_z_model[key.replace("model.", "")] = sample_model_state_dict.pop(
                        key)

            sample_model = Sampler(e_dim=model_info['voxel_validity_model']['voxel_latent_dim'],
                                   z_dim=model_info['z_dim'],
                                   c_dim=model_info['c_dim'],
                                   layer_size=model_info['sample_model']['h_dim'])
            sample_model.load_state_dict(sample_model_state_dict_z_model)
            # save pt
            os.makedirs('model/{exp_name}/weights/{tag}'.format(exp_name=exp_name, tag=tag), exist_ok=True)
            torch.save(sample_model,
                       'model/{exp_name}/weights/{tag}/voxel_validity_model.pt'.format(exp_name=exp_name, tag=tag))

        else:
            raise NotImplementedError
        sample_model.eval()
        # return constraint_model, validity_model, sample_model
    else:
        sample_model = None

    if load_distance_model:
        tag = model_info['voxel_distance_model']['tag']
        distance_model_path = 'model/{exp_name}/{model_path}'.format(exp_name=exp_name,
                                                                     model_path=model_info['voxel_distance_model'][
                                                                         'path'])
        distance_model = SDFDistanceNet(z_dim=model_info['z_dim'],
                                          c_dim=model_info['c_dim'],
                                          h_dim=model_info['voxel_distance_model']['h_dim'],
                                          voxel_latent_dim=model_info['voxel_distance_model']['voxel_latent_dim'])

        distance_model_type = distance_model_path.split('.')[-1]
        if distance_model_type == 'pt':
            distance_model = torch.load(distance_model_path)

        elif distance_model_type == 'ckpt':
            distance_model_checkpoint = torch.load(distance_model_path)
            distance_model_state_dict = distance_model_checkpoint['state_dict']
            distance_model_state_dict_z_model = {}
            for key in list(distance_model_state_dict):
                if key.startswith('model.'):
                    distance_model_state_dict_z_model[key.replace("model.", "")] = distance_model_state_dict.pop(key)

            distance_model.load_state_dict(distance_model_state_dict_z_model)
            # save pt
            os.makedirs('model/{exp_name}/weights/{tag}'.format(exp_name=exp_name, tag=tag), exist_ok=True)
            torch.save(distance_model, 'model/{exp_name}/weights/{tag}/voxel_distance_model.pt'.format(exp_name=exp_name, tag=tag))

        else:
            raise NotImplementedError

        # distance_model.threshold = model_info['voxel_distance_model']['threshold']
        distance_model.eval()
    else:
        distance_model = None
    # if load_validity_model and load_sample_model and load_distance_model:
    #     return constraint_model, validity_model, sample_model, distance_model
    # if not load_validity_model and load_sample_model and load_distance_model:
    #     return constraint_model, sample_model, distance_model
    # if load_validity_model and not load_sample_model and load_distance_model:
    #     return constraint_model, validity_model, distance_model
    # if load_validity_model and load_sample_model and not load_distance_model:
    #     return constraint_model, sample_model, validity_model
    # if not load_validity_model and not load_sample_model and load_distance_model:
    #     return constraint_model, distance_model
    # if not load_validity_model and load_sample_model and not load_distance_model:
    #     return constraint_model, sample_model
    # if load_validity_model and not load_sample_model and not load_distance_model:
    #     return constraint_model, validity_model
    # return constraint_model, None
    return constraint_model, validity_model, sample_model, distance_model


def generate_constrained_config(constraint_setup_fn, exp_name,
                                workers_seed_range=range(0,2), dataset_size=30000, 
                                samples_per_condition=10,
                                save_top_k=1, save_every=1000, timeout=1.0,
                                fixed_condition=None,
                                display=False, display_delay=0.5):
    save_dir = f'dataset/{exp_name}/manifold/'
    model_info = yaml.load(open('model/{exp_name}/model_info.yaml'.format(exp_name=exp_name), 'r'), Loader=yaml.FullLoader)

    dataset_size_per_worker = dataset_size // len(workers_seed_range)
    
    def generate_constrained_config_worker(seed, pos):
        np.random.seed(seed)

        save_dir_local = os.path.join(save_dir, str(seed))
        os.makedirs(save_dir_local, exist_ok=True)

        tq = tqdm.tqdm(total=dataset_size_per_worker, position=pos,desc='Generating dataset for seed {}'.format(seed))
        q_dataset = []
        jac_dataset = []
        null_dataset = []
        constraint, set_constraint_by_condition = constraint_setup_fn()
        pc = constraint.planning_scene

        if fixed_condition is not None:
            set_constraint_by_condition(fixed_condition)
            c = fixed_condition

        c_dim = model_info['c_dim']
        numRecord1 = 0
        numRecord2 = 0
        while len(q_dataset) < dataset_size_per_worker:
            if len(q_dataset) - numRecord1 > 50:
                data_list = []
                for seed_i in workers_seed_range:
                    save_dir_local_i = os.path.join(save_dir, str(seed_i))
                    try:
                        data_seed = np.load(f'{save_dir_local_i}/data_{seed_i}.npy')
                        data_list.append(data_seed)
                    except:
                        pass
                data = np.concatenate(data_list)

                q_dataset_norm = normalization(data, c_dim, [model_info['c_lb'], model_info['c_ub']])
                nbrs = initNbrs(q_dataset_norm, 4.5)
                numRecord1 = len(q_dataset)

            if fixed_condition is None:
                c = np.random.uniform(model_info['c_lb'], model_info['c_ub'])
                set_constraint_by_condition(c)

            # q = constraint.sample_valid(pc.is_valid, timeout=timeout)
            #
            # if q is False:
            #     continue
            #
            # if (q > constraint.ub).any() or (q < constraint.lb).any():
            #     continue
            #
            # jac = constraint.jacobian(q)
            # null = null_space(jac)
            # q_dataset.append(np.concatenate((c,q)))
            # jac_dataset.append(jac)
            # null_dataset.append(null)
            #
            # tq.update(1)
            q_temp = []
            for _ in range(samples_per_condition):
                q = constraint.sample_valid(pc.is_valid, timeout=timeout)

                if q is False:
                    continue

                if (q > constraint.ub).any() or (q < constraint.lb).any():
                    continue

                q_temp.append(np.concatenate((c,q)))

                jac = constraint.jacobian(q)
                null = null_space(jac)
                q_dataset.append(np.concatenate((c,q)))
                jac_dataset.append(jac)
                null_dataset.append(null)
                
                tq.update(1)

                if display:
                    # print('display')
                    pc.display(q)
                    time.sleep(display_delay)
                
                if save_every > 0:
                    if len(q_dataset) - numRecord2 > save_every:
                        current_len = len(q_dataset)
                        numRecord2 = len(q_dataset)
                        # delete_len = current_len - save_every * save_top_k
                        try:
                            np.save(f'{save_dir_local}/data_{seed}.npy', np.array(q_dataset))
                            np.save(f'{save_dir_local}/null_{seed}.npy', np.array(null_dataset))
                            # np.save(f'{save_dir_local}/data_{seed}_{current_len}.npy', np.array(q_dataset))
                            # np.save(f'{save_dir_local}/null_{seed}_{current_len}.npy', np.array(null_dataset))
                            
                            # if delete_len > 0:
                            #     os.remove(f'{save_dir_local}/data_{seed}_{delete_len}.npy')
                            #     os.remove(f'{save_dir_local}/null_{seed}_{delete_len}.npy')
                        except:
                            print('save failed')


            if len(q_temp) > 0:
                nearNumList = []
                if len(q_dataset) > 100:
                    for i in range(len(q_temp)):
                        qTempNorm = normalization([q_temp[i]], c_dim, [model_info['c_lb'], model_info['c_ub']])
                        nearNum = getNearPointsNum(nbrs, qTempNorm)
                        if nearNum[0] > 0:
                            nearNumList.append(nearNum[0])
                        else:
                            nearNumList.append(1)
                else:
                    nearNumList = None
                # diffusionNumList = []
                if nearNumList is not None:
                    nearNumList = np.max(nearNumList) / np.array(nearNumList)
                    softNumList = softmax(nearNumList)
                    diffusionNumList = 10 * len(q_temp) * softNumList
                    diffusionNumList = np.round(diffusionNumList).astype(int)
                else:
                    diffusionNumList = 10 * np.ones(len(q_temp))
                    diffusionNumList = np.round(diffusionNumList).astype(int)

                for i in range(len(q_temp)):
                    for j in range(diffusionNumList[i]):
                        c_local_range = (np.array(model_info['c_ub']) - np.array(model_info['c_lb'])) * 0.1
                        c_local = np.random.uniform(-c_local_range, c_local_range)
                        c_local = c + c_local
                        set_constraint_by_condition(c_local)

                        # qqTemp = q_temp[i][0:3] + np.random.uniform([-0.3, -0.3, -0.3], [0.3, 0.3, 0.3])
                        # qq = env.project(qqTemp, 100)
                        # if qq is not None:
                        #     qc = np.concatenate((c_local, qq))
                        #     qcList.append(qc)

                        q_temp_j = q_temp[i].copy()[c_dim:] + np.random.uniform(-0.5 * np.ones(len(q_temp[i])-c_dim), 0.5 * np.ones(len(q_temp[i])-c_dim))
                        # q_temp_j = q_temp[i].copy()[c_dim:]
                        r = constraint.project(q_temp_j)
                        if r is True and pc.is_valid(q_temp_j):
                            jac = constraint.jacobian(q_temp_j)
                            null = null_space(jac)
                            q_dataset.append(np.concatenate((c_local, q_temp_j)))
                            jac_dataset.append(jac)
                            null_dataset.append(null)
                            tq.update(1)

                            if display:
                                # print('display')
                                pc.display(q_temp_j)
                                time.sleep(display_delay)

                            if save_every > 0:
                                if len(q_dataset) - numRecord2 > save_every:
                                    current_len = len(q_dataset)
                                    numRecord2 = len(q_dataset)
                                    # delete_len = current_len - save_every * save_top_k
                                    try:
                                        np.save(f'{save_dir_local}/data_{seed}.npy', np.array(q_dataset))
                                        np.save(f'{save_dir_local}/null_{seed}.npy', np.array(null_dataset))
                                        # np.save(f'{save_dir_local}/data_{seed}_{current_len}.npy', np.array(q_dataset))
                                        # np.save(f'{save_dir_local}/null_{seed}_{current_len}.npy', np.array(null_dataset))

                                        # if delete_len > 0:
                                        #     os.remove(f'{save_dir_local}/data_{seed}_{delete_len}.npy')
                                        #     os.remove(f'{save_dir_local}/null_{seed}_{delete_len}.npy')
                                    except:
                                        print('save failed')


            # if len(q_temp) > 0 :
            #     for _ in range(10):
            #         c_local = (np.array(model_info['c_ub']) - np.array(model_info['c_lb'])) * 0.1
            #         c = c + np.random.uniform(-c_local, c_local)
            #         set_constraint_by_condition(c)
            #         for j in range(len(q_temp)):
            #             q_temp_j = q_temp[j].copy()
            #             r = constraint.project(q_temp_j)
            #             if r is True and pc.is_valid(q_temp_j):
            #                 jac = constraint.jacobian(q_temp_j)
            #                 null = null_space(jac)
            #                 q_dataset.append(np.concatenate((c, q_temp_j)))
            #                 jac_dataset.append(jac)
            #                 null_dataset.append(null)
            #                 tq.update(1)
            #
            #                 if display:
            #                     # print('display')
            #                     pc.display(q_temp_j)
            #                     time.sleep(display_delay)
            #
            #                 if save_every > 0:
            #                     if len(q_dataset) % save_every == 0:
            #                         current_len = len(q_dataset)
            #                         delete_len = current_len - save_every * save_top_k
            #                         try:
            #                             np.save(f'{save_dir_local}/data_{seed}_{current_len}.npy', np.array(q_dataset))
            #                             np.save(f'{save_dir_local}/null_{seed}_{current_len}.npy', np.array(null_dataset))
            #
            #                             if delete_len > 0:
            #                                 os.remove(f'{save_dir_local}/data_{seed}_{delete_len}.npy')
            #                                 os.remove(f'{save_dir_local}/null_{seed}_{delete_len}.npy')
            #                         except:
            #                             print('save failed')
            #                     break

        np.save(f'{save_dir_local}/data_{seed}_{dataset_size_per_worker}.npy', np.array(q_dataset[:dataset_size_per_worker]))
        np.save(f'{save_dir_local}/null_{seed}_{dataset_size_per_worker}.npy', np.array(null_dataset[:dataset_size_per_worker]))
        tq.close()

    p_list = []
    for pos, seed in enumerate(workers_seed_range):
        p = mp.Process(target=generate_constrained_config_worker, args=(seed, pos))
        p.start()
        p_list.append(p)

    for p in p_list:
        p.join()

    print('Merge dataset')

    time.sleep(1)

    data_list = []
    null_list = []
    for seed in workers_seed_range:
        save_dir_local = os.path.join(save_dir, str(seed))
        data_list.append(np.load(f'{save_dir_local}/data_{seed}_{dataset_size_per_worker}.npy'))
        null_list.append(np.load(f'{save_dir_local}/null_{seed}_{dataset_size_per_worker}.npy'))
    
    data = np.concatenate(data_list)
    null = np.concatenate(null_list)

    if fixed_condition is not None:
        np.save(f'{save_dir}/data_fixed_{dataset_size}.npy', data)
        np.save(f'{save_dir}/null_fixed_{dataset_size}.npy', null)
    else:
        np.save(f'{save_dir}/data_{dataset_size}.npy', data)
        np.save(f'{save_dir}/null_{dataset_size}.npy', null)

    print('Done')


# def dataAug(seed, qcNumRange, diffusionNumList, qcList, model_info, constraint, set_constraint_by_condition):
def dataAug(seed, qcNumRange, diffusionNumList, qcList, model_info, nullList):
    # print('process:', seed)
    qcList_T = np.copy(qcList)
    nullList_T = np.copy(nullList)
    qcListAdd = []
    jac_datasetAdd  = []
    null_datasetAdd = []
    jac_dataset = []
    np.random.seed(seed)
    pc = constraint.planning_scene
    # for i in range(qcNumRange[0], qcNumRange[1]):
    #     for j in range(diffusionNumList[i]):
    #         # c_local = np.random.uniform(-0.5, 0.5)
    #         # c_local = qcList[i][3] + c_local
    #         # c_local = np.clip(c_local,0.5,2.5)
    #         c_local_range = [qcList_T[i][3] - 0.5, qcList_T[i][3] + 0.5]
    #         c_local_range = np.clip(c_local_range, 0.5, 2.5)
    #         c_local = np.random.uniform(c_local_range[0], c_local_range[1])
    #         self.setConstrant(c_local)
    #         qqTemp = qcList_T[i][0:3] + np.random.uniform([-0.3, -0.3, -0.3], [0.3, 0.3, 0.3])
    #         qq = self.project(qqTemp, 100)
    #         if qq is not None:
    #             qc = np.concatenate((qq, [c_local]))
    #             qcListAdd.append(qc)

    for i in range(qcNumRange[0], qcNumRange[1]):
        for j in range(diffusionNumList[i]):
            # c_local_range = (np.array(model_info['c_ub']) - np.array(model_info['c_lb'])) * 0.1
            # c_local_range = [qcList_T[i][0:c_dim] - c_local_range, qcList_T[i][0:c_dim] + c_local_range]
            # c_local_range = np.clip(c_local_range, model_info['c_lb'], model_info['c_ub'])
            # c_local = np.random.uniform(c_local_range[0], c_local_range[1])

            c_local = 100 * np.ones(c_dim)
            while any(c_local > model_info['c_ub']) or any(c_local < model_info['c_lb']):
                c_std = 0.2 * np.ones(c_dim)
                c_local = np.random.normal(qcList_T[i][0:c_dim], c_std)

            set_constraint_by_condition(c_local)

            # q_temp_j = qcList_T[i].copy()[c_dim:] + np.random.uniform(-0.3 * np.ones(len(qcList_T[i]) - c_dim),
            #                                                            0.3 * np.ones(len(qcList_T[i]) - c_dim))

            mean = np.zeros(model_info['z_dim'])
            z_std = np.ones(model_info['z_dim']) * 0.5
            epsilon = np.random.normal(mean, z_std)
            q_temp_j = qcList_T[i].copy()[c_dim:] + np.dot(nullList_T[i], epsilon)

            r = constraint.project(q_temp_j)
            if r is True and pc.is_valid(q_temp_j):
                jac = constraint.jacobian(q_temp_j)
                null = null_space(jac)
                qc = np.concatenate((c_local, q_temp_j))
                qcListAdd.append(qc)
                jac_datasetAdd.append(jac)
                # q_dataset.append(np.concatenate((c_local, q_temp_j)))
                # q_dataset = np.concatenate((qcList_T, [qc]))
                jac_dataset.append(jac)
                null_datasetAdd.append(null)
                # null_dataset.append(null)
                # null_dataset = np.concatenate((null_dataset, [null]))
                # print('data num:', len(q_dataset))

    print('qcList num:', seed, len(qcListAdd))
    return [qcListAdd,null_datasetAdd]

def initNbrs(qcList, radius):
    nbrs = NearestNeighbors(radius=radius, metric='euclidean').fit(qcList)
    return nbrs

def getNearPointsNum(nbrs, query_point):

    # time3 = time.time()
    # nbrs = NearestNeighbors(radius=0.5, metric='euclidean').fit(qcList)
    # time4 = time.time()
    # print('time4 - time3:',time4 - time3)

    #
    # query_point = qcList[0] + np.array([0.1, 0.1, 0, 0.1])
    # query_point = np.array([query_point])
    distances, indices = nbrs.radius_neighbors(query_point)
    # print("point num:", len(distances[0]), len(indices[0]))
    # print("Distances of neighbors from the query point:", distances)
    # print("Indices of neighbors from the query point:", indices)
    # time5 = time.time()
    # print('time5 - time4:', time5 - time4)
    numList = []
    for i in range(len(distances)):
        numList.append(len(distances[i]))
    return numList

def normalization(qcList, cDim, cRange):
    qcListNew = np.copy(qcList)
    for i in range(len(qcList)):
        for j in range(cDim):
            qcListNew[i][j] = (qcListNew[i][j] - cRange[0][j]) / (cRange[1][j] - cRange[0][j]) * 2 * math.pi - math.pi
        # qcListNew.append(qcList[i])
    return qcListNew

def softmax(z):
    #
    exp_z = np.exp(z)

    #
    softmax_output = exp_z / np.sum(exp_z)

    return softmax_output

def get_path(seed, constraint, model_info, update_scene_from_yaml, set_constraint, exp_name, device, max_time, start=0,
             end=500, path_size=10, display = False):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # constraint, model_info, _, update_scene_from_yaml, set_constraint, _ = generate_environment(exp_name)

    constraint_model, validity_model = load_model(exp_name, model_info,
                                                  load_validity_model=True)

    constraint_model.to(device=device)
    validity_model.to(device=device)

    z_dim = model_info['z_dim']
    z = torch.normal(mean=torch.zeros([constraint_model.default_batch_size, z_dim]),
                     std=torch.ones([constraint_model.default_batch_size, z_dim])).to(device=device)
    _ = validity_model(z)

    save_dir = f"dataset/{model_info['name']}/scene_data"
    # os.makedirs(save_dir, exist_ok=True)

    path_set = []
    tq_scene = tqdm.tqdm(range(start, end), position=0, leave=False)
    for cnt in tq_scene:
    # for cnt in range(start, end):
        tq_scene.set_description('scene: {:04d}'.format(cnt))
        save_dir_local = '{}/scene_{:04d}'.format(save_dir, cnt)
        if os.path.exists(f'{save_dir_local}/scene.yaml'):
            scene_data = yaml.load(open(f'{save_dir_local}/scene.yaml', 'r'), Loader=yaml.FullLoader)
            update_scene_from_yaml(scene_data)

            inner_tq = tqdm.tqdm(total=path_size, position=1, leave=False)
            # inner_tq = tqdm.tqdm(range(path_size), position=1, leave=False)
            path_set_i = []
            while len(path_set_i) < path_size:
                c_lb = model_info['c_lb']
                c_ub = model_info['c_ub']
                c = np.random.uniform(c_lb, c_ub)
                constraint_model.set_condition(c)
                validity_model.set_condition(c)
                _ = set_constraint(c)

                valid_sets = []
                try_num = 0
                while len(valid_sets) < 2:
                    try_num = try_num + 1
                    if try_num > 10:
                        break
                    xs, zs = constraint_model.sample(1)
                    for x, z in zip(xs, zs):
                        r = constraint.project(x)
                        if r is False:
                            continue
                        if (x < constraint.lb).any() or (x > constraint.ub).any():
                            continue
                        r = constraint.planning_scene.is_valid(x)
                        if r:
                            # valid_sets.append((z, x, c))
                            valid_sets.append(x)
                if try_num > 10:
                    continue
                start_q = valid_sets[0]
                goal_q = valid_sets[1]
                planner = ConstrainedLatentBiRRT(constraint_model, validity_model, constraint, latent_jump=True)
                planner.set_start(start_q)
                planner.set_goal(goal_q)
                r, z_path, q_path, ref_path = planner.solve(max_time=max_time)
                if r:
                    path = {}
                    path['q_path'] = q_path
                    path['z_path'] = z_path
                    path['c'] = c
                    path['scene'] = cnt
                    path_set.append(path)
                    path_set_i.append(path)
                    inner_tq.update(1)
                    # print('seed num:',seed, len(path_set))

                if display and r is True:
                    hz = 20
                    # print(q_path)
                    duration, qs_sample, qds_sample, qdds_sample, ts_sample = time_parameterize(q_path, model_info,
                                                                                                hz=hz)
                    for q in qs_sample:
                        slow_down = 3.0
                        constraint.planning_scene.display(q)
                        time.sleep(1.0 / hz * slow_down)
    return path_set

def generate_path_data(exp_name,
                       constraint,
                       model_info,
                       set_constraint,
                       device,
                       update_scene_from_yaml,
                       max_time,
                       start=0,
                       end=500,
                       path_size=10,
                       worksNum = 10,
                       display = False,
                       ):

    seed = random.sample(range(1, 500),1)
    path_set = get_path(seed[0], constraint, model_info, update_scene_from_yaml, set_constraint,
                        exp_name, device, max_time, start=start,
                        end=end, path_size=path_size, display = display)


    save_dir = f"dataset/{model_info['name']}/path_data"
    os.makedirs(save_dir, exist_ok=True)
    np.save(save_dir + '/pathSet_' + str(start) + '_' + str(end)+'.npy', path_set)


def generate_scene_start_goal(exp_name, constraint, model_info, set_constraint,
                              update_scene_from_yaml, device,
                              start=0, end=500, pair_size=20):
    constraint_model, validity_model = load_model(exp_name, model_info,
                                                  load_validity_model=True)

    constraint_model.to(device=device)
    validity_model.to(device=device)

    z_dim = model_info['z_dim']
    z = torch.normal(mean=torch.zeros([constraint_model.default_batch_size, z_dim]),
                     std=torch.ones([constraint_model.default_batch_size, z_dim])).to(device=device)
    _ = validity_model(z)


    save_dir = f"dataset/{model_info['name']}/scene_data"
    os.makedirs(save_dir, exist_ok=True)

    tq_scene = tqdm.tqdm(range(start, end))
    goal_position_range = [[0.4, -0.65, 0.65],[0.6, -0.45, 0.75]]
    start_position_range = [[0.4, 0.45, 0.65],[0.6, 0.65, 0.75]]


    goal_pose_list = []
    start_pose_list = []
    for cnt in tq_scene:
        tq_scene.set_description('scene: {:04d}'.format(cnt))
        save_dir_local = '{}/scene_{:04d}'.format(save_dir, cnt)

        if os.path.exists(f'{save_dir_local}/scene.yaml'):

            scene_data = yaml.load(open(f'{save_dir_local}/scene.yaml', 'r'), Loader=yaml.FullLoader)
            update_scene_from_yaml(scene_data)
            voxel = np.load(os.path.join(save_dir_local, 'voxel.npy')).flatten()
            validity_model.set_voxel(voxel)

            start_q_sets = []
            goal_q_sets = []
            start_pose_sets = []
            goal_pose_sets = []
            # goal_pose_list.append(scene_data['goal_pose'])
            # start_pose_list.append(scene_data['start_pose'])
            tq = tqdm.tqdm(total=pair_size, leave=False)

            try_num_scene = 0
            while len(start_q_sets) < pair_size and try_num_scene<10:
                # c = np.array(scene_data['c'])
                c = np.random.uniform(model_info['c_lb'], model_info['c_ub'])
                constraint_model.set_condition(c)
                validity_model.set_condition(c)
                _ = set_constraint(c)

                start_position = np.random.uniform(start_position_range[0], start_position_range[1])
                goal_position = np.random.uniform(goal_position_range[0], goal_position_range[1])
                start_pose = np.concatenate((start_position, [0, 0, 0, 1]))
                goal_pose = np.concatenate((goal_position, [0, 0, 0, 1]))
                # goal_pose = np.array(scene_data['goal_pose'])
                # start_pose = np.array(scene_data['start_pose'])

                lrs_start = LatentRegionSampler(constraint_model, constraint, validity_model)
                lrs_start.set_target_pose(start_pose)
                lrs_goal = LatentRegionSampler(constraint_model, constraint, validity_model)
                lrs_goal.set_target_pose(goal_pose)

                tryNum = 0
                timeout = 0.1
                q_start = lrs_start.sample(timeout=timeout)
                q_goal = lrs_goal.sample(timeout=timeout)
                while q_start is None and tryNum <3:
                    q_start = lrs_start.sample(timeout=timeout)
                    tryNum = tryNum + 1
                tryNum = 0
                while q_goal is None and tryNum <3 :
                    q_goal = lrs_goal.sample(timeout=timeout)
                    tryNum = tryNum + 1
                if q_start is not None and q_goal is not None:
                    start_q_sets.append(q_start)
                    goal_q_sets.append(q_goal)
                try_num_scene = try_num_scene + 1
                # print(try_num_scene)

            tq.close()
            save_dir_local_tag = '{}/{}'.format(save_dir_local, model_info['constraint_model']['tag'])
            os.makedirs(save_dir_local_tag, exist_ok=True)
            pickle.dump({'start_q_sets': start_q_sets, 'goal_q_sets': goal_q_sets,
                         'start_pose_sets': start_pose_sets, 'goal_pose_sets': goal_pose_sets},
                        open(f'{save_dir_local_tag}/start_goal_set.pkl', 'wb'))

        else:
            print('scene {} not found'.format(cnt))
            break
    # goal_pose_list = np.array(goal_pose_list)
    # start_pose_list = np.array(start_pose_list)
    # print('x-----------')
    # print(np.max(goal_pose_list[:, 0]))
    # print(np.min(goal_pose_list[:, 0]))
    # print('y-----------')
    # print(np.max(goal_pose_list[:, 1]))
    # print(np.min(goal_pose_list[:, 1]))
    # print('x-----------')
    # print(np.max(start_pose_list[:, 0]))
    # print(np.min(start_pose_list[:, 0]))
    # print('y-----------')
    # print(np.max(start_pose_list[:, 1]))
    # print(np.min(start_pose_list[:, 1]))


def generate_scene_config(constraint, constraint_model, model_info, condition, update_scene_from_yaml, start=0, end=500, config_size=100):
    save_dir = f"dataset/{model_info['name']}/scene_data"
    os.makedirs(save_dir, exist_ok=True)

    tq_scene = tqdm.tqdm(range(start, end))
    for cnt in tq_scene:
        tq_scene.set_description('scene: {:04d}'.format(cnt))
        save_dir_local = '{}/scene_{:04d}'.format(save_dir, cnt)
        if os.path.exists(f'{save_dir_local}/scene.yaml'):

            scene_data = yaml.load(open(f'{save_dir_local}/scene.yaml', 'r'), Loader=yaml.FullLoader)
            update_scene_from_yaml(scene_data)
            invalid_by_projection = 0
            invalid_by_out_of_range = 0
            invalid_by_collision = 0
            valid_sets = []
            invalid_sets = []
            tq = tqdm.tqdm(total=config_size, leave=False)
            
            while len(valid_sets) < config_size or len(invalid_sets) < config_size:
                xs, zs = constraint_model.sample(100)    
                for x, z in zip(xs, zs):
                    r = constraint.project(x)

                    if r is False:
                        if len(invalid_sets) < config_size:
                            invalid_sets.append((z,x,condition))
                        invalid_by_projection += 1
                        continue
                    
                    if (x < constraint.lb).any() or (x > constraint.ub).any():
                        if len(invalid_sets) < config_size:
                            invalid_sets.append((z,x,condition))
                        invalid_by_out_of_range += 1
                        continue

                    r = constraint.planning_scene.is_valid(x)
                    if r:
                        if len(valid_sets) < config_size:
                            valid_sets.append((z, x, condition))
                            tq.update(1)
                    else:
                        if len(invalid_sets) < config_size:
                            invalid_sets.append((z, x, condition))
                        invalid_by_collision += 1
                    
                    tq.set_description(f'valid: {len(valid_sets)}, invalid: {len(invalid_sets)} (P: {invalid_by_projection}, R: {invalid_by_out_of_range}, C: {invalid_by_collision})')
            tq.close()
            save_dir_local_tag = '{}/{}'.format(save_dir_local, model_info['constraint_model']['tag'])
            os.makedirs(save_dir_local_tag, exist_ok=True)
            pickle.dump({'valid_set':valid_sets, 'invalid_set':invalid_sets}, open(f'{save_dir_local_tag}/config.pkl', 'wb'))
                    
        else:
            print('scene {} not found'.format(cnt))
            break


def precomputedRoadmap(nodeNum, exp_name, model_info,update_scene_from_yaml,
              constraint, device='cpu', condition=None,load_validity_model=True):
    # ready for model
    constraint_model, validity_model = load_model(exp_name, model_info,
                                                  load_validity_model=load_validity_model)

    constraint_model.to(device=device)
    validity_model.to(device=device)

    if condition is not None:
        constraint_model.set_condition(condition)
        validity_model.set_condition(condition)

    # warm up
    z_dim = model_info['z_dim']
    z = torch.normal(mean=torch.zeros([constraint_model.default_batch_size, z_dim]),
                     std=torch.ones([constraint_model.default_batch_size, z_dim])).to(device=device)
    _ = validity_model(z)

    scene_dir = f'dataset/{exp_name}/scene_data'

    scene_dir_local = '{}/scene_{:04d}'.format(scene_dir, 501)
    if not os.path.exists(f'{scene_dir_local}/scene.yaml'):
        print(f'{scene_dir_local}/scene.yaml not exist')
    else:
        print(scene_dir_local)

    scene_data = yaml.load(open(f'{scene_dir_local}/scene.yaml', 'r'), Loader=yaml.FullLoader)
    update_scene_from_yaml(scene_data)
    scene = yaml.load(open(os.path.join(scene_dir_local, 'scene.yaml'), 'r'), Loader=yaml.FullLoader)
    voxel = np.load(os.path.join(scene_dir_local, 'voxel.npy')).flatten()
    validity_model.set_voxel(voxel)

    planner = LatentPRM(constraint=constraint,
                        state_dim=model_info['x_dim'],
                        model=constraint_model,
                        validity_model=validity_model,
                        start_region_fn=None,
                        goal_region_fn=None)
    file_name = exp_name + '_LatentRodmap_'
    planner.precomputedRoadmap(nodeNum, file_name)

def compute_sdf(vg, exp_name):
    if exp_name == 'panda_orientation':
        resolution = 0.046875
    elif exp_name == 'panda_dual' or exp_name == 'panda_dual_orientation':
        resolution = 0.03125
    else:
        raise ValueError
    # res = 0.03125
    # shape = [25, 20, 15]
    origin_point = np.array([0, 0, 0], dtype=np.float32)
    # vg = point_cloud_to_voxel_grid(pc, shape, res, origin_point)
    sdf, sdf_grad = compute_sdf_and_gradient(vg, resolution, origin_point)
    return sdf, sdf_grad

def benchmark(exp_name, model_info, method, update_scene_from_yaml, 
              constraint, set_constraint=None, nodeNum=None, device='cpu', condition=None, max_time=300.0,
              use_given_start_goal=False, debug=False, display=False,
              load_validity_model=True, 
              trials=1, test_scene_start_idx=500, num_test_scenes=100,
              use_given_condition=True, use_sample_model = False,
              use_distance_model = False, path_check_method = None, whole_path_check_freq = None):
    """benchmark function

    Args:
        exp_name (str): experiment name
        model_info (dict): model information
        method (str): method name (e.g. 'latent_rrt', 'latent_rrt_latent_jump', 'sampling_rrt', and 'precomputed_graph_rrt')
        update_scene_from_yaml (function): function to update scene from yaml file
        constraint (ConstraintBase): constraint
        device (str, optional): device. Defaults to 'cpu'.
        condition (dict, optional): condition. Defaults to None.
        max_time (float, optional): maximum planning time. Defaults to 500.0.
        use_given_start_goal (bool, optional): use given start and goal. Defaults to False.
        debug (bool, optional): debug mode. Defaults to False.
        display (bool, optional): display mode. Defaults to False.
        load_validity_model (bool, optional): load validity model. Defaults to True.
        trials (int, optional): number of trials. Defaults to 1.
        test_scene_start_idx (int, optional): test scene start index. Defaults to 500.
        num_test_scenes (int, optional): number of test scenes. Defaults to 100.
    """
    # ready for model
    # if use_sample_model:
    #     constraint_model, validity_model, sample_model = load_model(exp_name, model_info,
    #                                                   load_validity_model=load_validity_model,
    #                                                   load_sample_model=True)
    #     sample_model.to(device=device)
    # else:
    #     constraint_model, validity_model = load_model(exp_name, model_info,
    #                                                 load_validity_model=load_validity_model)
    #     sample_model = None

    constraint_model, validity_model, sample_model, distance_model = load_model(exp_name, model_info,
                                                                load_validity_model=load_validity_model,
                                                                load_sample_model=use_sample_model,
                                                                load_distance_model=use_distance_model)
    
    constraint_model.to(device=device)
    validity_model.to(device=device)
    if distance_model is not None:
        distance_model.to(device=device)
    if sample_model is not None:
        sample_model.to(device=device)

    c_dim = model_info['c_dim']
    if condition is not None:
        constraint_model.set_condition(condition)
        validity_model.set_condition(condition)
        if distance_model is not None:
            if c_dim > 0:
                distance_model.set_condition(condition)

    # warm up
    z_dim = model_info['z_dim']
    z = torch.normal(mean=torch.zeros([constraint_model.default_batch_size, z_dim]), 
                    std=torch.ones([constraint_model.default_batch_size, z_dim])).to(device=device)
    _ = validity_model(z)


    if 'precomputed_roadmap' in method:
        tag = model_info['precomputed_roadmap']['tag']

        precomputed_roadmap_path = os.path.join('model', 
                                                exp_name, 
                                                model_info['precomputed_roadmap']['path'])
        
        precomputed_roadmap = nx.read_gpickle(precomputed_roadmap_path)

        print(colored('precomputed_roadmap tag: ', 'green'), tag)
        print(colored('precomputed_roadmap path: ', 'green'), precomputed_roadmap_path)
        print(colored('precomputed_roadmap nodes: ', 'green'), len(precomputed_roadmap.nodes))
        print(colored('precomputed_roadmap edges: ', 'green'), len(precomputed_roadmap.edges))

    if 'precomputed_graph' in method:
        tag = model_info['precomputed_graph']['tag']

        precomputed_graph_path = os.path.join('dataset',
                                               exp_name,
                                               model_info['precomputed_graph']['path'])
        
        configs = np.load(precomputed_graph_path)
        
        planner = PrecomputedGraph(state_dim=model_info['x_dim'], constraint=constraint)
        planner.from_configs(configs[:, model_info['c_dim']:])
        
        precomputed_graph = planner.graph

    if nodeNum is not None:
        precomputed_latent_roadmap_path = exp_name + '_LatentRodmap_' + str(nodeNum)
        precomputed_latent_roadmap = nx.read_gpickle(precomputed_latent_roadmap_path)

        print(colored('precomputed_roadmap path: ', 'green'), precomputed_latent_roadmap)
        print(colored('precomputed_roadmap nodes: ', 'green'), len(precomputed_latent_roadmap.nodes))
        print(colored('precomputed_roadmap edges: ', 'green'), len(precomputed_latent_roadmap.edges))
        
    # benchmark
    scene_dir = f'dataset/{exp_name}/scene_data'

    test_range = range(test_scene_start_idx, test_scene_start_idx + num_test_scenes)
    check_times = []
    test_times = []
    test_paths = []
    test_path_lenghts = []
    test_paths_z = [] # only for latent_rrt
    test_path_refs = [] # only for latent_rrt
    test_suc_cnt = 0
    test_cnt = 0

    q_path_set = []
    z_path_set = []
    scene_set = []

    print(colored('test_range: ', 'green'), test_range)

    tq = tqdm.tqdm(test_range, position=0)


    # c_lb = model_info['c_lb']
    # c_ub = model_info['c_ub']

    for i in tq:
        scene_dir_local = '{}/scene_{:04d}'.format(scene_dir, i)
        print(scene_dir_local)
        if not os.path.exists(f'{scene_dir_local}/scene.yaml'):
            print(f'{scene_dir_local}/scene.yaml not exist')
            break
        if i ==502:
            print(i)
        # # c = np.random.uniform(c_lb, c_ub)
        # c = np.array([0.3, 0.05, 0.9], dtype=np.float32) + np.array([0.01,0.01,0.05])
        # # c = np.array([0.3, 0.05, 0.9], dtype=np.float32)
        # constraint_model.set_condition(c)
        # validity_model.set_condition(c)
        # _ = set_constraint(c)


        scene_data = yaml.load(open(f'{scene_dir_local}/scene.yaml', 'r'), Loader=yaml.FullLoader)
        update_scene_from_yaml(scene_data)

        scene = yaml.load(open(os.path.join(scene_dir_local,  'scene.yaml'), 'r'), Loader=yaml.FullLoader)
        voxel = np.load(os.path.join(scene_dir_local, 'voxel.npy'))
        if use_given_start_goal:
            start_q = np.loadtxt(os.path.join(scene_dir_local, 'start_q.txt'))
            goal_q = np.loadtxt(os.path.join(scene_dir_local, 'goal_q.txt'))
        validity_model.set_voxel(voxel.flatten())
        if distance_model is not None:
            sdf, _ = compute_sdf(voxel,exp_name)
            distance_model.set_sdf(sdf.flatten())
            # distance_model.set_sdf(voxel.flatten())

        start_pose = np.array(scene['start_pose'])
        goal_pose = np.array(scene['goal_pose'])

        inner_tq = tqdm.tqdm(range(trials), position=1, leave=False)

        latent_jump = False
        if 'latent_jump' in method:
            latent_jump = True

        for trial in inner_tq:

            if 'latent_rrt' in method:
                if use_given_start_goal:
                    planner = ConstrainedLatentBiRRT(constraint_model, validity_model, distance_model, constraint, sample_model,
                                                     # start_region_fn=lrs_start.sample, goal_region_fn=lrs_goal.sample,
                                                     latent_jump=latent_jump, path_check_method=path_check_method, whole_path_check_freq=whole_path_check_freq)
                    planner.set_start(start_q)
                    planner.set_goal(goal_q)
                else:
                    lrs_start = LatentRegionSampler(constraint_model, constraint, validity_model)
                    lrs_start.set_target_pose(start_pose)
                    lrs_goal = LatentRegionSampler(constraint_model, constraint, validity_model)
                    lrs_goal.set_target_pose(goal_pose)

                    planner = ConstrainedLatentBiRRT(constraint_model, validity_model, distance_model, constraint, sample_model, latent_jump=latent_jump,
                                                    start_region_fn=lrs_start.sample, goal_region_fn=lrs_goal.sample,
                                                     path_check_method=path_check_method, whole_path_check_freq=whole_path_check_freq)

                planner.max_distance = model_info['planning']['max_distance_q'] / model_info['planning']['alpha']
                planner.max_distance_q = model_info['planning']['max_distance_q']
                planner.off_manifold_threshold = model_info['planning']['off_manifold_threshold']
                planner.p_q_plan = model_info['planning']['p_q_plan']
                planner.delta = model_info['planning']['delta']
                planner.lambda1 = model_info['planning']['lambda']
                planner.max_latent_jump_trials = model_info['planning']['max_latent_jump_trials']
                planner.debug = debug
                r, z_path, q_path, path_ref, q_path_raw = planner.solve(max_time=max_time)
                if r:
                    q_path_set.append(q_path)
                    z_path_set.append(z_path)
                    scene_set.append(i)
                    if q_path_raw is not None:
                        os.makedirs(f'result/pathdata/{exp_name}', exist_ok=True)
                        np.save(f'result/pathdata/{exp_name}/{i}_q_path_raw.npy', q_path_raw)
                        np.save(f'result/pathdata/{exp_name}/{i}_q_path.npy', q_path)
                

            elif method == 'sampling_rrt_latent_sample':
                if use_given_start_goal:
                    planner = SampleBiasedConstrainedBiRRT(state_dim=model_info['x_dim'], model=constraint_model,
                                                           constraint=constraint)
                    planner.set_start(start_q)
                    planner.set_goal(goal_q)
                else:
                    lrs_start = LatentRegionSampler(constraint_model, constraint, validity_model)
                    lrs_start.set_target_pose(start_pose)
                    lrs_goal = LatentRegionSampler(constraint_model, constraint, validity_model)
                    lrs_goal.set_target_pose(goal_pose)
                    planner = SampleBiasedConstrainedBiRRT(state_dim=model_info['x_dim'], model=constraint_model, constraint=constraint,
                                                        start_region_fn=lrs_start.sample, 
                                                        goal_region_fn=lrs_goal.sample)
                planner.max_distance = model_info['planning']['max_distance_q']
                planner.qnew_threshold = model_info['planning']['qnew_threshold']
                planner.delta = model_info['planning']['delta']
                planner.lambda1 = model_info['planning']['lambda']
                planner.debug = debug
                r, q_path = planner.solve(max_time=max_time)

            elif method == 'sampling_rrt_project':
                if use_given_start_goal:
                    planner = ConstrainedBiRRT(state_dim=model_info['x_dim'], constraint=constraint)
                    planner.set_start(start_q)
                    planner.set_goal(goal_q)
                else:
                    lrs_start = LatentRegionSampler(constraint_model, constraint, validity_model)
                    lrs_start.set_target_pose(start_pose)
                    lrs_goal = LatentRegionSampler(constraint_model, constraint, validity_model)
                    lrs_goal.set_target_pose(goal_pose)
                    planner = ConstrainedBiRRT(state_dim=model_info['x_dim'], constraint=constraint,
                                                        start_region_fn=lrs_start.sample,
                                                        goal_region_fn=lrs_goal.sample)
                planner.max_distance = model_info['planning']['max_distance_q']
                planner.qnew_threshold = model_info['planning']['qnew_threshold']
                planner.delta = model_info['planning']['delta']
                planner.lambda1 = model_info['planning']['lambda']
                planner.debug = debug
                r, q_path = planner.solve(max_time=max_time)

            elif method == 'precomputed_roadmap_prm':
                if use_given_start_goal:
                    planner = PrecomputedRoadmap(state_dim=model_info['x_dim'], constraint=constraint)
                    planner.set_graph(graph=precomputed_roadmap)
                    planner.set_start(start_q)
                    planner.set_goal(goal_q)
                else:
                    raise NotImplementedError
                
                planner.debug = debug
                planner.max_distance = model_info['planning']['max_distance_q']
                planner.delta = model_info['planning']['delta']
                planner.lambda1 = model_info['planning']['lambda']

                # r, q_path = planner.solve(max_time=max_time)
                r, q_path, path_node_list, length_list = planner.solve(max_time=max_time)

            elif method == 'precomputed_graph_rrt':
                if use_given_start_goal:
                    planner = PrecomputedGraph(state_dim=model_info['x_dim'], constraint=constraint)
                    planner.set_graph(graph=precomputed_graph)
                    planner.set_start(start_q)
                    planner.set_goal(goal_q)
                else:
                    raise NotImplementedError
                
                planner.debug = debug
                planner.max_distance = model_info['planning']['max_distance_q']
                planner.delta = model_info['planning']['delta']
                planner.lambda1 = model_info['planning']['lambda']
                r, q_path = planner.solve(max_time=max_time)

            elif method == 'latent_prm':
                # precomputed_roadmap_path = exp_name + '_LatentRodmap_' + str(nodeNum)
                # precomputed_roadmap = nx.read_gpickle(precomputed_roadmap_path)

                # print(colored('precomputed_roadmap path: ', 'green'), precomputed_roadmap_path)
                # print(colored('precomputed_roadmap nodes: ', 'green'), len(precomputed_roadmap.nodes))
                # print(colored('precomputed_roadmap edges: ', 'green'), len(precomputed_roadmap.edges))
                if use_given_start_goal:
                    planner = LatentPRM(constraint=constraint,
                                        state_dim=model_info['x_dim'],
                                        model = constraint_model,
                                        validity_model=validity_model,
                                        start_region_fn=None,
                                        goal_region_fn=None)
                    # planner.set_graph(graph=precomputed_roadmap)
                    if nodeNum is not None:
                        planner.set_graph(precomputed_latent_roadmap)
                    planner.set_start(q = start_q)
                    planner.set_goal(q = goal_q)
                else:
                    lrs_start = LatentRegionSampler(constraint_model, constraint, validity_model)
                    lrs_start.set_target_pose(start_pose)
                    lrs_goal = LatentRegionSampler(constraint_model, constraint, validity_model)
                    lrs_goal.set_target_pose(goal_pose)
                    planner = LatentPRM(constraint=constraint,
                                        state_dim=model_info['x_dim'],
                                        model=constraint_model,
                                        validity_model=validity_model,
                                        start_region_fn=lrs_start.sample,
                                        goal_region_fn=lrs_goal.sample)
                    # planner.set_graph(graph=precomputed_roadmap)
                    if nodeNum is not None:
                        planner.set_graph(precomputed_latent_roadmap)
                    # planner.set_start(q=start_q)
                    # planner.set_goal(q=goal_q)

                planner.max_distance = model_info['planning']['max_distance_q'] / model_info['planning']['alpha']
                planner.max_distance_q = model_info['planning']['max_distance_q']
                planner.off_manifold_threshold = model_info['planning']['off_manifold_threshold']
                # planner.p_q_plan = model_info['planning']['p_q_plan']
                planner.delta = model_info['planning']['delta']
                planner.lambda1 = model_info['planning']['lambda']
                planner.latent_dim = model_info['z_dim']
                # planner.max_latent_jump_trials = model_info['planning']['max_latent_jump_trials']
                planner.debug = debug

                # r, q_path = planner.solve(max_time=max_time)
                r, q_path, length, path_node = planner.solve(max_time=max_time)

            elif method == 'computed_roadmap':
                configs_path = os.path.join('dataset', exp_name, model_info['precomputed_graph']['path'])

                configs = np.load(configs_path)
                q_set = configs[:, model_info['c_dim']:]
                roadmapComputer = PrecomputedRoadmap(constraint=constraint,
                                        state_dim=model_info['x_dim'])
                roadmapComputer.compute(q_set, max_num_edges=20)
                file_name = os.path.join('model',
                                               exp_name,
                                               model_info['precomputed_roadmap']['path'])
                roadmapComputer.save_graph(file_name)
            
            else:
                raise NotImplementedError

            if debug:
                print('planning time', planner.solved_time)

            test_cnt += 1
            
            if r is True:
                path_length = np.array([np.linalg.norm(q_path[i+1] - q_path[i]) for i in range(len(q_path)-1)]).sum()
                test_suc_cnt += 1
                solved_time = planner.solved_time
                if 'latent_rrt' in method:
                    check_time = planner.total_check_time
                    check_time_num = planner.check_times
                    # check_for_solution_time = planner.check_for_solution_time
                    # print('solved_time/check_time/check_for_solution_time:', solved_time, check_time, check_for_solution_time, check_times)
                    print('solved_time/check_time/check_num:', solved_time, check_time,
                      check_time_num)

            else:
                if debug:
                    print('failed to find a path')
                q_path = None
                z_path = None
                path_ref = None

                solved_time = -1.0
                path_length = -1.0
                check_time = -1.0

            test_paths.append(q_path)
            test_times.append(solved_time)
            test_path_lenghts.append(path_length)
            if 'latent_rrt' in method:
                check_times.append(check_time)
                test_paths_z.append(z_path)
                test_path_refs.append(path_ref)

            mean_test_times = np.mean(test_times, where=np.array(test_times) > 0)
            mean_test_path_lenghts = np.mean(test_path_lenghts, where=np.array(test_path_lenghts) > 0)
            
            tq.set_description('test suc rate: {:.3f}, avg time: {:.3f}, avg path length: {:.3f}'.format(test_suc_cnt/test_cnt, mean_test_times, mean_test_path_lenghts))

            if display and r is True:
                # for q in q_path:
                #     constraint.planning_scene.display(q)
                #     time.sleep(0.25)
                hz = 20
                # print(q_path)
                duration, qs_sample, qds_sample, qdds_sample, ts_sample = time_parameterize(q_path, model_info, hz=hz)

                if debug:
                    print('duration', duration)

                for q in qs_sample:
                    slow_down = 3.0
                    constraint.planning_scene.display(q)
                    time.sleep(1.0/hz * slow_down)
                    # time.sleep(1)

    save_dir = f"dataset/{model_info['name']}/path_data"
    os.makedirs(save_dir, exist_ok=True)
    pickle.dump({'q_path_set': q_path_set, 'z_path_set': z_path_set, 'scene_set': scene_set,}, open(f'{save_dir}/path_{test_scene_start_idx}_{test_scene_start_idx+num_test_scenes}_hard.pkl', 'wb'))

    test_paths_cartesian = []
    for path in test_paths:
        if path is None:
            test_paths_cartesian.append(None)
            continue
        
        path_cartesian = []
        for q in path:
            cur_idx = 0
            cartesian_vector = []
            for arm_name, dof in zip(constraint.arm_names, constraint.arm_dofs):
                pos, quat = constraint.forward_kinematics(arm_name, q[cur_idx:cur_idx+dof])
                cur_idx += dof
                cartesian_vector.append(np.concatenate([pos, quat]))
                
            cartesian_vector = np.concatenate(cartesian_vector)
            path_cartesian.append(cartesian_vector)
        test_paths_cartesian.append(path_cartesian)


    mean_test_times = np.mean(test_times, where=np.array(test_times) > 0)
    std_test_times = np.std(test_times, where=np.array(test_times) > 0)
    mean_test_path_lenghts = np.mean(test_path_lenghts, where=np.array(test_path_lenghts) > 0)
    std_test_path_lenghts = np.std(test_path_lenghts, where=np.array(test_path_lenghts) > 0)
    if 'latent_rrt' in method:
        mean_check_times = np.mean(check_times, where=np.array(check_times) > 0)
        std_check_times = np.std(check_times, where=np.array(check_times) > 0)

    ret =  {'experiment_name': exp_name,
            'model_tag_name': model_info['constraint_model']['tag'],
            'method': method,
            'use_given_start_goal': use_given_start_goal,
            'max_time': max_time,

            # test scene info
            'test_scene_start_idx': test_scene_start_idx,
            'test_scene_cnt': num_test_scenes,
            
            # result overview
            'test_cnt': test_cnt, 
            'test_suc_cnt': test_suc_cnt, 
            'success_rate': test_suc_cnt/test_cnt,
            'mean_test_times': mean_test_times,
            'std_test_times': std_test_times,
            'mean_test_path_lenghts': mean_test_path_lenghts,
            'std_test_path_lenghts': std_test_path_lenghts,

            # result details
            'test_times': test_times, 
            'test_paths': test_paths, 
            'test_path_lenghts': test_path_lenghts, 
            'test_paths_cartesian': test_paths_cartesian}
    
    if 'latent_rrt' in method:
        ret['test_paths_z'] = test_paths_z
        ret['test_path_refs'] = test_path_refs
        ret['mean_check_times'] = mean_check_times
        ret['std_check_times'] = std_check_times

    return ret
