# export DISPLAY=:1.0
from srmt.planning_scene import PlanningScene, VisualSimulator
from ljcmp.utils.generate_environment import generate_environment
from voxel_utils import *
from ljcmp.utils.model_utils_zjw import  load_model
import numpy as np
import os
import sys
import argparse
import pickle
import yaml
import random

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', '-E', type=str, default='panda_dual', help='panda_orientation, panda_dual, panda_dual_orientation, panda_triple')
parser.add_argument('--seed', type=int, default=1107)
parser.add_argument('--scene_start_idx', type=int, default=0)
parser.add_argument('--num_scenes', type=int, default=500)
parser.add_argument('--num_each_scenes', type=int, default=100)
parser.add_argument('--device', type=str, default='cuda')

args = parser.parse_args()

# device = args.device
# constraint, model_info, condition, update_scene_from_yaml, set_constraint, _ = generate_environment(args.exp_name)
# constraint_model, _ = load_model(args.exp_name, model_info,
#                                               load_validity_model=False)
# constraint_model.to(device=device)
#
# if args.exp_name == 'panda_dual':
#     pc = PlanningScene(arm_names=['panda_arm_2', 'panda_arm_1'], arm_dofs=[7, 7])
#     link_name_list = [
#         ['panda_1_link0', 'panda_1_link1', 'panda_1_link2', 'panda_1_link3', 'panda_1_link4', 'panda_1_link5',
#          'panda_1_link6', 'panda_1_link7', 'panda_1_link8', 'panda_1_hand'],
#         ['panda_2_link0', 'panda_2_link1', 'panda_2_link2', 'panda_2_link3', 'panda_2_link4', 'panda_2_link5',
#          'panda_2_link6', 'panda_2_link7', 'panda_2_link8', 'panda_2_hand']]
# else:
#     raise ValueError
# current_file_path = os.path.abspath(__file__)
# current_file_path = os.path.abspath(__file__).rpartition('/')[0]
# sys.path.append(current_file_path)
# dataPath = os.path.join(current_file_path, 'dataset/panda_dual_old/manifold/data_10000.npy')
# vgPath = os.path.join(current_file_path,  'dataset/panda_dual/scene_data')
tag_name = 'w_tsa'
# c_dim = 3
if args.exp_name == 'panda_orientation':
    dataPath = f'dataset/{args.exp_name}_old/manifold/data_fixed_10000.npy'
else:
    dataPath = f'dataset/{args.exp_name}_old/manifold/data_10000.npy'

scene_data_path = f'dataset/{args.exp_name}/scene_data'
qdata = np.load(dataPath, allow_pickle=True)
sdfList = []
vgList = []
positionListSum1 = []
positionListSum2 = []
zList=[]

data_c = []
data_q = []
data_z = []
data_vg = []
data_sdf = []
data_dis = []
pc = None
model_info = None
def reset():
    global sdfList, vgList, positionListSum1, positionListSum2, zList, pc, link_name_list, model_info
    device = args.device
    constraint, model_info, condition, update_scene_from_yaml, set_constraint, _ = generate_environment(args.exp_name)
    constraint_model, _, _, _ = load_model(args.exp_name, model_info,
                                     load_validity_model=False)
    constraint_model.to(device=device)

    if args.exp_name == 'panda_dual' or args.exp_name == 'panda_dual_orientation':
        pc = PlanningScene(arm_names=['panda_arm_2', 'panda_arm_1'], arm_dofs=[7, 7])
        link_name_list = [
            ['panda_1_link0', 'panda_1_link1', 'panda_1_link2', 'panda_1_link3', 'panda_1_link4', 'panda_1_link5',
             'panda_1_link6', 'panda_1_link7', 'panda_1_link8', 'panda_1_hand'],
            ['panda_2_link0', 'panda_2_link1', 'panda_2_link2', 'panda_2_link3', 'panda_2_link4', 'panda_2_link5',
             'panda_2_link6', 'panda_2_link7', 'panda_2_link8', 'panda_2_hand']]
    elif args.exp_name == 'panda_orientation':
        pc = PlanningScene(arm_names = model_info['arm_names'], arm_dofs = model_info['arm_dofs'], base_link=model_info['base_link'])
        link_name_list = [
            ['panda_link0', 'panda_link1', 'panda_link2', 'panda_link3', 'panda_link4', 'panda_link5',
             'panda_link6', 'panda_link7', 'panda_link8', 'panda_hand']]
    else:
        raise ValueError

    for i in range(args.scene_start_idx, args.scene_start_idx + args.num_scenes):
        scene_dir_local = '{}/scene_{:04d}'.format(scene_data_path, i)
        vg = np.load(os.path.join(scene_dir_local, 'voxel.npy'))
        plotVoxel(vg)
        sdf, sdf_grad = compute_sdf(vg, args.exp_name)
        plotSdf(sdf)
        sdfList.append(sdf)
        vgList.append(vg)
    print('compute sdf done.')

    # for i in range(len(qdata)):
    # for i in range(200,300):
    #     print(i)
    #     if model_info['c_dim']>0:
    #         q = qdata[i][3:]
    #         c = qdata[i][0:3]
    #     else:
    #         q = qdata[i]
    #         c=None
    #     positionList1, positionList2 = getCoordinate(q, c, pc, link_name_list, constraint,args.exp_name)
    #     positionListSum1.append(positionList1)
    #     positionListSum2.append(positionList2)
    #     if model_info['c_dim'] > 0:
    #         constraint_model.set_condition(c)
    #     z = constraint_model.to_latent(q)
    #     zList.append(z)
    #     time.sleep(0.3)
    # print('compute positionList done.')
    return constraint, model_info, condition, update_scene_from_yaml, set_constraint


def makeDisSet1():
    for i in range(args.scene_start_idx, args.scene_start_idx + args.num_scenes):
        scene_name = 'scene_{:04d}'.format(i)
        scene = yaml.load(open(os.path.join(scene_data_path, scene_name, 'scene.yaml'), 'r'), Loader=yaml.FullLoader)
        if c_dim > 0:
            c = scene['c']
        joint_configs = pickle.load(open(os.path.join(scene_data_path, scene_name, tag_name, 'config.pkl'), 'rb'))
        config_len = min(len(joint_configs['valid_set']), len(joint_configs['invalid_set']))
        # config_len = min(len(joint_configs['valid_set']), len(joint_configs['invalid_set']), max_config_len // 2)
        # assert (len(joint_configs['valid_set']) >= config_len) and (len(joint_configs['invalid_set']) >= config_len)

        valid_set = joint_configs['valid_set'][:config_len]
        invalid_set = joint_configs['invalid_set'][:config_len]
        scene_dir_local = '{}/scene_{:04d}'.format(scene_data_path, i)
        vg = np.load(os.path.join(scene_dir_local, 'voxel.npy'))
        sdf, sdf_grad = compute_sdf(vg, args.exp_name)
        for j in range(config_len):
            q = valid_set[j][1]
            z = valid_set[j][0]
            c = valid_set[j][2]
            positionList1, positionList2 = getCoordinate(q, c, pc, link_name_list, constraint, args.exp_name)
            minDis = getMinDis(q, c, pc, sdf, link_name_list, positionList1_new=positionList1, positionList2_new=positionList2)
            data_c.append(c)
            data_q.append(q)
            data_z.append(z)
            data_vg.append(vg)
            data_sdf.append(sdf)
            data_dis.append(minDis)
        for j in range(config_len):
            q = invalid_set[j][1]
            z = invalid_set[j][0]
            c = invalid_set[j][2]
            positionList1, positionList2 = getCoordinate(q, c, pc, link_name_list,constraint, args.exp_name)
            minDis = getMinDis(q, c, pc, sdf, link_name_list, positionList1_new=positionList1, positionList2_new=positionList2)
            data_c.append(c)
            data_q.append(q)
            data_z.append(z)
            data_vg.append(vg)
            data_sdf.append(sdf)
            data_dis.append(minDis)
    pickle.dump({'data_c': data_c, 'data_q': data_q, 'data_z': data_z, 'data_vg': data_vg, 'data_sdf': data_sdf, 'data_dis': data_dis}, open(f'dataset/{args.exp_name}/scene_data/config_distace.pkl', 'wb'))

def makeDisSet2():
    global data_c, data_q, data_z, data_vg, data_sdf, data_dis, pc, positionListSum1, positionListSum2, link_name_list, model_info
    for i in range(args.scene_start_idx, args.scene_start_idx + args.num_scenes):
        print('scene:', i )
        # qdataId = random.sample(range(0, len(qdata)), 200)
        # for j in qdataId:
        data_c_invalid = []
        data_q_invalid = []
        data_z_invalid = []
        data_vg_invalid = []
        data_sdf_invalid = []
        data_dis_invalid = []

        data_c_valid = []
        data_q_valid = []
        data_z_valid = []
        data_vg_valid = []
        data_sdf_valid = []
        data_dis_valid = []
        for j in range(len(qdata)):
        # for j in range(500):
            sdf = sdfList[i]
            vg = vgList[i]
            if model_info['c_dim'] > 0:
                q = qdata[j][3:]
                c = qdata[j][0:3]
            else:
                q = qdata[j]
                c = None
            # q = qdata[j][3:]
            # c = qdata[j][0:3]
            positionList1 = positionListSum1[j]
            positionList2 = positionListSum2[j]
            minDis = getMinDis(q, c, pc, sdf, link_name_list, args.exp_name, positionList1_new=positionList1,
                               positionList2_new=positionList2)
            if minDis<0:
                data_c_invalid.append(c)
                data_q_invalid.append(q)
                data_z_invalid.append(zList[j])
                data_vg_invalid.append(vg)
                data_sdf_invalid.append(sdf)
                data_dis_invalid.append(minDis)
            else:
                data_c_valid.append(c)
                data_q_valid.append(q)
                data_z_valid.append(zList[j])
                data_vg_valid.append(vg)
                data_sdf_valid.append(sdf)
                data_dis_valid.append(minDis)
        dataId = random.sample(range(0, len(data_c_invalid)), min(len(data_c_invalid), args.num_each_scenes))
        if len(data_c)==0:
            data_c = np.array(data_c_invalid)[dataId]
            data_q = np.array(data_q_invalid)[dataId]
            data_z = np.array(data_z_invalid)[dataId]
            data_vg = np.array(data_vg_invalid)[dataId]
            data_sdf = np.array(data_sdf_invalid)[dataId]
            data_dis = np.array(data_dis_invalid)[dataId]
        else:
            data_c = np.concatenate((data_c , np.array(data_c_invalid)[dataId]))
            data_q = np.concatenate((data_q , np.array(data_q_invalid)[dataId]))
            data_z = np.concatenate((data_z , np.array(data_z_invalid)[dataId]))
            data_vg = np.concatenate((data_vg , np.array(data_vg_invalid)[dataId]))
            data_sdf = np.concatenate((data_sdf , np.array(data_sdf_invalid)[dataId]))
            data_dis = np.concatenate((data_dis , np.array(data_dis_invalid)[dataId]))

        dataId = random.sample(range(0, len(data_c_valid)), 2 * args.num_each_scenes - min(len(data_c_invalid), args.num_each_scenes))
        data_c = np.concatenate((data_c , np.array(data_c_valid)[dataId]))
        data_q = np.concatenate((data_q , np.array(data_q_valid)[dataId]))
        data_z = np.concatenate((data_z , np.array(data_z_valid)[dataId]))
        data_vg = np.concatenate((data_vg , np.array(data_vg_valid)[dataId]))
        data_sdf = np.concatenate((data_sdf , np.array(data_sdf_valid)[dataId]))
        data_dis = np.concatenate((data_dis , np.array(data_dis_valid)[dataId]))

    pickle.dump({'data_c': data_c, 'data_q': data_q, 'data_z': data_z, 'data_vg': data_vg, 'data_sdf': data_sdf, 'data_dis': data_dis}, open(f'dataset/{args.exp_name}/scene_data/config_distace.pkl', 'wb'))
    pickle.dump({'data_c': data_c, 'data_q': data_q, 'data_z': data_z, 'data_vg': data_vg, 'data_dis': data_dis}, open(f'dataset/{args.exp_name}/scene_data/config_distace_vg.pkl', 'wb'))
    pickle.dump({'data_c': data_c, 'data_q': data_q, 'data_z': data_z, 'data_sdf': data_sdf, 'data_dis': data_dis}, open(f'dataset/{args.exp_name}/scene_data/config_distace_sdf.pkl', 'wb'))

def dataResave():
    data_c = []
    data_q = []
    data_z = []
    data_vg = []
    data_sdf = []
    data_dis = []
    joint_configs = pickle.load(open(f'dataset/{args.exp_name}/scene_data/config_distace2.pkl', 'rb'))
    # print(len(joint_configs))
    # data_dis = np.array(joint_configs['data_dis'])
    # data_dis2 = data_dis[np.where(data_dis < 0)]
    # print('data num / data<0:', len(data_dis), len(data_dis2))


    data_c_r = joint_configs['data_c']
    data_q_r = joint_configs['data_q']
    data_z_r = joint_configs['data_z']
    data_vg_r = joint_configs['data_vg']
    data_sdf_r = joint_configs['data_sdf']
    data_dis_r = joint_configs['data_dis']
    for i in range(len(data_c_r)):
        data_c.append(data_c_r[i])
        data_q.append(data_q_r[i])
        data_z.append(data_z_r[i])
        data_vg.append(data_vg_r[i])
        data_sdf.append(data_sdf_r[i])
        data_dis.append(data_dis_r[i])
        # data_sdf.append(np.around(data_sdf_r[i], decimals=4))

    # pickle.dump({'data_c': data_c, 'data_q': data_q, 'data_z': data_z, 'data_sdf': data_sdf,
    #              'data_dis': data_dis}, open(f'dataset/{args.exp_name}/scene_data/config_distace_sdf.pkl', 'wb'))
    pickle.dump({'data_c': data_c, 'data_q': data_q, 'data_z': data_z, 'data_vg': data_vg,
                 'data_dis': data_dis}, open(f'dataset/{args.exp_name}/scene_data/config_distace_vg.pkl', 'wb'))

def testDataSet(constraint, update_scene_from_yaml):
    # constraint, model_info, condition, update_scene_from_yaml, set_constraint, _ = generate_environment(
    #     args.exp_name)
    data = pickle.load(open(f'dataset/{args.exp_name}/scene_data/config_distace_sdf.pkl', 'rb'))
    data_c = data['data_c']
    data_q = data['data_q']
    data_z = data['data_z']
    data_sdf = data['data_sdf']
    # data_vg = data['data_vg']
    data_dis = data['data_dis']
    success_list = []
    for i in range(5000):
        # if i < 200:
        #     scene_id = 0
        scene_id = int(i / 200)
        print('i scene_id',i, scene_id)
        scene_dir = f'dataset/{args.exp_name}/scene_data'
        scene_dir_local = '{}/scene_{:04d}'.format(scene_dir, scene_id)
        scene_data = yaml.load(open(f'{scene_dir_local}/scene.yaml', 'r'), Loader=yaml.FullLoader)
        # constraint, model_info, condition, update_scene_from_yaml, set_constraint, _ = generate_environment(
        #     args.exp_name)
        update_scene_from_yaml(scene_data)
        # constraint.planning_scene.updata(data_q[i])
        constraint.planning_scene.display(data_q[i])
        true_valid = constraint.planning_scene.is_valid(data_q[i])
        scene_id_2 = int(i / 100)
        # if scene_id_2 % 2 ==0:
        #     pre_valid = False
        # else:
        #     pre_valid = True

        if data_dis[i] < 0:
            pre_valid = False
        else:
            pre_valid = True
        print('pre_valid, true_valid :', pre_valid, true_valid )
        if true_valid != pre_valid:
            success_list.append(0)
            # time.sleep(5)
        else:
            success_list.append(1)
    print('mean_success:',np.mean(success_list))

if __name__ == '__main__':
    constraint, model_info, condition, update_scene_from_yaml, set_constraint = reset()
    # makeDisSet2()
    # testDataSet(constraint, update_scene_from_yaml)
    # dataResave()
# sdf, sdf_grad = compute_sdf(vg)
# # plotSdf(sdf)
# # print(sdf)
# getCoordinate(q, c, pc, link_name_list)
# minDisList = []
# for i in range(100):
#     q = qdata[i][3:]
#     c = qdata[i][0:3]
#     minDis = getMinDis(q, c, pc, sdf, link_name_list)
#     minDisList.append(minDis)
# print(np.min(minDisList))