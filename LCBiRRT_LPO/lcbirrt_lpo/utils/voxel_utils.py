from srmt.planning_scene import PlanningScene, VisualSimulator
from math import pi
from sdf_tools.utils_3d import compute_sdf_and_gradient
from math import cos, sin, pi
from scipy.spatial.transform import Rotation as R
from srmt.utils.transform_utils import get_transform, get_pose
from ljcmp.utils.generate_environment import generate_environment

import numpy as np
import time
import matplotlib.pyplot as plt
import os
import sys

import matplotlib.colors
import matplotlib.ticker
import matplotlib.pyplot as plt
import random

from sympy.abc import alpha


# sys.path.append('/home/zhang/.cache/JetBrains/PyCharmCE2024.2/python_stubs/2068908067')
# os.sys.path.append('/home/zhang/.cache/JetBrains/PyCharmCE2024.2/python_stubs/2068908067')

def positionInterp(position1, position2):
    positionList = [position1]
    distanceThreshold = 0.05
    if np.linalg.norm(position2 - position1)>0:
        direction = (position2 - position1) / np.linalg.norm(position2 - position1)
        distance = np.linalg.norm(position2 - position1)
        positionStep = position1
        while distance > distanceThreshold:
            positionStep = positionStep + distanceThreshold * direction
            positionList.append(positionStep)
            distance = np.linalg.norm(position2 - positionStep)
    return positionList

def getlinkPositionList(q,pc,link_name_list):

    for n in range(len(link_name_list)):
        linkPositionList = []
        linkPositionListSum = []
        for i in range(len(link_name_list[n])):
            link_name = link_name_list[n][i]
            pose = pc.get_link_pose(link_name)
            rot = np.array(pose)[0:3, 0:3]
            # print(pose)
            position = np.array(pose)[0:3, 3]
            # quat = [0,0,0,1]
            # pc.add_sphere(link_name, 0.06, position, quat)

            if i == 1:
                linkPositionList_i = []
                position_i = position + rot.dot(np.array([0, 0.08, 0]))
                linkPositionList_i.append(position_i)
                position_i = position + rot.dot(np.array([0, -0.08, 0]))
                linkPositionList_i.append(position_i)
                linkPositionListSum.append(linkPositionList_i)
            if i == 3:
                position = np.array(pose)[0:3, 3] + rot.dot(np.array([0, 0, -0.07]))

            if i == 4:
                linkPositionList_i = []
                position_i = position + rot.dot(np.array([0, 0, -0.08]))
                linkPositionList_i.append(position_i)
                position_i = position + rot.dot(np.array([0, 0, 0.08]))
                linkPositionList_i.append(position_i)
                linkPositionListSum.append(linkPositionList_i)
            if i == 5:
                position = np.array(pose)[0:3, 3] + rot.dot(np.array([0, 0, -0.2]))
            if i == 7:
                position = np.array(pose)[0:3, 3] + rot.dot(np.array([0, 0, -0.03]))
            # if i == 8:
            #     position = np.array(pose)[0:3, 3] + rot.dot(np.array([0, 0, 0.03]))
            if i == 9:
                position = np.array(pose)[0:3, 3] + rot.dot(np.array([0, 0, 0.02]))
                linkPositionList_i = []
                position_i = position + rot.dot(np.array([0, 0.065, 0]))
                linkPositionList_i.append(position_i)
                position_i = position + rot.dot(np.array([0, -0.09, 0]))
                linkPositionList_i.append(position_i)
                linkPositionListSum.append(linkPositionList_i)

            linkPositionList.append(position)
            if i == 4:
                linkPositionList.append(position + rot.dot(np.array([-0.085, 0.07, 0])))
            if i == 5:
                linkPositionList.append(position + rot.dot(np.array([0, 0.03, 0])))
                linkPositionList.append(np.array(pose)[0:3, 3] + rot.dot(np.array([0, 0.08, 0])))
                # linkPositionList1.append(np.array(pose)[0:3, 3] + rot.dot(np.array([0,-0.08,0])))
        if n==0:
            positionList = positionInterp(linkPositionList[0], linkPositionList[1])
            for i in range(1, len(linkPositionList) - 1):
                # print(i)
                positionListi = positionInterp(linkPositionList[i], linkPositionList[i + 1])
                positionList = np.concatenate((positionList, positionListi))
        else:
            for i in range(0, len(linkPositionList) - 1):
                # print(i)
                positionListi = positionInterp(linkPositionList[i], linkPositionList[i + 1])
                positionList = np.concatenate((positionList, positionListi))
        for i in range(len(linkPositionListSum)):
            # print(i)
            for j in range(len(linkPositionListSum[i]) - 1):
                positionListi = positionInterp(linkPositionListSum[i][j], linkPositionListSum[i][j + 1])
                positionList = np.concatenate((positionList, positionListi))

    for i in range(len(positionList)):
        # print(i)
        quat = [0, 0, 0, 1]
        link_name = 'link' + str(i)
        pc.add_sphere(link_name, 0.06, positionList[i], quat)
    pc.display(q)
    return positionList

def getObjPositionList(pc,c,q,constraint):
    objPose, obj_size = getObjPose(pc, c, q, constraint)
    objPosition = objPose[0:3, 3]
    objRot = objPose[0:3, 0:3]
    positionList = []
    xList = np.linspace(-obj_size[0]/2,obj_size[0]/2,10)
    yList = np.linspace(-obj_size[1]/2,obj_size[1]/2, 10)
    for i in range(10):
        for j in range(10):
            vector = np.array([xList[i],yList[j],0])
            position_ij = objRot.dot(vector) + objPosition
            positionList.append(position_ij)

    for i in range(len(positionList)):
        # print(i)
        quat = [0, 0, 0, 1]
        link_name = 'link_o_' + str(i)
        pc.add_sphere(link_name, 0.02, positionList[i], quat)
    pc.display(q)
    return positionList

def getObjPose(pc,c,q,constraint):

    d1, d2, theta = c
    l = d1 + 2 * d2 * cos(theta)
    ly = l * sin(theta)
    lz = l * cos(theta)

    dt = pi - 2 * theta
    chain_pos = np.array([0.0, ly, lz])
    chain_rot = np.array([[1, 0, 0], [0, cos(dt), -sin(dt)], [0, sin(dt), cos(dt)]])
    chain_quat = R.from_matrix(chain_rot).as_quat()

    t1 = np.concatenate([chain_pos, chain_quat])
    constraint.set_chains([t1])
    pc.detach_object('tray', 'panda_2_hand_tcp')

    constraint.set_early_stopping(True)

    l_obj_z = d2 + d1 / 2 * cos(theta)
    l_obj_y = d1 / 2 * sin(theta)
    ee_to_obj_pos = np.array([0.0, l_obj_y, l_obj_z])
    obj_dt = -(pi / 2 + theta)
    ee_to_obj_rot = np.array([[1, 0, 0], [0, cos(obj_dt), -sin(obj_dt)], [0, sin(obj_dt), cos(obj_dt)]])
    ee_to_obj_quat = R.from_matrix(ee_to_obj_rot).as_quat()

    # q = np.array([0, 0, 0, -pi / 2, 0, pi / 2, pi / 4, 0, 0, 0, -pi / 2, 0, pi / 2, pi / 4])
    pos, quat = constraint.forward_kinematics('panda_arm_2', q[:7])
    T_0g = get_transform(pos, quat)
    T_go = get_transform(ee_to_obj_pos, ee_to_obj_quat)
    T_0o = np.dot(T_0g, T_go)
    np.set_printoptions(precision=5, suppress=True)
    obj_pos, obj_quat = get_pose(T_0o)

    obj_size = [d1 * 3 / 4, d1, 0.01]
    pc.add_box('tray', [d1 * 3 / 4, d1, 0.01], obj_pos, obj_quat)
    pc.update_joints(q)
    pc.attach_object('tray', 'panda_2_hand_tcp', [])
    constraint.set_grasp_to_object_pose(go_pos=ee_to_obj_pos, go_quat=ee_to_obj_quat)
    return T_0o, obj_size

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

def plotVoxel(voxelValues):
    # fig = plt.figure()
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(voxelValues, edgecolor='k', shade=False)
    ax.grid(False)

    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.zlabel('Z')
    # plt.title('3D Voxel Map')

    plt.tight_layout()
    # plt.show()
    plt.savefig('scene1Voxel.jpg',dpi=300)

def plotSdf(sdf):

    v = sdf

    xyzminvalue = v.min()
    xyzmaxvalue = v.max()
    relativevalue = np.zeros((v.shape[0], v.shape[1], v.shape[2]))
    for i in range(0, relativevalue.shape[0]):
        for j in range(0, relativevalue.shape[1]):
            for k in range(0, relativevalue.shape[2]):
                relativevalue[i][j][k] = (v[i][j][k] - xyzminvalue) / (xyzmaxvalue - xyzminvalue)

    colorsvalues = np.empty(v.shape, dtype=object)
    alpha = 0.5
    mycolormap = plt.get_cmap("plasma")
    for i in range(0, v.shape[0]):
        for j in range(0, v.shape[1]):
            for k in range(0, v.shape[2]):
                tempc = mycolormap(relativevalue[i][j][k])
                colorreal = (tempc[0], tempc[1], tempc[2], alpha)
                colorsvalues[i][j][k] = colorreal

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')

    plt.set_cmap(plt.get_cmap("YlOrRd", 100))

    im = ax.voxels(v, facecolors = colorsvalues, edgecolor=None, shade=False, alpha=0.6)
    ax.grid(False)
    plt.tight_layout()
    plt.savefig('scene1SDF.jpg',dpi=300)
    plt.show()

def getCoordinate(q, c, pc, link_name_list,constraint,exp_name):
    positionList1 = getlinkPositionList(q, pc, link_name_list)
    if exp_name == 'panda_orientation':
        vg_low = np.array([-0.75, -0.75, 0])
        vg_high = np.array([0.75, 0.75, 1.5])
        resolution = 0.046875
    elif exp_name == 'panda_dual' or exp_name == 'panda_dual_orientation':
        vg_low = np.array([0, -0.5, 0.5])
        vg_high = np.array([1, 0.5, 1.5])
        resolution = 0.03125
    else:
        raise ValueError
    # vg_low = np.array([0, -0.5, 0.5])
    # vg_high = np.array([1, 0.5, 1.5])
    positionList1_new = []
    for i in range(len(positionList1)):
        # abandon = False
        positionList1_i = (positionList1[i] - vg_low)/resolution
        positionList1_i = [int(positionList1_i[j]) for j in range(len(positionList1_i))]
        positionList1_i=np.clip(positionList1_i, [0,0,0],[31,31,31])
        positionList1_new.append(positionList1_i)

    if len(link_name_list) >1 :
        positionList2 = getObjPositionList(pc, c, q, constraint)
        positionList2_new = []
        for i in range(len(positionList2)):
            # abandon = False
            positionList2_i = (positionList2[i] - vg_low) / resolution
            positionList2_i = [int(positionList2_i[j]) for j in range(len(positionList2_i))]
            positionList2_i = np.clip(positionList2_i, [0, 0, 0], [31, 31, 31])
            positionList2_new.append(positionList2_i)
    else:
        positionList2_new = None
    return positionList1_new, positionList2_new

def getMinDis(q, c, pc, sdf, link_name_list, exp_name, positionList1_new=None, positionList2_new=None):
    # vg_low = np.array([0, -0.5, 0.5])
    # vg_high = np.array([1, 0.5, 1.5])
    if exp_name == 'panda_orientation':
        vg_low = np.array([-0.75, -0.75, 0])
        vg_high = np.array([0.75, 0.75, 1.5])
        resolution = 0.046875
    elif exp_name == 'panda_dual' or exp_name == 'panda_dual_orientation':
        vg_low = np.array([0, -0.5, 0.5])
        vg_high = np.array([1, 0.5, 1.5])
        resolution = 0.03125
    else:
        raise ValueError
    if positionList1_new is None:
        positionList1 = getlinkPositionList(q, pc, link_name_list)
        positionList1_new = []
        for i in range(len(positionList1)):
            # abandon = False
            positionList1_i = (positionList1[i] - vg_low)/resolution
            positionList1_i = [int(positionList1_i[j]) for j in range(len(positionList1_i))]
            positionList1_i=np.clip(positionList1_i, [0,0,0],[31,31,31])
            positionList1_new.append(positionList1_i)

    if positionList2_new is None and len(link_name_list)>1:
        positionList2 = getObjPositionList(pc, c, q, constraint)
        positionList2_new = []
        for i in range(len(positionList2)):
            # abandon = False
            positionList2_i = (positionList2[i] - vg_low) / resolution
            positionList2_i = [int(positionList2_i[j]) for j in range(len(positionList2_i))]
            positionList2_i = np.clip(positionList2_i, [0, 0, 0], [31, 31, 31])
            positionList2_new.append(positionList2_i)

    # sdf, sdf_grad = compute_sdf(vg)
    distanceList = []
    if positionList1_new is not None:
        for i in range(len(positionList1_new)):
            distance_i = sdf[positionList1_new[i][0]][positionList1_new[i][1]][positionList1_new[i][2]]-0.03
            distanceList.append(distance_i)
    if positionList2_new is not None:
        for i in range(len(positionList2_new)):
            distance_i = sdf[positionList2_new[i][0]][positionList2_new[i][1]][positionList2_new[i][2]]-0.01
            distanceList.append(distance_i)
    distanceMin = np.min(distanceList)

    # fig = plt.figure()
    # positionList1_new = np.array(positionList1_new)
    # positionList2_new = np.array(positionList2_new)
    # ax = fig.add_subplot(111, projection='3d')
    # # im = ax.scatter(positionList2_new[:,0], positionList2_new[:,1], positionList2_new[:,2], s=100, c='r', marker='.')
    # im = ax.scatter(positionList1_new[:,0], positionList1_new[:,1], positionList1_new[:,2], s=100, c='r', marker='.')
    #
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # plt.show()

    return distanceMin


if __name__ == '__main__':
    current_file_path = os.path.abspath(__file__)
    exp_name = 'panda_dual_orientation' #'panda_dual'
    # current_file_path = os.path.abspath(__file__).rpartition('/')[0]
    # sys.path.append(current_file_path)
    # dataPath = os.path.join(current_file_path, 'dataset/panda_dual_old/manifold/data_10000.npy')
    # vgPath = os.path.join(current_file_path,  'dataset/panda_dual/scene_data')
    # dataPath = f'dataset/{exp_name}_old/manifold/data_10000.npy'
    dataPath = f'dataset/{exp_name}_old/manifold/data_fixed_10000.npy'
    vgPath = f'dataset/{exp_name}/scene_data'
    qdata = np.load(dataPath, allow_pickle=True)
    i = 0
    scene_dir_local = '{}/scene_{:04d}'.format(vgPath, i)
    vg = np.load(os.path.join(scene_dir_local, 'voxel.npy'))
    constraint, model_info, condition, update_scene_from_yaml, set_constraint, _ = generate_environment(exp_name)

    pc = PlanningScene(arm_names=['panda_arm_2', 'panda_arm_1'], arm_dofs=[7, 7])

    # q = np.array([-0.2, -0.4, 0, -pi / 2, 0, pi / 2, -pi, -0.2, 0.3, 0, -pi / 2, 0, pi / 2, -pi])
    q = qdata[2][3:]
    c = qdata[2][0:3]

    link_name_list = [
        ['panda_1_link0', 'panda_1_link1', 'panda_1_link2', 'panda_1_link3', 'panda_1_link4', 'panda_1_link5',
         'panda_1_link6', 'panda_1_link7', 'panda_1_link8', 'panda_1_hand'],
        ['panda_2_link0', 'panda_2_link1', 'panda_2_link2', 'panda_2_link3', 'panda_2_link4', 'panda_2_link5',
         'panda_2_link6', 'panda_2_link7', 'panda_2_link8', 'panda_2_hand']]

    # pc = PlanningScene(arm_names=model_info['arm_names'], arm_dofs=model_info['arm_dofs'],
    #                    base_link=model_info['base_link'])
    # link_name_list = [
    #     ['panda_link0', 'panda_link1', 'panda_link2', 'panda_link3', 'panda_link4', 'panda_link5',
    #      'panda_link6', 'panda_link7', 'panda_link8', 'panda_hand']]
    # q = qdata[2]
    # c=None


    sdf, sdf_grad = compute_sdf(vg, exp_name)
    # plotSdf(sdf)
    # print(sdf)
    minDisList = []
    for i in range(0,1):
        print(i)
        q = qdata[i][3:]
        c = qdata[i][0:3]
        # q = qdata[i]
        # c = None
        minDis = getMinDis(q, c, pc, sdf, link_name_list, exp_name)
        minDisList.append(minDis)
        time.sleep(0.4)
    print(np.min(minDisList))


