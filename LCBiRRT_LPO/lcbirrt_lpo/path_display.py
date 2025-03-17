
from ljcmp.utils.time_parameterization import time_parameterize
from ljcmp.utils.generate_environment import generate_environment
from lcbirrt_lpo.utils.voxel_utils import getObjPose
import time
import numpy as np
import yaml
exp_name='panda_orientation' # panda_dual_orientation panda_dual panda_orientation
constraint, model_info, condition, update_scene_from_yaml, set_constraint, _ = generate_environment(exp_name)
path_dir = 'result/pathdata/'
# 513, 514
scene_num =  513
scene_dir = f'dataset/{exp_name}/scene_data'
scene_dir_local = '{}/scene_{:04d}'.format(scene_dir, scene_num)
scene_data = yaml.load(open(f'{scene_dir_local}/scene.yaml', 'r'), Loader=yaml.FullLoader)
update_scene_from_yaml(scene_data)

path_dir = f'result/pathdata/{exp_name}/'
q_path=np.load(f'{path_dir}{scene_num}_q_path.npy')
q_path_raw=np.load(f'{path_dir}{scene_num}_q_path_raw.npy')
q_path_raw[-1]=q_path[-1]
hz = 20

duration, qs_sample, qds_sample, qdds_sample, ts_sample = time_parameterize(q_path, model_info, hz=hz)
pc = constraint.planning_scene
objPositionList=[]
for q in qs_sample:
    if exp_name == 'panda_orientation':
        pos, quat = constraint.forward_kinematics('panda_arm', q[:7])
        objPositionList.append(pos)
    else:
        objPose, obj_size = getObjPose(pc, condition, q, constraint)
        objPositionList.append(objPose[0:3, 3])

for i in range(len(objPositionList)):
    # print(i)
    quat = [0, 0, 0, 1]
    link_name = 'link' + str(i)
    pc.add_sphere(link_name, 0.015, objPositionList[i], quat)
pc.display(qs_sample[0])
time.sleep(1.0)

for q in qs_sample:
    slow_down = 3.0
    constraint.planning_scene.display(q)
    time.sleep(1.0/hz * slow_down)


for i in range(len(objPositionList)):
    link_name = 'link' + str(i)
    pc.remove_object(link_name)
# pc.display(qs_sample[-1])
time.sleep(1.0)
