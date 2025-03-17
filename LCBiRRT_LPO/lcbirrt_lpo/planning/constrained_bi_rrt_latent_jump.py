
import time
from copy import deepcopy

import numpy as np
import torch

from srmt.constraints.constraints import ConstraintBase

from ljcmp.models.latent_model import LatentModel, LatentValidityModel
from ljcmp.models.sampler import Sampler
from ljcmp.planning.motion_trees import LatentMotionTree
from ljcmp.planning.distance_functions import distance_z, distance_q
from ljcmp.planning.status_description import GrowStatus

from scipy.stats import norm
from scipy.special import ndtri
import copy
import random

class ConstrainedLatentBiRRT():
    def __init__(self, model : LatentModel, validity_model : LatentValidityModel=None,
                 distance_model: LatentValidityModel = None,
                 constraint : ConstraintBase=None, samplerModel : Sampler = None,
                 latent_dim=2, state_dim=3, validity_fn=None, latent_jump=True,
                 start_region_fn=None, goal_region_fn=None,
                 path_check_method=None, whole_path_check_freq=None) -> None:
        self.learned_model = model
        self.validity_model = validity_model
        self.distance_model = distance_model
        self.path_check_method = path_check_method
        self.whole_path_check_freq = whole_path_check_freq
        self.samplerModel = samplerModel
        self.useSamplerModel = False
        self.constraint=constraint
        self.start_tree = LatentMotionTree(name='start_tree', model=model, multiple_roots=True if start_region_fn is not None else False)
        self.goal_tree = LatentMotionTree(name='goal_tree', model=model, multiple_roots=True if goal_region_fn is not None else False)
        self.start_q = None
        self.goal_q = None
        self.start_list = []
        self.goal_list = []
        self.sampled_goals = []
        self.next_goal_index = 0
        self.distance_btw_trees = float('inf')
        self.distance_btw_trees_q = float('inf')
        # self.use_latent_jump = latent_jump
        self.use_latent_jump =False
        self.max_latent_jump_trials = 1
        self.solved = False
        self.off_manifold_threshold = 1.0
        self.p_q_plan = 0.001
        self.validity_fn = validity_fn
        if validity_fn is None:
            self.planning_scene = constraint.planning_scene

        self.latent_dim = latent_dim
        self.state_dim = state_dim
        
        self.ub = constraint.ub
        self.lb = constraint.lb

        self.p_sample = 0.02
        self.region_timeout = 0.1

        self.start_region_fn = start_region_fn
        self.goal_region_fn = goal_region_fn
        
        # properties
        self.max_distance = 0.1 
        self.max_distance_q = 0.5
        self.solved_time = -1.0

        self.delta = 0.05
        self.lambda1 = 5 # curvature length ratio

        self.debug = False

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

    def print(self, *args):
        if self.debug:
            with np.printoptions(precision=3, suppress=True, linewidth=200):
                print(*args)

    def node_dis(self, z_in, max_steps=10):
        qlist = []
        q0 = self.learned_model.to_state(z_in)
        self.constraint.project(q0)
        qlist.append(q0)
        z = torch.from_numpy(np.array(z_in)).float().to(self.device)
        z.requires_grad_()
        z.retain_grad()
        # z = Variable(D0[0:10]
        # , requires_grad=True)
        v = self.distance_model.sdf
        if self.distance_model.c_dim>0:
            c = self.distance_model.given_c.unsqueeze(dim=0)
        else:
            c = None
        for k in range(max_steps):
            output, voxel_latent = self.distance_model(z.unsqueeze(dim=0), voxel=v, c=c)
            # output, voxel_latent = self.model(D0[0], voxel=V0[0], c=C0[0])
            output[0].sum().backward()
            # print('distance:', output)
            # print('grad:', z.grad)
            # z = z + 1.0 * z.grad
            z = z + 0.8 * z.grad
            z.requires_grad_()
            z.retain_grad()
            # print('z:', z)
            if self.distance_model.c_dim > 0:
                self.learned_model.set_condition(c.squeeze(dim=0).cpu().data.numpy())
            q = self.learned_model.to_state(z.cpu().data.numpy())
            r1 = self.constraint.project(q)
            r2 = self.is_valid(q=q, approx=False)
            # r3 = self.planning_scene.is_valid(q)
            qlist.append(q)
            # print('r1,r2,r3:',r1,r2,r3)
            # if r is True and self.is_valid(z.cpu().data.numpy(), q=q, approx=False) is True:
            if not r1:
                break
            if r1 and r2:
                # for i in range(len(qlist)):
                #     self.constraint.planning_scene.display(qlist[i])
                #     time.sleep(0.5)
                return z.cpu().data.numpy() , q
                # break
            # print('q:', q)
            # qlist.append(q)
        if k==max_steps-1:
            # print('node dis timeout')
            # for i in range(len(qlist)):
            #     self.constraint.planning_scene.display(qlist[i])
            #     time.sleep(0.5)
            return None, qlist
        return None, None

    def interpolate_path(self, path_q):
        path_q_new = []
        success = True
        for i in range(len(path_q)-1):
            try_num = 0
            start = path_q[i]
            end = path_q[i+1]
            path_q_new.append(start)
            dist_q = distance_q(start, end)

            while dist_q > self.delta:
                scratch, r = self.interpolate_q(start, end, self.delta / dist_q)
                if r and self.is_valid(q=scratch, approx=False):
                    path_q_new.append(scratch)
                else:
                    success = False
                    print('interpolate path faile!')
                    break
                dist_q = distance_q(scratch, end)
                start = scratch
                try_num = try_num +1
                if try_num > 30:
                    break
        path_q_new.append(path_q[-1])
        return path_q_new, success

    def path_dis(self, q_start, q_end):
        debug = True
        success = True
        q_local = []
        dist_q = distance_q(q_start, q_end)
        num_node_int = int(dist_q / self.delta)

        q_ints = [q_start]
        z_ints = [self.learned_model.to_latent(q_start)]
        for j in range(num_node_int):
            q_int, r = self.interpolate_q(q_start, q_end, (j + 1) / (num_node_int + 1))
            if r is False:
                if debug:
                    print('interpolate_q failure-2')
                success = False
                return success, []
            q_ints.append(q_int)
            z_ints.append(self.learned_model.to_latent(q_int))

        z_ints_new = [z_ints[0]]
        q_ints_new = [q_ints[0]]
        for j in range(1, len(z_ints)):
            z_int_new, q_int_new = self.node_dis(z_ints[j])
            if z_int_new is not None:

                # if self.debug:
                if debug:
                    print('node moved successfully-2')
                z_ints_new.append(z_int_new)
                q_ints_new.append(q_int_new)
            else:
                # if self.debug:
                if debug:
                    print('node moved failure-2')
                success = False
                return success, []

        z_ints_new.append(self.learned_model.to_latent(q_end))
        q_ints_new.append(q_end)
        for j in range(len(z_ints_new) - 1):
            aborted, q_ints = self.check_validity(q_ints_new[j], q_ints_new[j + 1])
            # gs1, geodesic_states = self.check_motion(q_ints_new[j], q_ints_new[j+1], end_checked=True)
            # if gs1 is GrowStatus.REACHED:
            if not aborted:
                if len(q_ints) > 0:
                    # last_node = nodes[i]
                    # z_ints = self.learned_model.to_latent(q_ints)
                    # for q_int, z_int in zip(q_ints, z_ints):
                    #     last_node = tree.add_node(last_node, q=q_int, z=z_int, valid=True, checked=True,
                    #                               projected=True, ref='i')
                    for q_int in q_ints:
                        q_local.append(q_int)

                if j != len(z_ints_new)-2:
                    # path_q.append(q_ints_new[j])
                    q_local.append(q_ints_new[j])
            else:
                if debug:
                    print('check motion failure-2')
                success = False
                return success, []

        if debug:
            print('------------------path check success!-------------------')
        return success, q_local

    def check_validity(self, q_start, q_end):
        dist_q = distance_q(q_start, q_end)
        q_ints = []
        aborted = False
        if dist_q > self.max_distance_q:
            num_node_int = int(dist_q / self.max_distance_q)
            for j in range(num_node_int):
                q_int, r = self.interpolate_q(q_start, q_end, (j + 1) / (num_node_int + 1))
                if r is False or self.is_valid(q=q_int, approx=False) is False:
                    # self.print('invalid path at', i, 'due to interpolation')
                    aborted = True
                    break
                gs, _ = self.check_motion(q_start, q_int, end_checked=True)
                # if gs is GrowStatus.ADVANCED:
                #     print('advance')
                if gs is GrowStatus.TRAPPED:
                # if gs is not GrowStatus.REACHED:
                    # self.print('check_motion failed invalid path at', i, 'due to collision')
                    aborted = True
                    break
                q_ints.append(q_int)
            # if aborted is True:
            #     break
            # last_node = nodes[i - 1]
            # z_ints = self.learned_model.to_latent(q_ints)
            # for q_int, z_int in zip(q_ints, z_ints):
            #     last_node = tree.add_node(last_node, q=q_int, z=z_int, valid=True, checked=True,
            #                               projected=True, ref='i')
            # header_changes.append([node, last_node])
        else:
            gs, _ = self.check_motion(q_start, q_end, end_checked=True)
            if gs is GrowStatus.TRAPPED:
            # if gs is not GrowStatus.REACHED:
                # self.print('check_motion failed invalid path at', i, 'due to collision')
                aborted = True
                # break
        return aborted, q_ints

    def check_tree_validity(self, tree: LatentMotionTree, path_z, nodes, z_other_node):
        path_q = self.learned_model.to_state(path_z)
        header_changes = []  # [child, new parent node]
        aborted = False
        debug = False

        for i in range(len(path_z)):
            node = nodes[i]
            if node.checked is True:
                if node.valid is True:
                    path_q[i] = node.q
                    continue

            else:  # if not checked
                manifold_distances = self.constraint.function(path_q[i])
                manifold_distance = np.linalg.norm(manifold_distances)
                if manifold_distance > self.off_manifold_threshold:
                    aborted = True
                    break

                r = self.constraint.project(path_q[i])
                if r is False:
                    aborted = True
                    break

                node.q = path_q[i]
                node.projected = True
                if self.is_valid(path_z[i], q=path_q[i], approx=False) is True:
                    node.q = path_q[i]  # updated q (after projection)
                    dist_q = distance_q(nodes[i - 1].q, node.q)

                    if dist_q > self.max_distance_q:
                        num_node_int = int(dist_q / self.max_distance_q)

                        q_ints = []
                        for j in range(num_node_int):
                            q_int, r = self.interpolate_q(nodes[i - 1].q, node.q, (j + 1) / (num_node_int + 1))
                            if r is False or self.is_valid(q=q_int, approx=False) is False:
                                self.print('invalid path at', i, 'due to interpolation')
                                aborted = True
                                break
                            gs, _ = self.check_motion(nodes[i - 1].q, q_int, end_checked=True)
                            # if gs is GrowStatus.ADVANCED:
                            #     print('advance')
                            if gs is GrowStatus.TRAPPED:
                                self.print('check_motion failed invalid path at', i, 'due to collision')
                                aborted = True
                                break
                            q_ints.append(q_int)
                        if aborted is True:
                            break
                        last_node = nodes[i - 1]
                        z_ints = self.learned_model.to_latent(q_ints)
                        for q_int, z_int in zip(q_ints, z_ints):
                            last_node = tree.add_node(last_node, q=q_int, z=z_int, valid=True, checked=True,
                                                      projected=True, ref='i')
                        header_changes.append([node, last_node])
                    else:
                        gs, _ = self.check_motion(nodes[i - 1].q, node.q, end_checked=True)
                        if gs is GrowStatus.TRAPPED:
                            self.print('check_motion failed invalid path at', i, 'due to collision')
                            aborted = True
                            break

                    node.checked = True
                    node.valid = True
                    tree.q_nodes.append(node)
                else:  # if not valid
                    if self.distance_model is not None and self.path_check_method != 'whole':
                        z_new, q_new = self.node_dis(path_z[i])
                        if z_new is not None:
                            if debug:
                                print('node moved successfully-1')
                            node.q = q_new
                            dist_q = distance_q(nodes[i - 1].q, node.q)

                            if dist_q > self.max_distance_q:
                                num_node_int = int(dist_q / self.max_distance_q)

                                q_ints = []
                                for j in range(num_node_int):
                                    q_int, r = self.interpolate_q(nodes[i - 1].q, node.q, (j + 1) / (num_node_int + 1))
                                    if r is False or self.is_valid(q=q_int, approx=False) is False:
                                        self.print('invalid path at', i, 'due to interpolation')
                                        aborted = True
                                        break
                                    gs, _ = self.check_motion(nodes[i - 1].q, q_int, end_checked=True)
                                    # if gs is GrowStatus.ADVANCED:
                                    #     print('advance')
                                    if gs is GrowStatus.TRAPPED:
                                        self.print('check_motion failed invalid path at', i, 'due to collision')
                                        aborted = True
                                        break
                                    q_ints.append(q_int)
                                if aborted is True:
                                    break
                                last_node = nodes[i - 1]
                                z_ints = self.learned_model.to_latent(q_ints)
                                for q_int, z_int in zip(q_ints, z_ints):
                                    last_node = tree.add_node(last_node, q=q_int, z=z_int, valid=True, checked=True,
                                                              projected=True, ref='i')
                                header_changes.append([node, last_node])
                            else:
                                gs, _ = self.check_motion(nodes[i - 1].q, node.q, end_checked=True)
                                if gs is GrowStatus.TRAPPED:
                                    self.print('check_motion failed invalid path at', i, 'due to collision')
                                    aborted = True
                                    break

                            node.checked = True
                            node.valid = True
                            tree.q_nodes.append(node)
                        else:
                            aborted = True
                            break
                    else:
                        aborted = True
                        break

                    # print(f'invalid path at {i}')
                    # aborted = True
                    # break

        for child, new_parent in header_changes:
            child.parent = new_parent

        if aborted:
            tree.delete_node(node)
            return False, nodes[i - 1]

        # nodes_new = z_other_node.path[1:]
        # for i in range(len(nodes_new) - 1):
        #     gs1, geodesic_states = self.check_motion(nodes_new[i].q, nodes_new[i + 1].q, end_checked=True)
        #     if gs1 is not GrowStatus.REACHED:
        #         print('------------------check error-----------------')
        #         # raise ValueError

        return True, nodes[-1]


    def check_tree_validity_dis2(self, tree: LatentMotionTree, path_z, nodes, z_other_node, try_max=15):
        path_q = self.learned_model.to_state(path_z)
        path_q_raw = deepcopy(path_q)
        header_changes = []  # [child, new parent node]
        aborted = False
        debug = False
        node_q_temp = []
        q_local = []

        for i in range(len(path_z)):
            node = nodes[i]
            if node.checked is True and node.valid is True:
                path_q[i] = node.q
                node_q_temp.append(path_q[i])
                continue
            # if node.checked is True:
            #     if node.valid is True:
            #         path_q[i] = node.q
            #         node_q_temp.append(path_q[i])
            #         continue
            #
            else:  # if not checked
                manifold_distances = self.constraint.function(path_q[i])
                manifold_distance = np.linalg.norm(manifold_distances)
                if manifold_distance > self.off_manifold_threshold:
                    aborted = True
                    break

                r = self.constraint.project(path_q[i])
                if r is False:
                    aborted = True
                    break

                node.q = path_q[i]
                node.projected = True
                if self.is_valid(path_z[i], q=path_q[i], approx=False) is True:
                    node_q_temp.append(path_q[i])
                    node.q = path_q[i]  # updated q (after projection)
                else:  # if not valid
                    # print(f'invalid path at {i}')
                    # tree.delete_node(node)
                    # aborted = True
                    # break
                    z_new, q_new = self.node_dis(path_z[i])
                    if z_new is not None:
                        # if self.debug:
                        if debug:
                            print('node moved successfully-11')
                        node_q_temp.append(q_new)
                    else:
                        # if node in tree.nodes:
                        #     tree.delete_node(node)
                        if debug:
                            print('node moved failure-11')
                        aborted = True
                        break

        if aborted:
            if debug:
                print('node aborted-11')
            if node in tree.nodes:
                tree.delete_node(node)
            return False, nodes[i - 1], None, None
        else:
            path_q = []
            success = True
            for i in range(len(nodes) - 1):
                # print(i)
                # if nodes[i].checked is True:
                #     if nodes[i].valid is True:
                #         path_q.append(nodes[i].q)
                #         continue
                # else:
                if len(q_local)>0:
                    last_node = nodes[i - 1]
                    for q in q_local:
                        z = self.learned_model.to_latent(q)
                        last_node = tree.add_node(last_node, q=q, z=z, valid=True, checked=True, projected=True,
                                                  ref='i')
                        # tree.q_nodes .append(last_node)
                    nodes[i].parent = last_node

                q_local = []
                nodes[i].q = node_q_temp[i]
                nodes[i].checked = True
                nodes[i].valid = True
                tree.q_nodes.append(nodes[i])

                # q_start = nodes[i].q
                # q_end = nodes[i+1].q
                q_start = node_q_temp[i]
                q_end = node_q_temp[i+1]

                path_q.append(q_start)
                # check_list = [[q_start, q_end]]
                dist_q = distance_q(q_start, q_end)
                if dist_q < self.delta:
                    if debug:
                        print('continue21')
                    continue
                gs1, geodesic_states = self.check_motion(q_start, q_end, end_checked=True)
                if gs1 is GrowStatus.REACHED:
                    if debug:
                        print('continue22')
                    continue
                else:

                    num_node_int = int(dist_q / self.delta)

                    q_ints = [q_start]
                    z_ints = [self.learned_model.to_latent(q_start)]
                    for j in range(num_node_int):
                        q_int, r = self.interpolate_q(q_start, q_end, (j + 1) / (num_node_int + 1))
                        if r is False:
                            if debug:
                                print('interpolate_q failure-2')
                            success = False
                            break
                        q_ints.append(q_int)
                        z_ints.append(self.learned_model.to_latent(q_int))
                    if not success:
                        break
                    z_ints_new = [z_ints[0]]
                    q_ints_new = [q_ints[0]]
                    for j in range(1,len(z_ints)):
                        z_int_new, q_int_new = self.node_dis(z_ints[j])
                        if z_int_new is not None:

                            # if self.debug:
                            if debug:
                                print('node moved successfully-2')
                            z_ints_new.append(z_int_new)
                            q_ints_new.append(q_int_new)
                        else:
                            # if self.debug:
                            if debug:
                                print('node moved failure-2')
                            success = False
                            break
                    if not success:
                        break

                    z_ints_new.append(self.learned_model.to_latent(q_end))
                    q_ints_new.append(q_end)
                    for j in range(len(z_ints_new)-1):
                        # aborted, q_ints = self.check_validity(q_ints_new[j], q_ints_new[j+1])
                        gs1, geodesic_states = self.check_motion(q_ints_new[j], q_ints_new[j+1], end_checked=True)
                        if gs1 is GrowStatus.REACHED:
                            if j != len(z_ints_new)-2:
                                path_q.append(q_ints_new[j])
                                q_local.append(q_ints_new[j])
                        else:
                            if debug:
                                print('check motion failure-2')
                            success = False
                            break
                    if not success:
                        break
                    else:
                        if debug:
                            print('------------------path check success!-------------------')

        # path_q_raw.append(node_q_temp[-1])
        path_q_raw=np.concatenate((path_q_raw,[node_q_temp[-1]]))
        if success:
            if len(q_local) > 0:
                last_node = nodes[-2]
                for q in q_local:
                    z = self.learned_model.to_latent(q)
                    last_node = tree.add_node(last_node, q=q, z=z, valid=True, checked=True, projected=True,
                                              ref='i')
                    # tree.q_nodes.append(last_node)
                nodes[-1].parent = last_node

            path_q.append(node_q_temp[-1])

            # nodes_new = z_other_node.path[1:]
            # for i in range(len(nodes_new) - 1):
            #     gs1, geodesic_states = self.check_motion(nodes_new[i].q, nodes_new[i + 1].q, end_checked=True)
            #     if gs1 is not GrowStatus.REACHED:
            #         print('------------------check error-----------------')
            #         # raise ValueError
            # print('return true')
            return True, nodes[-1], path_q, path_q_raw
        else:
            tree.delete_node(nodes[i+1])
            return False, nodes[i], None, path_q_raw



    def get_path_q(self,nodes,try_max=30):
        path_q = []
        success = True
        for i in range(len(nodes) - 1):

            q_start = nodes[i].q
            q_end = nodes[i + 1].q
            path_q.append(q_start)
            check_list = [[q_start, q_end]]
            dist_q = distance_q(q_start, q_end)
            if dist_q < self.delta:
                continue
            gs1, geodesic_states = self.check_motion(q_start, q_end, end_checked=True)
            if gs1 is GrowStatus.REACHED:
                continue
            else:
                q_mid = (q_start + q_end) / 2
                z_mid = self.learned_model.to_latent(q_mid)
                if self.is_valid(q=q_mid, approx=False) is False:
                    z_new, q_new = self.node_dis(z_mid)
                    if z_new is not None:
                        if self.debug:
                            print('node moved successfully-2')
                        q_mid = q_new
                    else:
                        if self.debug:
                            print('node moved failure-2')
                        success = False
                        break
                check_list = [[q_start, q_mid], [q_mid, q_end]]

            # q_start = nodes[i].q
            # q_end = nodes[i + 1].q
            # check_list = [[q_start, q_end]]
            # path_q.append(q_start)
                try_n = 0
                # for q_pair in check_list:
                check = False
                # for i in range(len(check_list)):
                #     if check:
                #         i = i - 1
                while len(check_list) > 0:
                    q_pair = check_list[0]
                    # q_pair = check_list[i]
                    start = q_pair[0]
                    end = q_pair[1]
                    dist_q = distance_q(start, end)
                    if dist_q < self.delta:
                        check_list = check_list[1:]
                        if len(check_list)>0:
                            path_q.append(end)
                        continue
                    gs1, geodesic_states = self.check_motion(start, end, end_checked=True)
                    try_n = try_n + 1
                    if gs1 is GrowStatus.REACHED:
                        check = False
                        check_list = check_list[1:]
                        if len(check_list)>0:
                            path_q.append(end)
                    else:
                        check = True
                        q_mid = (start + end) / 2
                        z_mid = self.learned_model.to_latent(q_mid)
                        if self.is_valid(q=q_mid, approx=False) is False:
                            z_new, q_new = self.node_dis(z_mid)
                            if z_new is not None:
                                if self.debug:
                                    print('node moved successfully-3')
                                q_mid = q_new
                            else:
                                if self.debug:
                                    print('node moved failure-3')
                                success = False
                                break
                        # check_list = np.concatenate(([[start, q_mid], [q_mid, end]], check_list[1:]))
                        if len(check_list) > 1:
                            check_list = np.concatenate(([[start, q_mid], [q_mid, end]], check_list[1:]))
                        else:
                            check_list = [[start, q_mid], [q_mid, end]]
                    i = 0
                    if try_n > try_max:
                        success = False
                        if self.debug:
                            print('timeout-3')
                        break
            if not success:
                break
        # if success:
        #     return path_q
        # else:
        #     return None
        # assert success, "can not get path_q."
        path_q.append(nodes[-1].q)
        # for i in range(len(path_q)):
        #     if self.is_valid(q=path_q[i], approx=False) is False:
        #         print('----check error----')
        #         success = False
        assert success, "can not get path_q."
        return path_q


    def is_valid(self,z=None,q = None, approx=True):
        """if q is given, then, q should be projected
        """
        if approx:
            if self.validity_model is None:
                return True
            return self.validity_model.is_valid_estimated(z[None,:])[0]
            
        if q is None:
            q = self.learned_model.to_state(z)
            r = self.constraint.project(q)
            if r is False: 
                return False
        
        if (q <self.lb).any() or (q > self.ub).any():
            return False
        
        if self.validity_fn is not None:
            return self.validity_fn(q)

        return self.planning_scene.is_valid(q)

    def check_motion(self, start, end, end_checked=False):
        """check motion from start to end

        Args:
            start (numpy.array): start node
            end (numpy.array): end node

        Returns:
            GrowStatus: grow status
            list: geodesic states
        """
        if end_checked is False:
            if self.is_valid(q=end,approx=False) is False:
                return GrowStatus.TRAPPED, None
            if self.constraint.is_satisfied(end) is False:
                return GrowStatus.TRAPPED, None
        
        grow_status, geodesic_states, step_dists, distance_traveled = self.discrete_geodesic(start, end, True)

        return grow_status, geodesic_states
    
    
    def discrete_geodesic(self, start, end, validity_check_in_interpolation = True):
        """discrete geodesic from start to end

        Args:
            start (numpy.array): start node
            end (numpy.array): end node
            validity_check_in_interpolation (bool, optional): whether to check validity in interpolation. Defaults to True.

        Returns:
            GrowStatus: grow status
            list: geodesic states
            list: step distances
            float: total distance
        """
        tolerance = self.delta
        geodesic = [start]

        dist = distance_q(start, end)
        if dist <= tolerance:
            geodesic.append(end)
            step_dists = [dist]
            total_dist = dist
            return GrowStatus.REACHED, geodesic, step_dists, total_dist

        max = dist * self.lambda1

        previous = start
        step_dists = []
        total_dist = 0
        status = GrowStatus.TRAPPED
        while True:
            scratch, r = self.interpolate_q(previous, end, self.delta / dist)
            # r = self.constraint.project(scratch)

            if r is False: # project failed
                break

            if validity_check_in_interpolation and not self.is_valid(q=scratch, approx=False): # not valid (check only if not interpolating)
                break
            
            new_dist = distance_q(scratch, end)
            if new_dist >= dist: # went to backward
                break

            step_dist = distance_q(previous, scratch)
            if step_dist > self.lambda1 * self.delta: # too much deviation
                break
                
            total_dist += step_dist
            if total_dist > max:
                break

            step_dists.append(step_dist)

            dist = new_dist
            previous = scratch
            geodesic.append(copy.deepcopy(scratch))
            status = GrowStatus.ADVANCED
            
            if dist <= tolerance:
                return GrowStatus.REACHED, geodesic, step_dists, total_dist

        return status, geodesic, step_dists, total_dist

    def set_start(self, q):
        q = q.astype(np.double)
        self.start_q = q
        z = self.learned_model.to_latent(q)
        
        if self.start_tree.multiple_roots:
            start_node = self.start_tree.add_root_node(z, q)
            self.start_list.append(start_node)
        else:
            self.start_tree.set_root(z, q)

    def set_goal(self, q):
        q = q.astype(np.double)
        self.goal_q = q
        z = self.learned_model.to_latent(q)
        
        if self.goal_tree.multiple_roots:
            goal_node = self.goal_tree.add_root_node(z, q)
            self.goal_list.append(goal_node)
        else:
            self.goal_tree.set_root(z, q) 

    def solve(self, max_time=10.0):
        self.start_time = time.time()
        is_start_tree = True
        self.print(self.distance_btw_trees)
        self.print(self.distance_btw_trees_q)
        self.print ('[z] Estimated distance to go: {0}'.format(self.distance_btw_trees))
        self.print ('[q] Estimated distance to go: {0}'.format(self.distance_btw_trees_q))
        self.terminate = False

        total_check_time = 0
        check_times = -1
        # frequency = 30
        z_current = None
        z_target = None
        currentNode = None
        # self.useSamplerModel = False

        if self.start_q is None:
            if self.start_tree.multiple_roots:
                start_q = self.start_region_fn()
                # print('start_q:', start_q)
                self.set_start(start_q)
            else:
                raise ValueError('start_q is None, but start_tree.multiple_roots is False') 

        if self.goal_q is None:
            if self.goal_tree.multiple_roots:
                goal_q = self.goal_region_fn()
                self.set_goal(goal_q)
            else:
                raise ValueError('goal_q is None, but goal_tree.multiple_roots is False')

        while self.terminate is False: 
            if (time.time() - self.start_time) > max_time:
                self.print('timed out')
                self.terminate = True
                break

            if self.start_region_fn is not None:
                if self.start_tree.multiple_roots:
                    if random.random() < self.p_sample:
                        start_q = self.start_region_fn(timeout=self.region_timeout)
                        # if self.start_q is not None:
                        if start_q is not None:
                            # print('start_q:', start_q)
                            self.set_start(start_q)

                            self.print('added start', start_q)
            
            if self.goal_region_fn is not None:
                if self.goal_tree.multiple_roots:
                    if random.random() < self.p_sample:
                        goal_q = self.goal_region_fn(timeout=self.region_timeout)
                        if goal_q is not None:
                            self.set_goal(goal_q)
                            self.print('added goal', goal_q)


            if is_start_tree:
                cur_tree = self.start_tree
                other_tree = self.goal_tree
            else:
                cur_tree = self.goal_tree
                other_tree = self.start_tree

            # currentNode = None
            if z_current is None or z_target is None:
                self.useSamplerModel = False
                z_rand = self.random_sample()
                r, z_des, z_node_cur = self.grow(cur_tree, z_rand)
            else:
                if self.samplerModel is not None:

                    self.useSamplerModel = True
                    z_in = np.concatenate((z_current, z_target))
                    z_rand = self.random_sample(z_in)
                    z_rand = z_rand + np.random.normal(np.zeros(len(z_rand)), 0.05 * np.ones(len(z_rand)))
                    r, z_des, z_node_cur = self.grow(cur_tree, z_rand)

                else:
                    z_rand = self.random_sample()
                    r, z_des, z_node_cur = self.grow(cur_tree, z_rand)


            # r, z_des, z_node_cur = self.grow(cur_tree, z_rand)
            z_target = copy.deepcopy(z_des)


            if r != GrowStatus.TRAPPED:
                z_rand = copy.deepcopy(z_des)

                r, z_des, z_node = self.grow(other_tree, z_rand)
                z_current = copy.deepcopy(z_des)
                
                if r == GrowStatus.TRAPPED:
                    continue

                while r == GrowStatus.ADVANCED:
                    if time.time() - self.start_time > max_time:
                        self.print('timed out')
                        self.terminate = True
                        break
                    
                    r, z_des, x_node = self.grow(other_tree, z_rand)
                    z_current = copy.deepcopy(z_des)
                
                if self.terminate:
                    break
                    
                z_near_other, z_other_node = other_tree.get_nearest(z_rand)
                new_dist = distance_z(z_rand, z_near_other)
                if new_dist < self.distance_btw_trees:
                    self.distance_btw_trees = new_dist
                    self.print ('[z] Estimated distance to go: {0}'.format(new_dist))

                if r == GrowStatus.REACHED:
                    check_times = check_times + 1
                    # if check_times == 11:
                    #     print('check time.')
                    reached_time = time.time()
                    elapsed = reached_time - self.start_time
                    self.print ('solution reached elapsed_time:',elapsed)
                    self.print ('validity check...')
                    cur_path_z, cur_nodes = cur_tree.get_path(z_node_cur)
                    cur_path_ref, _ = cur_tree.get_path_ref(z_node_cur)
                    time1 = time.time()
                    if self.distance_model is None:
                        result, last_node_cur = self.check_tree_validity(cur_tree, cur_path_z, cur_nodes,z_node_cur)
                    else:
                        if check_times % self.whole_path_check_freq ==0 and self.path_check_method != 'local':
                            result, last_node_cur, cur_path_q_dis, cur_path_q_raw = self.check_tree_validity_dis2(cur_tree, cur_path_z, cur_nodes, z_node_cur)

                        else:
                            result, last_node_cur = self.check_tree_validity(cur_tree, cur_path_z, cur_nodes, z_node_cur)

                    time2 = time.time()
                    check_time = time2 - time1
                    total_check_time = total_check_time + check_time
                    
                    if result is False:
                        if self.use_latent_jump:
                            res, node = self.latent_jump(cur_tree, other_tree, last_node_cur)
                            if res == GrowStatus.REACHED:
                                z_node_cur = node
                                print('use_latent_jump')
                            else:
                                continue
                        else:
                            continue
                    other_path_z, other_nodes = other_tree.get_path(z_other_node)
                    other_path_ref, _ = other_tree.get_path_ref(z_other_node)

                    time1 = time.time()
                    if self.distance_model is None:
                        result, last_node_other = self.check_tree_validity(other_tree, other_path_z, other_nodes, z_other_node)
                    else:
                        if check_times % self.whole_path_check_freq == 0 and self.path_check_method != 'local':
                            result, last_node_other, other_path_q_dis, other_path_q_raw = self.check_tree_validity_dis2(other_tree, other_path_z, other_nodes, z_other_node)

                        else:
                            result, last_node_other = self.check_tree_validity(other_tree, other_path_z,
                                                                                   other_nodes, z_other_node)

                    time2 = time.time()
                    check_time = time2 - time1
                    total_check_time = total_check_time + check_time

                    if result is False:
                        # if self.distance_model is not None and check_times % self.whole_path_check_freq == 0 and self.path_check_method != 'local':
                        #     continue
                        if self.use_latent_jump:
                            res, node = self.latent_jump(other_tree, cur_tree, last_node_other)
                            if res == GrowStatus.REACHED:
                                z_other_node = node
                                print('use_latent_jump')
                            else:
                                continue
                        else:
                            continue
                    self.end_time = time.time()
                    elapsed = self.end_time - self.start_time
                    validity_elapsed = self.end_time - reached_time
                    self.total_check_time = total_check_time
                    self.check_times = check_times
                    self.print ('[z] found a solution! elapsed_time:',elapsed)
                    self.print ('validity check elapsed_time:',validity_elapsed)
                    self.solved_time = elapsed
                    # self.solved_time = reached_time - self.start_time
                    cur_path_z, cur_nodes = cur_tree.get_path(z_node_cur)


                    cur_path_q, cur_nodes2 = cur_tree.get_path_q(z_node_cur)
                    cur_path_q, r = self.interpolate_path(cur_path_q)
                    cur_path_ref, _ = cur_tree.get_path_ref(z_node_cur)

                    other_path_z, other_nodes = other_tree.get_path(z_other_node)
                    other_path_q, other_nodes2 = other_tree.get_path_q(z_other_node)
                    other_path_q, r = self.interpolate_path(other_path_q)

                    other_path_ref, _ = other_tree.get_path_ref(z_other_node)

                    self.print('original_path1\n', z_node_cur.path)
                    self.print('original_path2\n', z_other_node.path)

                    print('len(cur_path_q), len(other_path_q):',len(cur_path_q), len(other_path_q))
                    if is_start_tree:
                        q_path = np.concatenate((cur_path_q, np.flip(other_path_q,axis=0)), axis=0)
                        z_path = np.concatenate((cur_path_z, np.flip(other_path_z,axis=0)), axis=0)
                        ref_path = np.concatenate((cur_path_ref, np.flip(other_path_ref,axis=0)), axis=0)
                    else:
                        q_path = np.concatenate((other_path_q, np.flip(cur_path_q,axis=0)), axis=0)
                        z_path = np.concatenate((other_path_z, np.flip(cur_path_z,axis=0)), axis=0)
                        ref_path = np.concatenate((other_path_ref, np.flip(cur_path_ref,axis=0)), axis=0)


                    return True, z_path, q_path, ref_path, q_path_raw

            if self.use_latent_jump:
                if self.p_q_plan > np.random.rand():
                    q_rand = self.random_sample_q()
                    self.grow_q(cur_tree, q_rand)

            is_start_tree = not is_start_tree

        return False, None, None, None, None

    def latent_jump(self, tree_a, tree_b, node_last):
        q_near, q_near_node = tree_b.get_nearest_q(node_last.q)
        res, q_new, node = self.grow_q(tree_a, q_near, node_last)
        if self.debug:
            self.print ('[z] latent jump')
        
        num_latent_jump_trials = 0
        while res != GrowStatus.TRAPPED:
            num_latent_jump_trials += 1

            if num_latent_jump_trials > self.max_latent_jump_trials:
                return res, node
            
            if res == GrowStatus.REACHED:
                return res, node

            z = self.learned_model.to_latent(q_new)
            q_recon = self.learned_model.to_state(z)

            # arrived to new latent region
            e = np.linalg.norm(q_recon - q_new)
            if e < self.off_manifold_threshold:
                return res, node

            res, q_new, node = self.grow_q(tree_a, q_near, node_last)
            node_last = node

        # If it trapped, add stochastic extension once 
        q_rand = self.random_sample_q()
        res, q_new, node = self.grow_q(tree_a, q_rand)
        return res, node
            
    def random_sample(self, z_current_goal=None, c=None , e=None):
        if self.samplerModel is not None and random.random() < 0.6 and self.useSamplerModel:
            if z_current_goal is None:
                raise ValueError
            z = self.samplerModel.sample(1, z_current_goal, c= self.validity_model.given_c, e = self.validity_model.given_voxel_latent)[0]
            return z[0]

        if self.validity_model is not None:
            z = self.learned_model.sample_with_estimated_validity(1, self.validity_model)[0]
            return z

        z = np.random.normal(0, 1, self.latent_dim)
        
        return z

    def grow(self, tree, z, currentNode = None):
        if currentNode is None:
            z_near, z_near_node = tree.get_nearest(z)
            reach = True
            z_des = z

            dist = distance_z(z_near, z)
            if dist > self.max_distance:
                z_int = self.interpolate(z_near, z, self.max_distance / dist) # TODO: check ratio

                z_des = z_int
                reach = False

            if self.is_valid(z_des) is False:
                # print('TRAPPED')
                return GrowStatus.TRAPPED, z_des, 0

            z_node = tree.add_node(z_near_node, z_des, ref='z')

            if reach:
                # print('REACHED')
                return GrowStatus.REACHED, z_des, z_node
            # print('ADVANCED')
            return GrowStatus.ADVANCED, z_des, z_node
        else:
            # z_near_node = tree.nodes[currentNode]
            z_near_node = currentNode
            z_near = z_near_node.z
            reach = True
            z_des = z

            dist = distance_z(z_near, z)
            if dist > self.max_distance:
                z_int = self.interpolate(z_near, z, self.max_distance / dist)  # TODO: check ratio

                z_des = z_int
                reach = False

            if self.is_valid(z_des) is False:
                return GrowStatus.TRAPPED, z_des, 0

            z_node = tree.add_node(z_near_node, z_des, ref='z')

            if reach:
                return GrowStatus.REACHED, z_des, z_node

            return GrowStatus.ADVANCED, z_des, z_node

    def interpolate(self, z_from, z_to, ratio):
        cdf_zs = norm.cdf([z_from, z_to]) 
        cdf_int = cdf_zs[0] * (1-ratio) + cdf_zs[1] * (ratio)
        z_int = ndtri(cdf_int)

        return z_int

    def random_sample_q(self):
        q = np.random.uniform(self.lb, self.ub)
        return q

    def grow_q(self, tree, q, given_q_near_node=None):
        if given_q_near_node is None:
            q_near, q_near_node = tree.get_nearest_q(q)
        else:
            q_near, q_near_node = given_q_near_node.q, given_q_near_node

        if q_near is not None:
            reach = True

            dist = distance_q(q_near, q)
            if dist > self.max_distance:
                q_int, r = self.interpolate_q(q_near, q, self.max_distance / dist) # TODO: check ratio

                if r == GrowStatus.TRAPPED:
                    self.print('projection failed')
                    return GrowStatus.TRAPPED, q_near, 0

                grow_status, geodesic_states = self.check_motion(q_near, q_int)

                if grow_status != GrowStatus.REACHED:
                    return GrowStatus.TRAPPED, q_near, 0

                q_des = q_int
                reach = False
            else:
                q_pro = copy.deepcopy(q)
                r = self.constraint.project(q_pro)
                if r is False:
                    return GrowStatus.TRAPPED, q_near, 0
                q_des = q_pro

            dist_after_projection = distance_q(q_des, q)
            if dist_after_projection >= dist:
                # went to backward...
                self.print('went to backward...')
                return GrowStatus.TRAPPED, q_des, 0

            if self.is_valid(q=q_des,approx=False) is False:
                return GrowStatus.TRAPPED, q_des, 0

            node = tree.add_node(q_near_node, q=q_des,
                                checked=True, valid=True, projected=True, ref='q', z=self.learned_model.to_latent(q_des[None,:])[0])

            if reach:
                return GrowStatus.REACHED, q_des, node

            return GrowStatus.ADVANCED, q_des, node
        else:
            return GrowStatus.TRAPPED, None, 0

    def interpolate_q(self, q_from, q_to, ratio):
        # returning q_from means interpolation failed
        q_int = (q_to-q_from) * ratio + q_from
        r = self.constraint.project(q_int)
        
        if r is False:
            return q_from, False
        return q_int, True
    
