import os
import yaml
import numpy as np
import torch
import pickle

import argparse

from termcolor import colored

from lcbirrt_lpo.utils.model_utils_zjw import (benchmark, precomputedRoadmap,
                                         generate_path_data, generate_scene_start_goal,
                                         generate_scene_config, load_model)
from ljcmp.utils.generate_environment import generate_environment

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', '-E', type=str, default='panda_dual', help='panda_orientation, panda_dual, panda_dual_orientation, panda_triple')
parser.add_argument('--seed', type=int, default=1107)
parser.add_argument('--nodeNum', type=int, default=None)
parser.add_argument('--use_given_start_goal', type=bool, default=True)
parser.add_argument('--use_given_condition', type=bool, default=False)
parser.add_argument('--use_sample_model', type=bool, default=False)
parser.add_argument('--use_distance_model', type=bool, default=True)
parser.add_argument('--path_check_method', type=str, default='whole') # whole local both
parser.add_argument('--whole_path_check_freq', type=int, default=5) # whole local both
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--display', type=bool, default=False)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--method', '-M', type=str, default='latent_rrt_latent_jump', help='latent_rrt, '
                                                                                        'latent_rrt_latent_jump, '
                                                                                        'sampling_rrt_project, '
                                                                                        'sampling_rrt_latent_sample, '
                                                                                        'precomputed_roadmap_prm, '
                                                                                        'precomputed_graph_rrt, '
                                                                                        'project_rrt, '
                                                                                        'generate_scene_config')
parser.add_argument('--test_scene_start_idx', type=int, default=500)
parser.add_argument('--num_test_scenes', type=int, default=100)
parser.add_argument('--trials', type=int, default=1)


args = parser.parse_args()
if args.device == 'cuda':
    if torch.cuda.is_available():
        args.device = 'cuda'
    else:
        args.device = 'cpu'
        print(colored('CUDA is not available, use CPU instead', 'red'))
print(colored('Using device: {}'.format(args.device), 'green'))

device = args.device

np.random.seed(args.seed)
torch.manual_seed(args.seed)

constraint, model_info, condition, update_scene_from_yaml, set_constraint, _ = generate_environment(args.exp_name)

constraint_model=None
print(colored(' ---- Start benchmarking ----', 'green'))
print('exp_name :', args.exp_name)
print('tag      :', model_info['constraint_model']['tag'])
print('method   :', args.method)
if 'latent_rrt' in args.method:
    print('use_distance_model   :', args.use_distance_model)
    print('path_check_method   :', args.path_check_method)
    print('path_check_freq   :', args.whole_path_check_freq)

np.set_printoptions(precision=6, suppress=True)

if args.method == 'generate_scene_config':
    generate_scene_config(constraint,
                          constraint_model,
                          model_info,
                          condition,
                          update_scene_from_yaml,
                          start=0,
                          end=500,
                          config_size=100)
else:
    results = benchmark(exp_name=args.exp_name,
                        model_info=model_info,
                        method=args.method,
                        constraint=constraint,
                        set_constraint=set_constraint,
                        device=device,
                        nodeNum = args.nodeNum,
                        condition=condition,
                        use_given_start_goal=args.use_given_start_goal,
                        update_scene_from_yaml=update_scene_from_yaml,
                        debug=args.debug,
                        display=args.display,
                        trials=args.trials,
                        test_scene_start_idx=args.test_scene_start_idx,
                        num_test_scenes=args.num_test_scenes,
                        load_validity_model=True,
                        use_given_condition=args.use_given_condition,
                        use_sample_model = args.use_sample_model,
                        use_distance_model = args.use_distance_model,
                        path_check_method = args.path_check_method,
                        whole_path_check_freq = args.whole_path_check_freq)

    # model_tag = model_info['constraint_model']['tag']
    # model_tag = 'tsa_4000'
    if args.use_distance_model:
        # model_tag = args.exp_name + '_distance_model_500-600_' + args.path_check_method
        # model_tag = args.exp_name + '_distance_model_500-600_no_every_check'
        # model_tag = args.exp_name + '_distance_model_500-600_no_freq_check'
        if args.path_check_method == 'local':
            model_tag = f'{args.exp_name}_distance_model_500-600_{args.path_check_method}'
        else:
            model_tag = f'{args.exp_name}_distance_model_500-600_{args.path_check_method}_{args.whole_path_check_freq}'
    else:
        model_tag = args.exp_name + '_no_distance_model_500-600'
    result_save_dir = f'result/{args.exp_name}/{args.method}/{model_tag}/'
    os.makedirs(result_save_dir, exist_ok=True)

    print(colored(' ---- Benchmarking finished ----', 'green'))
    print('test suc rate', results['success_rate'])
    print('avg time', results['mean_test_times'])
    print('std time', results['std_test_times'])
    print('avg path length', results['mean_test_path_lenghts'])
    print('std path length', results['std_test_path_lenghts'])
    if 'latent_rrt' in args.method:
        print('avg check time', results['mean_check_times'])
        print('std check time', results['std_check_times'])

    print(colored(' ---- Saving results ----', 'green'))
    print('result_save_dir', result_save_dir)

    pickle.dump(results, open(f'{result_save_dir}/test_result.pkl', 'wb'))

    if 'latent_rrt' in args.method:
        results_overview = {'success_rate': results['success_rate'],
                           'mean_test_times': results['mean_test_times'].tolist(),
                           'std_test_times': results['std_test_times'].tolist(),
                           'mean_test_path_lenghts': results['mean_test_path_lenghts'].tolist(),
                           'std_test_path_lenghts': results['std_test_path_lenghts'].tolist(),
                           'mean_check_times': results['mean_check_times'].tolist(),
                           'std_check_times': results['std_check_times'].tolist()}
    else:
        results_overview = {'success_rate': results['success_rate'],
                            'mean_test_times': results['mean_test_times'].tolist(),
                            'std_test_times': results['std_test_times'].tolist(),
                            'mean_test_path_lenghts': results['mean_test_path_lenghts'].tolist(),
                            'std_test_path_lenghts': results['std_test_path_lenghts'].tolist()}

    yaml.dump(results_overview, open(f'{result_save_dir}/test_result_overview.yaml', 'w'))
