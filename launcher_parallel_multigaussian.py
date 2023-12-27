import datetime
import json

from experiment_utils import generate_base_command, generate_run_commands, hash_dict, sample_flag, RESULT_DIR

import run_exp_multigaussian
import argparse
import numpy as np
import copy
import os
import itertools
from pathlib import Path
from loguru import logger

# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'


# TODO: I am not sure if we need those in that case
applicable_configs = {}
default_configs = {}
search_ranges = {}

                       

# TODO: is this useful?
# check consistency of configuration dicts
# assert set(itertools.chain(*list(applicable_configs.values()))) == {*default_configs.keys(), *search_ranges.keys()}


def main(args_):
    
    sigmas = list(np.exp(np.linspace(np.log(args_.sigma_min), \
                                     np.log(args_.sigma_max), \
                                     args_.grid_size)))
    alphas = list(np.linspace(args_.alpha_min, \
                              args_.alpha_max, \
                              args_.grid_size)) 
    
    mode = args_.mode
    rds = np.random.RandomState(args_.seed)
    assert args_.num_seeds_per_hparam < 100
    init_seeds = list(rds.randint(0, 10 ** 6, size=(args_.num_seeds_per_hparam,)))

    # determine name of experiment
    if not Path(args_.result_dir).is_dir():
        Path(args_.result_dir).mkdir(parents=True, exist_ok=True)
    
    exp_path = Path(args_.result_dir) / args_.date
    if not exp_path.is_dir():
        exp_path.mkdir(parents=True, exist_ok=True)

    arg_path = exp_path / "arguments.json"
    with open(str(arg_path), "w") as exp_file:
        logger.info(f"Saving argments in {str(arg_path)}")
        json.dump(args_.__dict__, exp_file, indent=2)

    command_list = []
    exp_num = 0

    for s in range(len(sigmas)):
        for a in range(len(alphas)):

            # transfer flags from the args
            flags = copy.deepcopy(args.__dict__)
            [flags.pop(key) for key in
            ['num_cpus', 'num_gpus',\
            'sigma_min', 'sigma_max', 'alpha_min', 'alpha_max',
            'num_seeds_per_hparam', 'grid_size', 'seed',
            'date']]

            # randomly sample flags
            for flag in default_configs:
                if flag in search_ranges:
                    flags[flag] = sample_flag(sample_spec=search_ranges[flag], rds=rds)
                else:
                    flags[flag] = default_configs[flag]

            flags['sigma'] = sigmas[s]
            flags['alpha'] = alphas[a]
            flags['id_sigma'] = s
            flags['id_alpha'] = a

            for j in range(args.num_seeds_per_hparam):

                np.random.seed(init_seeds[j])

                data_seed = np.random.randint(1000)
                model_seed = np.random.randint(1000)

                flags['data_seed'] = data_seed
                flags['model_seed'] = model_seed

                run_results_folder = exp_path / str(init_seeds[j])
                if not run_results_folder.is_dir():
                    run_results_folder.mkdir(parents=True, exist_ok=True)
                logger.info(f"{str(run_results_folder)}_sigma_{sigmas[s]}_alpha_{alphas[a]}")

                flags['result_dir'] = str(run_results_folder)

                cmd = generate_base_command(run_exp_multigaussian,
                                             flags=dict(**flags))
                command_list.append(cmd)
                exp_num+=1

    # submit jobs
    generate_run_commands(command_list, 
                          num_gpus=args.num_gpus,
                            num_cpus=args.num_cpus, 
                            mode=mode, 
                            promt=False,
                            long=args_.long)



"""
Test Commmand
PYTHONPATH=$PWD python launcher_parallel_multigaussian.py --grid_size 2 --n 10 --n_val 10 --n_ergodic 10 --d 2 --depth 0 --horizon 10 --compute_gradients 1 --result_dir tests_directory 
"""


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default=str(datetime.datetime.now()).replace(" ", "_").replace(":", "_").split(".")[0])
    parser.add_argument('--mode', type=str, default="euler_slurm")
    parser.add_argument('--long', type=int, default=0)

    parser.add_argument('--num_cpus', type=int, default=15)
    parser.add_argument('--num_gpus', type=int, default=0)

    # Parameters which are launcher specific
    parser.add_argument('--sigma_min', type=float, default=0.1)
    parser.add_argument('--sigma_max', type=float, default=10.)
    parser.add_argument('--alpha_min', type=float, default=1.4)
    parser.add_argument('--alpha_max', type=float, default=2.)
    parser.add_argument('--grid_size', type=int, default=10)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--num_seeds_per_hparam', type=int, default=1)

    parser.add_argument('--result_dir', type=str, default=RESULT_DIR)

    # Parameters that are shared among all runs
    parser.add_argument('--horizon', type=int, default=10000)
    parser.add_argument('--d', type=int, default=4)
    parser.add_argument('--eta', type=float, default=0.001)
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--n_val', type=int, default=100)
    parser.add_argument('--n_ergodic', type=int, default=5000)
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--decay', type=float, default=0.01)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--width', type=int, default=50)
    parser.add_argument('--normalization', type=bool, default=False)
    parser.add_argument('--compute_gradients', type=int, default=1)
    parser.add_argument('--bias', type=int, default=0)
    parser.add_argument('--data_type', type=str, default="mnist")
    parser.add_argument('--stopping', type=bool, default=False) # whether or not use the stopping criterion

    # parameters used onlyfor mnist, or other image datasets
    parser.add_argument('--subset', type=float, default=0.01)
    parser.add_argument('--resize', type=int, default=28) # original size of mnist is 28
    parser.add_argument('--classes', nargs='+', required=False, default=None) # classes used in training


    args = parser.parse_args()
    main(args)
