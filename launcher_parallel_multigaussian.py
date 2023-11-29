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

# TODO: clean all those parallel computations files 

# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'


# TODO: I am not sure if we need those in that case
applicable_configs = {}
default_configs = {}
search_ranges = {}



Path(RESULT_DIR).mkdir(parents=True, exist_ok=True)
                       

# check consistency of configuration dicts
assert set(itertools.chain(*list(applicable_configs.values()))) == {*default_configs.keys(), *search_ranges.keys()}


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
    if not Path(RESULT_DIR).is_dir():
        Path(RESULT_DIR).mkdir(parents=True, exist_ok=True)
    
    exp_path = Path(RESULT_DIR) / args_.date
    if not exp_path.is_dir():
        exp_path.mkdir(parents=True, exist_ok=True)

    command_list = []
    exp_num = 0

    for s in range(len(sigmas)):
        for a in range(len(alphas)):

            # transfer flags from the args
            flags = copy.deepcopy(args.__dict__)
            [flags.pop(key) for key in
            ['exp_name', 'num_cpus', 'num_gpus',\
            'sigma_min', 'sigma_max', 'alpha_min', 'alpha_max',
            'num_seeds_per_hparam']]

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

                seed = init_seeds[j]

                run_results_folder = exp_path / str(seed)
                if not run_results_folder.is_dir():
                    run_results_folder.mkdir(parents=True, exist_ok=True)

                flags['exp_result_folder'] = str(run_results_folder)

                cmd = generate_base_command(run_exp_multigaussian,
                                             flags=dict(**flags, **{'seed': seed}))
                command_list.append(cmd)
                exp_num+=1

        # submit jobs
        generate_run_commands(command_list, num_gpus=args.num_gpus, num_cpus=args.num_cpus, mode=mode, promt=False)

        # Create the resulting json file
        final_results = {}

        json_list = [p for p in exp_path.rglob("**.json") if p.stem.startswith("result")]
        n = 0
        for p in json_list:
            with open(str(p), "r") as json_file:
                res = json.load(json_file)
                final_results.update({str(n): res})
            n += 1
        
        output_path = exp_path / f"results_final_{n}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(str(output_path), "w") as output_file:
            json.dump(final_results, output_file, indent=2)




if __name__ == '__main__':

    """
    Test Commmand
    PYTHONPATH=$PWD python launcher_parallel_multigaussian.py --grid_size 2 --n 10 --n_val 10 --n_ergodic 10 --d 2 --depth 0 --normalization true --horizon 10
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default=str(datetime.datetime.now()).replace(" ", "_").replace(":", "_").split(".")[0])
    parser.add_argument('--mode', type=str, default="euler_slurm")
    parser.add_argument('--long', type=int, default=0)

    parser.add_argument('--num_cpus', type=int, default=10)
    parser.add_argument('--num_gpus', type=int, default=0)

    # Parameters which are launcher specific
    parser.add_argument('--sigma_min', type=float, default=0.001)
    parser.add_argument('--sigma_max', type=float, default=0.1)
    parser.add_argument('--alpha_min', type=float, default=1.2)
    parser.add_argument('--alpha_max', type=float, default=2.)
    parser.add_argument('--grid_size', type=int, default=7)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_seeds_per_hparam', type=int, default=1)

    # Parameters that are shared among all runs
    parser.add_argument('--horizon', type=int, default=10000)
    parser.add_argument('--d', type=int, default=10)
    parser.add_argument('--eta', type=float, default=0.001)
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--n_val', type=int, default=1000)
    parser.add_argument('--n_ergodic', type=int, default=5000)
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--momentum', type=float, default=0.)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--width', type=int, default=50)
    parser.add_argument('--normalization', type=bool, default=False)

    args = parser.parse_args()
    main(args)
