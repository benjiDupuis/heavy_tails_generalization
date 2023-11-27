import json

from experiment_utils import generate_base_command, generate_run_commands, hash_dict, sample_flag, RESULT_DIR

import run_example_exp_mnist
import argparse
import numpy as np
import copy
import os
import itertools
from pathlib import Path

# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

applicable_configs = {
    'param_estim': ['dim'],
}

default_configs = {
    'dim': 100,
}

search_ranges = {
    'dim': ['choice', [100]],
}
                       
lr_min = 0.005
lr_max = 0.1
bs_min = 32
bs_max = 256

Path(RESULT_DIR).mkdir(parents=True, exist_ok=True)
                       
lrs = list(np.exp(np.linspace(np.log(lr_min), np.log(lr_max), 6)))
bs = list(np.linspace(bs_min, bs_max, 6, dtype=np.int64))

# check consistency of configuration dicts
assert set(itertools.chain(*list(applicable_configs.values()))) == {*default_configs.keys(), *search_ranges.keys()}


def main(args_):
    mode = args_.mode
    rds = np.random.RandomState(args_.seed)
    assert args_.num_seeds_per_hparam < 100
    init_seeds = list(rds.randint(0, 10 ** 6, size=(100,)))

    # determine name of experiment
    exp_base_path = os.path.join(RESULT_DIR, args_.exp_name)
    exp_path = os.path.join(exp_base_path, f'{args_.experiment}_{args_.method}')

    command_list = []
    exp_num = 0
    for (lr, b) in itertools.product(lrs, bs):
        # transfer flags from the args
        flags = copy.deepcopy(args.__dict__)
        [flags.pop(key) for key in
         ['seed', 'num_hparam_samples', 'num_seeds_per_hparam', 'exp_name', 'num_cpus', 'num_gpus']]

        # randomly sample flags
        for flag in default_configs:
            if flag in search_ranges:
                flags[flag] = sample_flag(sample_spec=search_ranges[flag], rds=rds)
            else:
                flags[flag] = default_configs[flag]

        flags['lr'] = lr
        flags['b'] = b

        # determine subdir which holds the repetitions of the exp
        flags_hash = str(exp_num)

        for j in range(args.num_seeds_per_hparam):

            seed = init_seeds[j]
            flags['exp_result_folder'] = os.path.join(exp_path, str(seed), flags_hash)

            cmd = generate_base_command(run_example_exp_mnist, flags=dict(**flags, **{'seed': seed}))
            command_list.append(cmd)
            exp_num+=1

    # submit jobs
    generate_run_commands(command_list, num_gpus=args.num_gpus, num_cpus=args.num_cpus, mode=mode, promt=False)

    # Create the resulting json file
    exp_path = Path(exp_path)
    final_results = {}

    json_list = [p for p in exp_path.rglob("*.json") if str(p).startswith("result")]
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
    import datetime
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='ph_chd')
    parser.add_argument('--method', type=str, default=str(datetime.datetime.now()).replace(" ", "_").replace(":", "_").split(".")[0])
    parser.add_argument('--mode', type=str, default="euler_slurm")
    parser.add_argument('--long', type=int, default=0)

    parser.add_argument('--seed', type=int, default=2, help='random number generator seed')
    parser.add_argument('--exp_name', type=str, required=True, default=None)
    parser.add_argument('--num_cpus', type=int, default=2, help='number of cpus to use')
    parser.add_argument('--num_gpus', type=int, default=0, help='number of gpus to use')

    parser.add_argument('--num_hparam_samples', type=int, default=1)
    parser.add_argument('--num_seeds_per_hparam', type=int, default=1)

    parser.add_argument('--width', type=int, default=100)
    parser.add_argument('--depth', type=int, default=5)
    parser.add_argument('--subset', type=float, default=1.)
    parser.add_argument('--metric', type=str, default="manhattan")

    args = parser.parse_args()
    main(args)
