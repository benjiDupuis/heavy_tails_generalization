"""Adapated from https://github.com/lasgroup/simulation_transfer by Jonas Rothfuss"""

import sys
import os
import stat
import json
import glob
import numpy as np
import pandas as pd
import subprocess
import multiprocessing

from typing import Dict, Optional, Any, List

""" Relevant Directories """

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = "./results_parallel"

LONG = 23
SHORT = 3


class AsyncExecutor:
    """ Async executer """

    def __init__(self, n_jobs=1):
        self.num_workers = n_jobs if n_jobs > 0 else multiprocessing.cpu_count()
        self._pool = []
        self._populate_pool()

    def run(self, target, *args_iter, verbose=False):
        workers_idle = [False] * self.num_workers
        tasks = list(zip(*args_iter))
        n_tasks = len(tasks)

        while not all(workers_idle):
            for i in range(self.num_workers):
                if not self._pool[i].is_alive():
                    self._pool[i].terminate()
                    if len(tasks) > 0:
                        if verbose:
                            print(n_tasks - len(tasks))
                        next_task = tasks.pop(0)
                        self._pool[i] = _start_process(target, next_task)
                    else:
                        workers_idle[i] = True

    def _populate_pool(self):
        self._pool = [_start_process(_dummy_fun) for _ in range(self.num_workers)]


def _dummy_fun():
    pass


def _start_process(target, args=None):
    if args:
        p = multiprocessing.Process(target=target, args=args)
    else:
        p = multiprocessing.Process(target=target)
    p.start()
    return p


def generate_base_command(module, flags: Optional[Dict[str, Any]] = None, unbuffered: bool = True) -> str:
    """ Generates the command to execute python module with provided flags

    Args:
        module: python module / file to run
        flags: dictonary of flag names and the values to assign to them.
               assumes that boolean flags are encoded as store_true flags with False as default.
        unbuffered: whether to invoke an unbuffered python output stream

    Returns: (str) command which can be executed via bash

    """

    """ Module is a python file to execute """
    interpreter_script = sys.executable
    base_exp_script = os.path.abspath(module.__file__)
    if unbuffered:
        base_cmd = interpreter_script + ' -u ' + base_exp_script
    else:
        base_cmd = interpreter_script + ' ' + base_exp_script
    if flags is not None:
        assert isinstance(flags, dict), "Flags must be provided as dict"
        for flag, setting in flags.items():
            if type(setting) == bool or type(setting) == np.bool_:
                if setting:
                    base_cmd += f" --{flag}"
                    # base_cmd += f" --{flag.replace('_', '-')}"
            else:
                base_cmd += f" --{flag}={setting}"
                # base_cmd += f" --{flag.replace('_', '-')}={setting}"
    return base_cmd


def generate_run_commands(command_list: List[str], num_cpus: int = 1, num_gpus: int = 0,
                          dry: bool = False, n_hosts: int = 1, mem: int = 6000, long: bool = True,  # fixme
                          mode: str = 'local', promt: bool = True) -> None:
    if mode == 'euler_lsf':
        cluster_cmds = []
        bsub_cmd = 'bsub ' + \
                   f'-W {LONG if long else SHORT}:59 ' + \
                   f'-R "rusage[mem={mem}]" ' + \
                   f'-n {num_cpus} ' + \
                   f'-R "span[hosts={n_hosts}]" ' + \
                   f'-o lsf_outputs/ '

        if num_gpus > 0:
            bsub_cmd += f'-R "rusage[ngpus_excl_p={num_gpus}]" '

        for python_cmd in command_list:
            cluster_cmds.append(bsub_cmd + python_cmd)

        if promt:
            answer = input(f"About to submit {len(cluster_cmds)} compute jobs to the cluster. Proceed? [yes/no]")
        else:
            answer = 'yes'
        if answer == 'yes':
            for cmd in cluster_cmds:
                if dry:
                    print(cmd)
                else:
                    os.system(cmd)

    elif mode == 'euler_slurm':
        cluster_cmds = []

        if num_gpus >= 1:
            partition_string = "#SBATCH --partition=gpu\n" +\
                f"#SBATCH --gres=gpu:rtx8000:{num_gpus}\n"
        else:
            partition_string = "#SBATCH --partition=cpu_devel\n"

        log_string_file = "#SBATCH --output=%x.out.%j\n"  +\
                "#SBATCH --error=%x.err.%j\n" 


        # os.system('mkdir slurm_outputs')
        bsub_cmd = '#!/bin/bash\n\n' + \
                   f'#SBATCH --time={LONG if long else SHORT}:59:59\n' + \
                   f'#SBATCH --mem-per-cpu={mem}\n' + \
                   f'#SBATCH -n {num_cpus}\n' + \
                   f'#SBATCH -o slurm_outputs/out_{np.random.randint(10000000)}.txt\n' + \
                   partition_string +\
                   log_string_file   
        
        # if num_gpus > 0:
        #     bsub_cmd += f'#SBATCH--gpus={num_gpus}\n'

        for python_cmd in command_list:
            cluster_cmds.append(bsub_cmd + f"\n{python_cmd}\n")

        if promt:
            answer = input(f"About to submit {len(cluster_cmds)} compute jobs to the cluster. Proceed? [yes/no]")
        else:
            answer = 'yes'
        if answer == 'yes':
            for cmd in cluster_cmds:
                if dry:
                    print(cmd)
                else:
                    f = open("temp_script.sh", "w")
                    f.write(cmd)
                    f.close()
                    st = os.stat('temp_script.sh')
                    os.chmod('temp_script.sh', st.st_mode | stat.S_IEXEC)
                    os.system('sbatch temp_script.sh')
                    os.system('rm temp_script.sh')

    elif mode == 'local':
        if promt:
            answer = input(f"About to run {len(command_list)} jobs in a loop. Proceed? [yes/no]")
        else:
            answer = 'yes'

        if answer == 'yes':
            for cmd in command_list:
                if dry:
                    print(cmd)
                else:
                    os.system(cmd)

    elif mode == 'local_async':
        if promt:
            answer = input(
                f"About to launch {len(command_list)} commands in {num_cpus} local processes. Proceed? [yes/no]")
        else:
            answer = 'yes'

        if answer == 'yes':
            if dry:
                for cmd in command_list:
                    print(cmd)
            else:
                exec = AsyncExecutor(n_jobs=num_cpus)
                cmd_exec_fun = lambda cmd: os.system(cmd)
                exec.run(cmd_exec_fun, command_list)
    else:
        raise NotImplementedError


# Hashing and Encoding dicts to JSON
class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)


def hash_dict(d):
    return str(abs(json.dumps(d, sort_keys=True, cls=NumpyArrayEncoder).__hash__()))


def sample_flag(sample_spec, rds=None):
    """ Randomly sampling flags """
    if rds is None:
        rds = np.random
    assert len(sample_spec) == 2

    sample_type, range = sample_spec
    if sample_type == 'loguniform':
        assert len(range) == 2
        return 10 ** rds.uniform(*range)
    elif sample_type == 'uniform':
        assert len(range) == 2
        return rds.uniform(*range)
    elif sample_type == 'choice':
        return rds.choice(range)
    else:
        raise NotImplementedError


def collect_exp_results(exp_name: str, dir_tree_depth: int = 3, verbose: bool = True):
    """ Collecting the exp result"""
    exp_dir = os.path.join(RESULT_DIR, exp_name)
    no_results_counter = 0
    success_counter = 0
    exp_dicts = []
    param_names = set()
    search_path = os.path.join(exp_dir, '/'.join(['*' for _ in range(dir_tree_depth)]) + '.json')
    results_jsons = glob.glob(search_path)
    for results_file in results_jsons:

        if os.path.isfile(results_file):
            try:
                with open(results_file, 'r') as f:
                    exp_dict = json.load(f)
                if isinstance(exp_dict, dict):
                    exp_dicts.append({**exp_dict['evals'], **exp_dict['params']})
                    param_names = param_names.union(set(exp_dict['params'].keys()))
                elif isinstance(exp_dict, list):
                    exp_dicts.extend([{**d['evals'], **d['params']} for d in exp_dict])
                    for d in exp_dict:
                        param_names = param_names.union(set(d['params'].keys()))
                else:
                    raise ValueError
                success_counter += 1
            except json.decoder.JSONDecodeError as e:
                print(f'Failed to load {results_file}', e)
        else:
            no_results_counter += 1

    assert success_counter + no_results_counter == len(results_jsons)
    if verbose:
        print(f'Parsed results in {search_path} - found {success_counter} folders with results'
              f' and {no_results_counter} folders without results')

    return pd.DataFrame(data=exp_dicts), list(param_names)


# Some aggregation functions
def ucb(row):
    assert row.shape[0] > 1
    return np.quantile(row, q=0.95, axis=0)


def lcb(row):
    assert row.shape[0] > 1
    return np.quantile(row, q=0.05, axis=0)


def median(row):
    assert row.shape[0] > 1
    return np.quantile(row, q=0.5, axis=0)


def count(row):
    return row.shape[0]
