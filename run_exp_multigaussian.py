import os
import sys
import argparse
from util import Logger, hash_dict
import logging
import os

from last_point.simulation import run_and_save_one_simulation


def main(args_):
    """"""

    ''' generate experiment hash and set up redirect of output streams '''
    exp_hash = hash_dict(args_.__dict__)
    # TODO: I removed logs saving, do we need it?
    # if args_.result_dir is not None:
    #     os.makedirs(args_.result_dir, exist_ok=True)
    #     log_file_path = os.path.join(args_.result_dir, '%s.log ' % exp_hash)
    #     logger = Logger(log_file_path)
    #     sys.stdout = logger
    #     sys.stderr = logger

    logger = logging.getLogger("root")

    class CheckTypesFilter(logging.Filter):
        def filter(self, record):
            return "check_types" not in record.getMessage()

    logger.addFilter(CheckTypesFilter())

    """"""""""""""""""""""""""""""""""" Hyperparameter selection """""""""""""""""""""""""""""""""""""""""""""""


    run_and_save_one_simulation(args_.result_dir,
                                args_.horizon,
                                args_.d,
                                args_.eta,
                                args_.sigma,
                                args_.alpha,
                                args_.n,
                                args_.n_val,
                                args_.n_ergodic,
                                args_.n_classes,
                                args_.momentum,
                                args_.depth,
                                args_.width,
                                args_.data_seed,
                                args_.model_seed,
                                args_.normalization,
                                args_.id_sigma,
                                args_.id_alpha)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Heavy-tailed simulations')

    # general args
    # parser.add_argument('--mode', type=str, default="slurm")
    parser.add_argument('--mode', type=str, default="euler_slurm")
    parser.add_argument('--long', type=int, default=0)
    parser.add_argument('--env', type=int, default=0)

    parser.add_argument('--result_dir', type=str, default=None)

    # run related args
    # It is: global parameters + (sigma, alpha)
    parser.add_argument('--horizon', type=int, default=10000)
    parser.add_argument('--d', type=int, default=10)
    parser.add_argument('--eta', type=float, default=0.001)
    parser.add_argument('--sigma', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=1.5)
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--n_val', type=int, default=1000)
    parser.add_argument('--n_ergodic', type=int, default=5000)
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--momentum', type=float, default=0.)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--width', type=int, default=50)
    parser.add_argument('--data_seed', type=int, default=1)
    parser.add_argument('--model_seed', type=int, default=42)
    parser.add_argument('--normalization', type=bool, default=False)
    parser.add_argument('--id_sigma', type=int, default=0)
    parser.add_argument('--id_alpha', type=int, default=0)
    

    args = parser.parse_args()
    main(args)
