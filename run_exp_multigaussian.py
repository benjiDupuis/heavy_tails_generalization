import os
import sys
import argparse
from util import Logger, hash_dict
import logging
import datetime
import json
import os
from pathlib import Path

import fire
import numpy as np
import wandb
from loguru import logger
from pydantic import BaseModel

from PHDim.train_risk_analysis import main as risk_analysis

def main(args_):
    """"""

    ''' generate experiment hash and set up redirect of output streams '''
    exp_hash = hash_dict(args_.__dict__)
    if args_.exp_result_folder is not None:
        os.makedirs(args_.exp_result_folder, exist_ok=True)
        log_file_path = os.path.join(args_.exp_result_folder, '%s.log ' % exp_hash)
        logger = Logger(log_file_path)
        sys.stdout = logger
        sys.stderr = logger

    logger = logging.getLogger("root")

    class CheckTypesFilter(logging.Filter):
        def filter(self, record):
            return "check_types" not in record.getMessage()

    logger.addFilter(CheckTypesFilter())

    """"""""""""""""""""""""""""""""""" Hyperparameter selection """""""""""""""""""""""""""""""""""""""""""""""

    _ = risk_analysis(eval_freq = 10000,
                    lr = args_.lr,
                    iterations = 100000000,
                    width = args_.width,
                    depth = args_.depth,
                    batch_size_train = args_.batch_size,
                    batch_size_eval=5000,
                    ripser_points = 5000,
                    jump = 20,
                    min_points = 1000,
                    dataset = "mnist",
                    data_path="~/data",
                    model = "fc",
                    stopping_criterion = 0.005,
                    data_proportion=1.,
                    exp_result_folder = args_.exp_result_folder,
                    seed=args_.seed,
                    metric=args_.metric,
                    subset=args_.subset)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Approximating SGD')

    # general args
    parser.add_argument('--experiment', type=str, default='heavy_tails')
    parser.add_argument('--method', type=str, default='approximation')
    parser.add_argument('--mode', type=str, default="slurm")
    parser.add_argument('--long', type=int, default=0)
    parser.add_argument('--env', type=int, default=0)

    parser.add_argument('--exp_result_folder', type=str, default=None)
    parser.add_argument('--seed', type=int, default=834, help='random number generator seed')

    # method related args
    parser.add_argument('--dim', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--width', type=int, default=100)
    parser.add_argument('--depth', type=int, default=5)
    parser.add_argument('--subset', type=float, default=1.)
    parser.add_argument('--metric', type=str, default="manhattan")


    

    args = parser.parse_args()
    main(args)
