#!/usr/bin/env python

import time
import numpy as np
from safe_rl.utils.load_utils import load_policy
from safe_rl.utils.logx import EpochLogger
import gym
from test_policy import run_policy
checkpoints = dict()
checkpoints["PPO-SLOW"]="normal_ppo_Envmergeslow-v0_penalty_init_100.0_seed_1"
checkpoints["PPO-FAST"]="normal_ppo_Envmergefast-v0_penalty_init_100.0_seed_1"
checkpoints["PPO-LAG-SLOW"]="ppo-lag/ppo_lag_mergeslow-v0_cost_lim_0.01_penalty_init_10_penalty_lr_0.05_penalty_iters_40_seed_2_pi_lr_0.0003" #"k3_lag_ppo_mixed"#
checkpoints["PPO-LAG-FAST"]="ppo-lag/ppo_lag_mergefast-v0_cost_lim_0.01_penalty_init_10_penalty_lr_0.05_penalty_iters_40_seed_1"

policies = ["PPO-SLOW",
            "PPO-FAST",
            "PPO-LAG-SLOW",
            "PPO-LAG-FAST"
            ]
env_names = ["mergeslow-v0",
             "mergefast-v0",
             "mergemixed-v0",]
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=100)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')

    args = parser.parse_args()

    for policy in policies:
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   Policy:: {}  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%".format(policy))

        for env_name in env_names:
            env, get_action, sess = load_policy(checkpoints[policy],
                                                args.itr if args.itr >=0 else 'last',
                                                args.deterministic)
            env=gym.make(env_name)
            print("%%%%%%%%%%% Environment: {}  %%%%%%%%%%%".format(env_name))
            avg_cost, avg_length = run_policy(env, get_action, args.len, args.episodes, not(args.norender))
            print("Avg cost: {}, Avg. Length: {}".format(avg_cost, avg_length))