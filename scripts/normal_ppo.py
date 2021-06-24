from safe_rl import ppo_lagrangian, ppo
import gym
import highway_env
import argparse
from multiprocessing import Pool
from itertools import product

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', nargs='*', type=int, default=123)
    parser.add_argument('--env_name', type=str, default="mergefast-v0")
    parser.add_argument('--penalty_init', type=float, default=0.01)
    parser.add_argument('--observation', type=str, default="LIST")
    parser.add_argument('--coop', nargs='*', type=float, default=0.0)

    args = parser.parse_args()

    seeds = args.seed if isinstance(args.seed, list) else [args.seed]
    env_names = args.seed if isinstance(args.env_name, list) else [args.env_name]
    coops = args.coop if isinstance(args.coop, list) else [args.coop]

    def launch_ppo_normal(seed, env_name, coop):
        ppo(
            env_fn = lambda : gym.make(env_name, observation=args.observation, cooperative_prob=coop),
            ac_kwargs = dict(hidden_sizes=(64,64)),
            logger_kwargs = {"output_dir": f"./results/normal_ppo/normal_ppo_Env{env_name}_coop{coop}_penalty_init_{args.penalty_init}_seed_{seed}"},
            render=False,
            max_ep_len=100,
            epochs=150,
            steps_per_epoch=5000,
            penalty_init=args.penalty_init,
            seed=seed,
        )
    with Pool() as pool:
        pool.starmap(launch_ppo_normal, product(seeds, env_names, coops))