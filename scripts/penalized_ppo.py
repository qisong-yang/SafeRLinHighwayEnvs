from safe_rl import ppo_lagrangian, ppo
import gym
import highway_env
import argparse
from multiprocessing import Pool
from itertools import product



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', nargs='*', type=int, default=123)
    parser.add_argument('--env_name', nargs='*', type=str, default="mergefast-v0")
    parser.add_argument('--cost_lim', type=float, default=0.01)
    parser.add_argument('--penalty_init', type=float, default=10)
    parser.add_argument('--penalty_lr', type=float, default=0.05)
    parser.add_argument('--penalty_iters', type=int, default=40)
    parser.add_argument('--observation', type=str, default="LIST")
    parser.add_argument('--coop', nargs='*', type=float, default=0.0)

    args = parser.parse_args()

    seeds = args.seed if isinstance(args.seed, list) else [args.seed]
    env_names = args.env_name if isinstance(args.env_name, list) else [args.env_name]
    coops = args.coop if isinstance(args.coop, list) else [args.coop]

    def launch_ppo_lagrangian(seed, env_name, coop):
        ppo_lagrangian(
            env_fn = lambda : gym.make(env_name, observation=args.observation, cooperative_prob=coop),
            ac_kwargs = dict(hidden_sizes=(64,64)),
            logger_kwargs = {"output_dir": f"./results/timeout2_ppo_lag_{env_name}_coop{coop}_cost_lim_{args.cost_lim}_penalty_init_{args.penalty_init}_penalty_lr_{args.penalty_lr}_penalty_iters_{args.penalty_iters}_seed_{seed}"},
            render=False,
            max_ep_len=100,
            epochs=150,
            steps_per_epoch=5000,
            cost_lim=args.cost_lim,
            penalty_init=args.penalty_init,
            penalty_lr=args.penalty_lr,
            penalty_iters=args.penalty_iters,
            seed=seed,
        )

    with Pool() as pool:
        pool.starmap(launch_ppo_lagrangian, product(seeds, env_names, coops))


