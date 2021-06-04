from safe_rl import ppo_lagrangian, ppo
import gym
import highway_env
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', nargs='*', default=123)
    parser.add_argument('--cost_lim', nargs='*', type=list, default=0.001)
    args = parser.parse_args()

    seeds = args.seed if isinstance(args.seed, list) else [args.seed]
    cost_lims = args.cost_lim if isinstance(args.cost_lim, list) else [args.cost_lim]
    exps = [
        dict(
            env_fn = lambda : gym.make("merge-v0"),
            ac_kwargs = dict(hidden_sizes=(64,64)),
            logger_kwargs = {"output_dir": f"./results/ppo_lag_cost_lim_{cost_lim}_seed_{seed}"},
            render=False,
            max_ep_len=70,
            epochs=600,
            steps_per_epoch=5000,
            penalty_iters=40,
            cost_lim=cost_lim,
            penalty_init=10,
            penalty_lr=0.05,
            seed=seed,
        )
        for seed in seeds
        for cost_lim in cost_lims
    ]
    for exp_kwargs in exps:
        ppo_lagrangian(**exp_kwargs)
