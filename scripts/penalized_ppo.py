from safe_rl import ppo_lagrangian, ppo
import gym
import highway_env
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', nargs='*', type=int, default=123)
    parser.add_argument('--env_name', type=str, default="mergefast-v0")
    parser.add_argument('--cost_lim', type=float, default=0.01)
    parser.add_argument('--penalty_init', type=float, default=10)
    parser.add_argument('--penalty_lr', type=float, default=0.05)
    parser.add_argument('--penalty_iters', type=int, default=40)
    parser.add_argument('--observation', type=str, default="LIST")

    args = parser.parse_args()

    seeds = args.seed if isinstance(args.seed, list) else [args.seed]
    exps = [
        dict(
            env_fn = lambda : gym.make(args.env_name, observation=args.observation),
            ac_kwargs = dict(hidden_sizes=(64,64)),
            logger_kwargs = {"output_dir": f"./results/ppo_lag_{args.env_name}_cost_lim_{args.cost_lim}_penalty_init_{args.penalty_init}_penalty_lr_{args.penalty_lr}_penalty_iters_{args.penalty_iters}_seed_{seed}"},
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
        for seed in seeds
    ]
    for exp_kwargs in exps:
        ppo_lagrangian(**exp_kwargs)
