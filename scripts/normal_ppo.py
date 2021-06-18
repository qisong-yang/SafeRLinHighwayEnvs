from safe_rl import ppo_lagrangian, ppo
import gym
import highway_env
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', nargs='*', type=int, default=123)
    parser.add_argument('--env_name', type=str, default="mergefast-v0")
    parser.add_argument('--penalty_init', type=float, default=100)
    parser.add_argument('--observation', type=str, default="LIST")

    args = parser.parse_args()

    seeds = args.seed if isinstance(args.seed, list) else [args.seed]
    exps = [
        dict(
            env_fn = lambda : gym.make(args.env_name, observation=args.observation),
            ac_kwargs = dict(hidden_sizes=(64,64)),
            logger_kwargs = {"output_dir": f"./results/ppo_normal_DS_dense_NOut_onlyCostValue_{args.env_name}_PI_{args.penalty_init}_SD_{args.seed}_OB_{args.observation}"},
            render=False,
            max_ep_len=100,
            epochs=150,
            steps_per_epoch=5000,
            penalty_init=args.penalty_init,
            seed=seed,
        )
        for seed in seeds
    ]
    for exp_kwargs in exps:
        ppo(**exp_kwargs)
