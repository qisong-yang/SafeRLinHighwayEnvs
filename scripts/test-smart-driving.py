from safe_rl import ppo_lagrangian
import gym
import highway_env


ppo_lagrangian(env_fn = lambda : gym.make("merge-v0"),ac_kwargs = dict(hidden_sizes=(64,64)))