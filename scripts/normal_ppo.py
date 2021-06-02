from safe_rl import ppo_lagrangian, ppo
import gym
import highway_env


#ppo_lagrangian(env_fn = lambda : gym.make("merge-v0"),ac_kwargs = dict(hidden_sizes=(64,64)))
ppo(env_fn = lambda : gym.make("merge-v0"),ac_kwargs = dict(hidden_sizes=(64,64)), logger_kwargs = {"output_dir": "./speed_0_30_penalty1"}, render=False, max_ep_len=70, epochs=150, steps_per_epoch=5000, penalty_init=1)