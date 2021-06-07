from safe_rl import ppo_lagrangian, ppo
import gym
import highway_env

penalty_init = 75
env_name = "mergefast"
ppo(env_fn=lambda: gym.make("{}-v0".format(env_name)),
    ac_kwargs=dict(hidden_sizes=(64, 64)),
    logger_kwargs={"output_dir": "./normal_ppo_Env{}_P{}_".format(env_name,penalty_init)},
    render=True,
    max_ep_len=100,
    epochs=150,
    steps_per_epoch=5000,
    penalty_init=penalty_init
    )
