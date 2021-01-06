from gym.envs.registration import register

register(
    id='baxter_env-v0',
    entry_point='baxter_env.envs:BaxterEnv',
)