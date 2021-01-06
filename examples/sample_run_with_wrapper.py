import gym
import baxter_env
import pybullet as p
import pybullet_data
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt


class ObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(ObsWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(low = -1, high = 1, shape =  (128, 128, 4))
    def observation(self, obs):
        # Take image from the state dictionary and scale in range [-1, 1]
        img = obs['eye_view']/255.
        # Take the depth map from the state dictionary and scale (max depth is 10 and min 0)
        depth = obs['depth'].reshape(img.shape[0],img.shape[0],1)/10.
        obs = np.concatenate([img, depth], axis=-1)
        obs = np.clip(obs, -1,1)
        return obs.reshape((128, 128, 4))

class ActWrapper(gym.ActionWrapper):
    def __init__(self, env, action_verbose=100):
        self.action_steps = 0
        self.action_verbose = action_verbose
        super(ActWrapper, self).__init__(env)
        self.action_space = gym.spaces.Box(-1, 1, (6,))
    def action(self, act):        
        act = (act+1)*10
        act = np.clip(act,0,20)
        if self.action_steps % self.action_verbose == 0:
            print(f"actions took in action_step {self.action_steps} are: {act}")
        self.action_steps += 1
        return act

def make_env():
    env = gym.make('baxter_env-v0')
    env = ObsWrapper(env) 
    env = ActWrapper(env)
    return env


if __name__ == "__main__":
    env = make_env()
    s = env.reset()
    num_step = 0
    while True:
        action = env.sample_action()
        next_state, r, done, info = env.step(action)
        s = next_state
        print(f"Compleated step: {num_step}")
        print("Reward: ", r)
        print("Done: ", done)
        print("Info: ", info)
