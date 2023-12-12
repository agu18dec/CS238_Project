from stable_baselines3 import DQN, PPO, A2C, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
#from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import gymnasium as gym
import sys
import chemistrylab
import numpy as np
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from datetime import datetime


env = gym.make("ContactProcess")
env.metadata["render_modes"] = None

env = make_vec_env("ContactProcess", n_envs=10)

model = A2C('MultiInputPolicy', env, verbose=1)
model.learn(total_timesteps=25000)
model.save("A2C_ContactProcess" + str(datetime.now()))

obs = env.reset()
env.verbose = True
stepNum = 0
for i in range(20):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    # print("Action: ", action[0], "Reward: ", rewards[0], "Dones: ", dones[0], "Info: ", info[0])
    # print("Observation: ", obs)
    # for key in obs.keys():
    #     print(key, obs[key][0])
    print("")