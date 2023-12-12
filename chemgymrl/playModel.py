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
# model = A2C.load('/Users/katie/Documents/JuniorFall/CS238_Final_Project/CS238_Project/A2C_ContactProcess2023-12-11')
env = make_vec_env("ContactProcess", n_envs=1)
# model = PPO.load('/Users/katie/Documents/JuniorFall/CS238_Final_Project/CS238_Project/FirstContactProcessModel.zip')
# model = PPO.load("/Users/katie/Documents/JuniorFall/CS238_Final_Project/CS238_Project/PPOContactProcessBestModel3kIters.zip")
# model = A2C.load("/Users/katie/Documents/JuniorFall/CS238_Final_Project/CS238_Project/ContactProcesA2C3kItersBestModel.zip")
model = A2C.load("/Users/katie/Documents/JuniorFall/CS238_Final_Project/CS238_Project/lastModelTrained.zip")
obs = env.reset()
stepNum = 0
done = False
env.verbose = True
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    done = dones[0]
    # print("Action: ", action[0], "Reward: ", rewards[0], "Dones: ", dones[0], "Info: ", info[0])
    # print("Observation: ", obs)
    # print("")
    #NOTE: IT IS EASIER TO JUST GO IN AND PRINT THINGS IN THE ENVIRONMENT ITSELF IN REACT_BENCH TO SEE WHATS HAPPENING IN ORDER
    # print("Action: ", action[0])
    # for key in obs.keys():
    #     print(key, obs[key][0])
    stepNum += 1
    print("Step Number: ", stepNum)