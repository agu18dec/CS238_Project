from chemistrylab.reactions.reaction_info import ReactInfo
from chemistrylab.reactions.reaction import Reaction
from chemistrylab import material,vessel
from chemistrylab.benches import *
import gymnasium as gym
import chemistrylab
import matplotlib,time
import numpy as np
from matplotlib import pyplot as plt
from chemistrylab.util import Visualization

import numpy as np
from IPython.display import display,clear_output,JSON
# from gym.envs.registration import register


name = "Contact Process"
REACTANTS = ["SO2", "O2"]
PRODUCTS = ["SO3"]
MATERIALS = ["SO2", "O2", "SO3"]

pre_exp_arr = np.array([55.34, 1e-14])*1e7 #placeholder
activ_energy_arr = np.array([1.0,1.0]) #placeholder

#of the form [reaction, reactants] setting stoichiometric coefficients
stoich_coeff_arr = np.array([
    [2, 1, 0],  # 2SO2 + O2 -> 2SO3
    [0, 0, 2]
]).astype(np.float32)

#represents changes in concentration. Shape is [materials, reactions]
conc_coeff_arr = np.array([
    [-2, 2], #SO2
    [-1, 1], #O2
    [2, -2] #SO3
]).astype(np.float32)


info = ReactInfo(name,REACTANTS,PRODUCTS,[],MATERIALS,pre_exp_arr,activ_energy_arr,stoich_coeff_arr, conc_coeff_arr)
#put filepath accordingly
info.dump_to_json("chemgymrl/chemistrylab/reactions/available_reactions/contact.json")
json_text = "".join(line for line in open("chemgymrl/chemistrylab/reactions/available_reactions/contact.json","r"))

##############################################################################################################
# TRYING TO USE GYMNASIUM

def printData(env):
    print(f"REACTANT: {env.unwrapped.shelf[0].get_material_dataframe()}")
    print(f"SO2 VESSEL: {env.unwrapped.shelf[1].get_material_dataframe()}")
    print(f"O2 VESSEL: {env.unwrapped.shelf[2].get_material_dataframe()}")
    print(f"PRODUCT VESSEL: {env.unwrapped.shelf[3].get_material_dataframe()}")

env = gym.make('ContactProcess')
env.reset()
printData(env)
print(f"Target Material: {env.unwrapped.target_material}")
total_reward = 0
action = np.zeros((37)) #[heat, 12 different options of moving contents from one vessel to another, then material to be moved]
# action = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1])
action[34] = .1 # making the action be to move the contents of the O2 vessel to the rxn vessel
action[19] = .2 # making the action be to move the contents of the SO2 vessel to the rxn vessel
# action = np.zeros((37)) 
#NOTE: moving contents from the SO2 vessel to the rxn vessel is 
print(action)
done = False
i = 0
while not done:
    o, r, done, _, info = env.step(action)
    total_reward += r
    printData(env)
    print("reward: ", r)
    print("total_reward: ", total_reward)
    # if (i == 1):
    #     action = np.zeros((37))

    # if i > 10:
    #     break
    i+=1 
print(f"Function ended on iteration {i}")






