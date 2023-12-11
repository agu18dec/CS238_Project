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

    
#2So2 + 02 --> 2SO3 (reversible)

# molsSO2 = 2
# molsO2 = 1
# molsSO2Used = 0
# molsO2Used = 0
# SO2perCycle = 0.5
# O2perCycle = 0.5
# done = False
# SO2_material = material.SO2(mol=molsSO2) 
# O2_material = material.O2(mol=molsO2) 
# SO3_material = material.SO3(mol=0.000001)  
# reaction = Reaction(info)
# noSO2 = material.SO2(mol=0)
# noO2 = material.O2(mol=0)
# v0 = vessel.Vessel("Contact Process Vessel")
# v0.material_dict = {SO3_material._name: SO3_material, SO2_material._name: noSO2, O2_material._name: noO2}
# v1 = vessel.Vessel("SO2 Vessel")
# v1.material_dict = {SO2_material._name: SO2_material}
# v2 = vessel.Vessel("O2 Vessel")
# v2.material_dict = {O2_material._name: O2_material}
# v0.default_dt = 0.1
# print("Before Reaction:\n", v0.get_material_dataframe())
# #NOTE: THIS CURRENTLY DOESNT HANDLE THE CASE WHERE THE REACTANTS PER CYCLE DOESNT ADD UP TO THE TOTAL REACTANTS

# while not done:
#     productBefore = v0.material_dict[SO3_material._name].mol
#     if (molsSO2Used < molsSO2):
#         v0.material_dict[SO2_material._name].mol += SO2perCycle
#         molsSO2Used += SO2perCycle
#     if (molsO2Used < molsO2):
#         v0.material_dict[O2_material._name].mol += O2perCycle
#         molsO2Used += O2perCycle
#     reaction.update_concentrations(v0)
#     print("After Reaction:\n", v0.get_material_dataframe())
#     productAfter = v0.material_dict[SO3_material._name].mol
#     if (productAfter > productBefore - .0001 and productAfter < productBefore + .0001):
#         done = True


##############################################################################################################
# RETYPING TRYING TO USE GYMNASIUM
env = gym.make('ContactProcess')
print(env.reset())
# rgb = env.render()
print(f"PRODUCT VESSEL: {env.unwrapped.shelf[0].get_material_dataframe()}")
print(f"SO2 VESSEL: {env.unwrapped.shelf[1].get_material_dataframe()}")
print(f"O2 VESSEL: {env.unwrapped.shelf[2].get_material_dataframe()}")
print(f"Target Material: {env.unwrapped.target_material}")
total_reward = 0
action = np.ones((3))
print(action)
# molsSO2 = 2
# molsO2 = 1
# molsSO2Used = 0
# molsO2Used = 0
# SO2perCycle = 0.5
# O2perCycle = 0.5
done = False
# SO2_material = material.SO2(mol=molsSO2) 
# O2_material = material.O2(mol=molsO2) 
# SO3_material = material.SO3(mol=0.000001)  
# reaction = Reaction(info)
# noSO2 = material.SO2(mol=0)
# noO2 = material.O2(mol=0)
# v0 = vessel.Vessel("Contact Process Vessel")
# v0.material_dict = {SO3_material._name: SO3_material, SO2_material._name: noSO2, O2_material._name: noO2}
# v1 = vessel.Vessel("SO2 Vessel")
# v1.material_dict = {SO2_material._name: SO2_material}
# v2 = vessel.Vessel("O2 Vessel")
# v2.material_dict = {O2_material._name: O2_material}
# v0.default_dt = 0.1
# print("Before Reaction:\n", v0.get_material_dataframe())
#NOTE: THIS CURRENTLY DOESNT HANDLE THE CASE WHERE THE REACTANTS PER CYCLE DOESNT ADD UP TO THE TOTAL REACTANTS
# i = 0
while not done:
    o, r, done, _, info = env.step(action)
    total_reward += r
    print(f"PRODUCT VESSEL: {env.unwrapped.shelf[0].get_material_dataframe()}")
    print(f"SO2 VESSEL: {env.unwrapped.shelf[1].get_material_dataframe()}")
    print(f"O2 VESSEL: {env.unwrapped.shelf[2].get_material_dataframe()}")
    print("reward: ", r)
    print("total_reward: ", total_reward)
    # if i > 20:
    #     break
    # i += 1
    # productBefore = v0.material_dict[SO3_material._name].mol
    # if (molsSO2Used < molsSO2):
    #     v0.material_dict[SO2_material._name].mol += SO2perCycle
    #     molsSO2Used += SO2perCycle
    # if (molsO2Used < molsO2):
    #     v0.material_dict[O2_material._name].mol += O2perCycle
    #     molsO2Used += O2perCycle
    # reaction.update_concentrations(v0)
    # print("After Reaction:\n", v0.get_material_dataframe())
    # productAfter = v0.material_dict[SO3_material._name].mol
    # if (productAfter > productBefore - .0001 and productAfter < productBefore + .0001):
    #     done = True






