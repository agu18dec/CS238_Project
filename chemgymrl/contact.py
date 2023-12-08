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
from gym.envs.registration import register


name = "Contact Process"
REACTANTS = ["SO2", "O2"]
PRODUCTS = ["SO3"]
MATERIALS = ["SO2", "O2", "SO3"]

pre_exp_arr = np.array([55.34,1e-14])*1e7 #placeholder
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
    [2, -2]#[2, -2] #SO3
]).astype(np.float32)

info = ReactInfo(name,REACTANTS,PRODUCTS,[],MATERIALS,pre_exp_arr,activ_energy_arr,stoich_coeff_arr, conc_coeff_arr)

#2So2 + 02 --> 2SO3 (reversible)
reaction = Reaction(info)
v = vessel.Vessel("Contact Process Vessel")
SO2_material = material.SO2(mol=2)  # Starting with 2 moles of SO2
O2_material = material.O2(mol=1)   # Starting with 1 mole of O2
v.material_dict = {SO2_material._name: SO2_material, O2_material._name: O2_material}
v.default_dt = 0.1

print("Before Reaction:\n", v.get_material_dataframe())

reaction.update_concentrations(v)
print("After Reaction:\n", v.get_material_dataframe())
print("\n".join([ a for a in gym.registry.keys() if "React" in a or "Extract" in a or "Distill" in a]))

#put filepath accordingly
info.dump_to_json("/Users/agam/CS238/Final_Project/chemgymrl/chemistrylab/reactions/available_reactions/contact.json")
json_text = "".join(line for line in open("/Users/agam/CS238/Final_Project/chemgymrl/chemistrylab/reactions/available_reactions/contact.json","r"))
print(json_text)


env = gym.make('ContactProcessReact-v0')
env.reset()
rgb = env.render()
plt.imshow(rgb)
plt.axis("off")
plt.show()

# action space is a 3-vector. Continous Action Spcae
# Environmental Conditions -> discretized via 0 being decreased by dT or dV, 1/2 being no change, and 1 being increase in dT or dV
# Chemicals -> discretized via 0 being no reagents added, 1 being all added, with negative reward if more than available is added
d = False
state = env.reset()
total_reward = 0

action = np.ones(env.action_space.shape[0])
print(f'Target: {env.target_material}')
for i, a in enumerate(env.actions):
    v,event = env.shelf[a[0][0][0]],a[0][0][1]
    action[i] = float(input(f'{v}: {event.name} -> {event.other_vessel}| '))


while not d:
    action = np.clip(action,0,1)
    o, r, d, *_ = env.step(action)
    total_reward += r
    time.sleep(0.1)
    clear_output(wait=True)
    print(f'reward: {r}')
    print(f'total_reward: {total_reward}')
    rgb = env.render()
    plt.imshow(rgb)
    plt.axis("off")
    plt.show()

