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




# info = ReactInfo(name,REACTANTS,PRODUCTS,[],MATERIALS,pre_exp_arr,activ_energy_arr,stoich_coeff_arr, conc_coeff_arr)


# info = ReactInfo.from_json("photosynthesis.json")
# reaction = Reaction(info)


env = gym.make('Photosynthesis-v0')
env.reset()
rgb = env.render()
plt.imshow(rgb)
plt.axis("off")
#plt.show()

# action space is a 3-vector. Continous Action Spcae
# Environmental Conditions -> discretized via 0 being decreased by dT or dV, 1/2 being no change, and 1 being increase in dT or dV
# Chemicals -> discretized via 0 being no reagents added, 1 being all added, with negative reward if more than available is added
d = False
state = env.reset()
print(state)
total_reward = 0
action = np.ones(env.unwrapped.action_space.shape[0])
print(f'Target: {env.unwrapped.target_material}')
for i, a in enumerate(env.unwrapped.actions):
    # print("Action data (i, a): ", i, a)

    # Extract vessel and event for this action
    v,event = env.unwrapped.shelf[a[0][0][0]],a[0][0][1]
    # print(v, event, a[1])
    action[i] = float(input(f'{v}: {event.name} -> {event.other_vessel}| '))

print(action)
while not d:
    action = np.clip(action,0,1)
    # env.unwrapped.shelf
    print(env.unwrapped.shelf[0].get_material_dataframe())
    print(env.unwrapped.shelf[1].get_material_dataframe())
    #0.832724
    o, r, d, *_ = env.step(action)
    total_reward += r
    time.sleep(0.1)
    # clear_output(wait=True)
    print(f'reward: {r}')
    print(f'total_reward: {total_reward}')
    # rgb = env.render()
    # plt.imshow(rgb)
    # plt.axis("off")
    # plt.show()
    

