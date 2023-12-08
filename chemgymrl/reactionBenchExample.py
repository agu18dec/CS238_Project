import gymnasium as gym
import chemistrylab
import matplotlib,time
import numpy as np
from matplotlib import pyplot as plt
from chemistrylab.util import Visualization
from IPython.display import display,clear_output

Visualization.use_mpl_light(size=2)
# IF you are using dark mode


env = gym.make('GenWurtzReact-v2')
env.reset()
rgb = env.render()
plt.imshow(rgb)
plt.axis("off")
plt.show()


# NON-OPTIMIZED VERSION:

# d = False
# state = env.reset()
# total_reward = 0

# action = np.ones(env.action_space.shape[0])
# print(f'Target: {env.target_material}')
# for i, a in enumerate(env.actions):
#     v,event = env.shelf[a[0][0][0]],a[0][0][1]
#     action[i] = float(input(f'{v}: {event.name} -> {event.other_vessel}| '))


# while not d:
#     action = np.clip(action,0,1)
#     o, r, d, *_ = env.step(action)
#     total_reward += r
#     time.sleep(0.1)
#     clear_output(wait=True)
#     print(f'reward: {r}')
#     print(f'total_reward: {total_reward}')
#     rgb = env.render()
#     plt.imshow(rgb)
#     plt.axis("off")
#     plt.show()



# OPTIMIZED VERSION:
def predict(observation):
    t = np.argmax(observation[-7:])
    #targs = {0: "dodecane", 1: "5-methylundecane", 2: "4-ethyldecane", 3: "5,6-dimethyldecane", 4: "4-ethyl-5-methylnonane", 5: "4,5-diethyloctane", 6: "NaCl"}
    actions=np.array([
    [1,1,0,0,1],#dodecane
    [1,1,1,0,1],#5-methylundecane
    [1,1,0,1,1],#4-ethyldecane
    [1,0,1,0,1],#5,6-dimethyldecane
    [1,0,1,1,1],#4-ethyl-5-methylnonane
    [1,0,0,1,1],#4,5-diethyloctane
    [1,1,1,1,1],#NaCl
    ],dtype=np.float32)
    return actions[t]

d=False
o,*_=env.reset()
total_reward=0
while not d:
    action = predict(o)
    o, r, d, *_ = env.step(action)
    total_reward += r
    time.sleep(0.1)
    clear_output(wait=True)
    print(f"Target: {env.target_material}")
    print(f"Action: {action}")
    print(f'reward: {r}')
    print(f'total_reward: {total_reward}')
    rgb = env.render()
    plt.imshow(rgb)
    plt.axis("off")
    plt.show()
