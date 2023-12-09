import os
import numpy as np
import gymnasium as gym
import chemistrylab
import time
from chemistrylab.util import Visualization,ActionDoc


if __name__ == "__main__":
    from gymnasium.utils.play import play
    # Visualization.set_backend("numba") 
    Visualization.set_backend("pygame")

    Visualization.matplotVisualizer.legend_update_delay=1

    #Visualization.use_mpl_dark()

    while True:
        print("Enter 0 for Extraction, 1 For Our Contact Process, or 2 for Reaction")
        x = input()
        if x in ["0","1","2","3"]:
            break
    x = int(x)
    env_id = ["WurtzExtractDemo-v0","ContactProcessReact-v0","FictReactDemo-v0","ExtractTest-v0"][x]

    env=gym.make(env_id)
    env.metadata["render_modes"] = ["rgb_array"]
    _=env.reset()



    if x==2:
        keys = dict()
        for i in range(5):
            arr=np.zeros(5)
            arr[i]=1    
            keys[str(i+1)] = arr
    if x==1:
        keys = dict()
        keys["1"] = np.array([1.0,0.0,0.0])
        keys["2"] = np.array([0.0,1.0,0.0])
        keys["3"] = np.array([0.0,0.0,1.0])
        keys["12"] = np.array([1.0,1.0,0.0])
        keys["13"] = np.array([1.0,0.0,1.0])
        keys["23"] = np.array([0.0,1.0,1.0])
        keys["123"] = np.array([1.0,1.0,1.0])
    else:
        keys = {k:i for i,k in enumerate(["1234567890","123456","0","12345qwert890"][x])}


    noop = [7,np.zeros(3),np.zeros(5),10][x]

    print("Actions: (Use the number keys)")

    table = ActionDoc.generate_table(env.shelf,env.actions)
    print(table)

    input("Press Enter to Start")

    ret=0
    def callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
        global ret
        ret+=rew
        if terminated:
            os.system('cls' if os.name=='nt' else 'clear')
            print(ret)
            ret=0
        print(sum(action))
        if sum(action) <.1 : #if the action vector is all 0s, so we havent pressed a key
            print("Action: ",action)
            print(env.unwrapped.shelf[0].get_material_dataframe())
        return [rew]
        
    play(env,keys_to_action=keys,noop=noop,fps=60,callback=callback)