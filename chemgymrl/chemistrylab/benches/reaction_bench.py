# pylint: disable=invalid-name
# pylint: disable=unused-import
# pylint: disable=wrong-import-position

import os
import sys
import numpy as np
from chemistrylab.util.reward import RewardGenerator
from chemistrylab import material, vessel
from chemistrylab.benches.general_bench import *
from chemistrylab.reactions.reaction_info import ReactInfo, REACTION_PATH
from chemistrylab.lab.shelf import Shelf
import gymnasium as gym
from gymnasium import spaces

def get_mat(mat,amount,name=None):
    "Makes a Vessel with a single material"
    
    my_vessel = vessel.Vessel(
        label=f'{mat} Vessel' if name is None else name,
        ignore_layout=True
    )
    # create the material dictionary for the vessel
    matclass = material.REGISTRY[mat]()
    matclass.mol=amount
    material_dict = {mat:matclass}
    # instruct the vessel to update its material dictionary

    my_vessel.material_dict=material_dict
    my_vessel.validate_solvents()
    my_vessel.validate_solutes()
    my_vessel.default_dt=0.01
    
    return my_vessel

        
class GeneralWurtzReact_v2(GenBench):
    """
    Class to define an environment which performs a Wurtz extraction on materials in a vessel.
    """

    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 10,
    }
    def __init__(self):
        r_rew = RewardGenerator(use_purity=False,exclude_solvents=False,include_dissolved=False)
        shelf = Shelf([
            get_mat("diethyl ether",4,"Reaction Vessel"),
            get_mat("1-chlorohexane",1),
            get_mat("2-chlorohexane",1),
            get_mat("3-chlorohexane",1),
            get_mat("Na",3),
        ])
        actions = [
            Action([0],    [ContinuousParam(156,307,0,(500,))],  'heat contact',   [0],  0.01,  False),
            Action([1],    [ContinuousParam(0,1,1e-3,())],      'pour by percent',  [0],   0.01,   False),
            Action([2],    [ContinuousParam(0,1,1e-3,())],      'pour by percent',  [0],   0.01,   False),
            Action([3],    [ContinuousParam(0,1,1e-3,())],      'pour by percent',  [0],   0.01,   False),
            Action([4],    [ContinuousParam(0,1,1e-3,())],      'pour by percent',  [0],   0.01,   False),
        ]

        react_info = ReactInfo.from_json(REACTION_PATH+"/chloro_wurtz.json")
        
        super(GeneralWurtzReact_v2, self).__init__(
            shelf,
            actions,
            ["PVT","spectra","targets"],
            targets=react_info.PRODUCTS,
            default_events = (Event("react", (Reaction(react_info),), None),),
            reward_function=r_rew,
            discrete=False,
            max_steps=20
        )
        
class GeneralWurtzReact_v0(GenBench):
    """
    Class to define an environment which performs a Wurtz extraction on materials in a vessel.
    """
    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 10,
    }
    def __init__(self):
        r_rew = RewardGenerator(use_purity=False,exclude_solvents=False,include_dissolved=False)
        shelf = Shelf([
            get_mat("diethyl ether",4,"Reaction Vessel"),
            get_mat("1-chlorohexane",1),
            get_mat("2-chlorohexane",1),
            get_mat("3-chlorohexane",1),
            get_mat("Na",3),
        ])
        actions = [
            Action([0],    [ContinuousParam(156,307,0,(500,))],  'heat contact',     [0],  0.01,  False),
            Action([1],    [ContinuousParam(0,1,1e-3,())],      'pour by percent',  [0],   0.01,   False),
            Action([2],    [ContinuousParam(0,1,1e-3,())],      'pour by percent',  [0],   0.01,   False),
            Action([3],    [ContinuousParam(0,1,1e-3,())],      'pour by percent',  [0],   0.01,   False),
            Action([4],    [ContinuousParam(0,1,1e-3,())],      'pour by percent',  [0],   0.01,   False),
        ]

        react_info = ReactInfo.from_json(REACTION_PATH+"/chloro_wurtz.json")
        
        super(GeneralWurtzReact_v0, self).__init__(
            shelf,
            actions,
            ["PVT","spectra","targets"],
            targets=react_info.PRODUCTS[:-1],
            default_events = (Event("react", (Reaction(react_info),), None),),
            reward_function=r_rew,
            discrete=False,
            max_steps=20
        )

class ContactProcessReact_v0(GenBench):
    """
    Class to define an environment which performs the Contact Process reaction.
    """

    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 60,
    }
    def __init__(self):
        r_rew = RewardGenerator(use_purity=False, exclude_solvents=False, include_dissolved=False)
        v = get_mat("SO3", .000001,"Reaction Vessel") # very low initial amount to avoid divide by zero problems
        v.default_dt=0.0008
        # Setting up the shelf with SO2, O2, and a vessel for the reaction
        shelf = Shelf([
            v,   # Initially, SO3 is not present
            get_mat("SO2", 2),  # Assuming starting with 2 moles of SO2
            get_mat("O2", 1),    # Assuming starting with 1 mole of O2
        ])

        """
        Action parameters:
        vessels: Tuple[int]
        parameters: Tuple[tuple]
        event_name: str
        affected_vessels: Optional[Tuple[int]]
        dt: float
        terminal: bool
        """

        # Defining actions for the Contact Process
        actions = [
            Action([0],    [ContinuousParam(156,307,0,(500,))],  'heat contact',    [0],  1.0,  False),
            Action([1],    [ContinuousParam(0,1,1e-3,())],      'pour by percent',  [0],   1.0,   False), # Pour SO2
            Action([2],    [ContinuousParam(0,1,1e-3,())],      'pour by percent',  [0],   1.0,   False), # Pour O2
        ]

        targets = ["SO2"]
        # Reaction information for the Contact Process
        react_info = ReactInfo.from_json(REACTION_PATH+"/contact.json")
        # Initialize the bench with the shelf, actions, and reaction information
        super(ContactProcessReact_v0, self).__init__(
            shelf,
            actions,
            ["PVT", "spectra", "targets"],
            targets=targets,
            default_events=(Event("react", (Reaction(react_info),), None),),
            reward_function=r_rew,
            discrete=False,
            max_steps=500
        )
    def get_keys_to_action(self):
        # Control with the numpad or number keys.
        keys = dict()
        keys["1"] = np.array([1,0,0])
        keys["2"] = np.array([0,1,0])
        keys["3"] = np.array([0,0,1])
        keys["12"] = np.array([1,1,0])
        keys["13"] = np.array([1,0,1])
        keys["23"] = np.array([0,1,1])
        keys["123"] = np.array([1,1,1])
        return keys

class FictReact_v2(GenBench):
    """
    Class to define an environment which performs a Wurtz extraction on materials in a vessel.
    """

    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 10,
    }
    
    def __init__(self):
        r_rew = RewardGenerator(use_purity=False,exclude_solvents=False,
                                include_dissolved=False, exclude_mat = "fict_E")
        shelf = Shelf([
            get_mat("H2O",30,"Reaction Vessel"),
            get_mat("fict_A",1),
            get_mat("fict_B",1),
            get_mat("fict_C",1),
            get_mat("fict_D",3),
        ])

        actions = [
            Action([0],    [ContinuousParam(273,373,0,(300,))],  'heat contact',     [0],   0.01,   False),
            Action([1],    [ContinuousParam(0,1,1e-3,())],      'pour by percent',  [0],   0.01,   False),
            Action([2],    [ContinuousParam(0,1,1e-3,())],      'pour by percent',  [0],   0.01,   False),
            Action([3],    [ContinuousParam(0,1,1e-3,())],      'pour by percent',  [0],   0.01,   False),
            Action([4],    [ContinuousParam(0,1,1e-3,())],      'pour by percent',  [0],   0.01,   False),
        ]
        
        targets = ["fict_E", "fict_F", "fict_G", "fict_H", "fict_I"]
        react_info = ReactInfo.from_json(REACTION_PATH+"/fict_react.json")

        super(FictReact_v2, self).__init__(
            shelf,
            actions,
            ["PVT","spectra","targets"],
            targets=targets,
            default_events = (Event("react", (Reaction(react_info),), None),),
            reward_function=r_rew,
            discrete=False,
            max_steps=20
        )
        

class FictReactBandit_v0(GenBench):
    """
    Class to define an environment which performs a Wurtz extraction on materials in a vessel.
    """

    def __init__(self,targets=None):
        r_rew = RewardGenerator(use_purity=False,exclude_solvents=False,
                                include_dissolved=False, exclude_mat = "fict_E")
        shelf = Shelf([
            get_mat("H2O",30,"Reaction Vessel"),
            get_mat("fict_A",1),
            get_mat("fict_B",1),
            get_mat("fict_C",1),
            get_mat("fict_D",3),
        ])

        actions = [
            Action([0],    [ContinuousParam(273,373,0,(300,))],  'heat contact',     [0],   0.01,   False),
            Action([1],    [ContinuousParam(0,1,1e-3,())],      'pour by percent',  [0],   0.01,   False),
            Action([2],    [ContinuousParam(0,1,1e-3,())],      'pour by percent',  [0],   0.01,   False),
            Action([3],    [ContinuousParam(0,1,1e-3,())],      'pour by percent',  [0],   0.01,   False),
            Action([4],    [ContinuousParam(0,1,1e-3,())],      'pour by percent',  [0],   0.01,   False),
        ]
        if targets is None:
            targets = ["fict_E", "fict_F", "fict_G", "fict_H", "fict_I"]
        react_info = ReactInfo.from_json(REACTION_PATH+"/fict_react.json")

        super().__init__(
            shelf,
            actions,
            ["targets"],
            targets=targets,
            default_events = (Event("react", (Reaction(react_info),), None),),
            reward_function=r_rew,
            discrete=False,
            max_steps=20
        )
        self.action_space = gym.spaces.Box(0, 1, (self.n_actions+4,), dtype=np.float32)

    def step(self,action):
        action=np.array(action)
        uaction = action[:-4]
        gate = action[-4:]*self.max_steps-0.5
        ret=0
        d=False
        while not d:
            act = uaction*1
            act[1:]*= (gate<self.steps)
            o,r,d,*_ = super().step(act)
            gate[gate<self.steps-1]=self.max_steps
            ret+=r
        return (o,ret,d,*_)



class FictReactDemo_v0(GenBench):
    """
    Class to define an environment which performs a Wurtz extraction on materials in a vessel.
    """

    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 60,
    }

    def __init__(self):
        r_rew = RewardGenerator(use_purity=False,exclude_solvents=False,
                                include_dissolved=False, exclude_mat = "fict_E")

        v = get_mat("H2O",30,"Reaction Vessel")
        v.default_dt=0.0008
        shelf = Shelf([
            v,
            get_mat("fict_A",1),
            get_mat("fict_B",1),
            get_mat("fict_C",1),
            get_mat("fict_D",3),
        ])

        actions = [
            Action([0],    [ContinuousParam(273,373,0,(12,))],  'heat contact',     [0],  0,   False),
            Action([1],    [ContinuousParam(0,1,1e-3,())],      'pour by percent',  [0],  0,   False),
            Action([2],    [ContinuousParam(0,1,1e-3,())],      'pour by percent',  [0],  0,   False),
            Action([3],    [ContinuousParam(0,1,1e-3,())],      'pour by percent',  [0],  0,   False),
            Action([4],    [ContinuousParam(0,1,1e-3,())],      'pour by percent',  [0],  0,   False),
        ]
        
        targets = ["fict_E", "fict_F", "fict_G", "fict_H", "fict_I"]
        react_info = ReactInfo.from_json(REACTION_PATH+"/fict_react.json")

        super(FictReactDemo_v0, self).__init__(
            shelf,
            actions,
            ["PVT","spectra","targets"],
            targets=targets,
            default_events = (Event("react", (Reaction(react_info),), None),),
            reward_function=r_rew,
            discrete=False,
            max_steps=500
        )

    
        
    def get_keys_to_action(self):
        # Control with the numpad or number keys.
        keys = dict()
        for i in range(5):
            arr=np.zeros(5)
            arr[i]=1    
            keys[str(i+1)] = arr
        keys[()] = np.zeros(5)
        return keys

class Photosynthesis_v0(GenBench):
    """
    Class to define an environment which performs a Wurtz extraction on materials in a vessel.
    """

    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 10,
    }
    def __init__(self):
        r_rew = RewardGenerator(use_purity=False,exclude_solvents=False,include_dissolved=False)
        shelf = Shelf([
            get_mat("C6H12O6",.0001,"Reaction Vessel"),
            get_mat("H2O",6),
            get_mat("CO2",6),
        ])
        actions = [
            Action([0],    [ContinuousParam(156,307,0,(500,))],  'heat contact',   [0],  0.01,  False),
            Action([1],    [ContinuousParam(0,1,1e-3,())],      'pour by percent',  [0],   0.01,   False),
            Action([2],    [ContinuousParam(0,1,1e-3,())],      'pour by percent',  [0],   0.01,   False),
        ]

        react_info = ReactInfo.from_json(REACTION_PATH+"/photosynthesis.json")
        
        super(Photosynthesis_v0, self).__init__(
            shelf,
            actions,
            ["PVT","spectra","targets"],
            targets=["C6H12O6"],
            default_events = (Event("react", (Reaction(react_info),), None),),
            reward_function=r_rew,
            discrete=False,
            max_steps=20
        )

 ############################################################################################################       

class ContactProcess(gym.Env):
    """
    Class to define an environment which performs the Contact Process reaction.
    """

    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 60,
    }
    def __init__(self, render_mode=None, numVessels=3):
        info = ReactInfo.from_json(REACTION_PATH+"/contact.json")
        self.reaction = Reaction(info)
        self.numVessels = 3  # Number of vessels used in the env
        self.window_size = 512  # The size of the PyGame window. Sure
        # r_rew = RewardGenerator(use_purity=False, exclude_solvents=False, include_dissolved=False)
        self.default_dt=0.01
        # Setting up the shelf with SO2, O2, and a vessel for the reaction
        self.shelf = Shelf([ #Using the shelf to store things, but we will manually update the contents. 
        get_mat("SO3", .000001,"Reaction Vessel"),   # Initially, SO3 is not present
        get_mat("SO2", 2),  # Assuming starting with 2 moles of SO2
        get_mat("O2", 1),
                # Assuming starting with 1 mole of O2
        ])
        SO2_material = material.SO2(mol=0) 
        O2_material = material.O2(mol=0) 
        SO3_material = material.SO3(mol=0.000001)  
        noSO2 = material.SO2(mol=0)
        noO2 = material.O2(mol=0)
        self.shelf[0].material_dict = {SO3_material._name: SO3_material, SO2_material._name: noSO2, O2_material._name: noO2}
        self.target_material = "SO3"
        self.max_steps = 300
        self.currStep = 0
        
        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "SO3": spaces.Box(0,2,(1,0), dtype=np.float32),
                "SO2": spaces.Box(0,2,(1,0), dtype=np.float32),
                "O2": spaces.Box(0,2,(1,0), dtype=np.float32),
            }
        )
        self.product = 0
        self.previous_product = 0
        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(100* (self.numVessels - 1)**2) #400 for 3 vessels
        print(self.action_space) #TODO: REMOVE THIS

        self.actions = [
            Action([0],    [ContinuousParam(156,307,0,(500,))],  'heat contact',   [0],  0.01,  False),
            Action([1],    [ContinuousParam(0,1,1e-3,())],      'pour by percent',  [0],   0.01,   False),
            Action([2],    [ContinuousParam(0,1,1e-3,())],      'pour by percent',  [0],   0.01,   False),
        ]
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        ############  
    def reset(self, seed=None, options=None):
        self.product = 0
        self.shelf = Shelf([ #Using the shelf to store things, but we will manually update the contents. 
        get_mat("SO3", .000001,"Reaction Vessel"),   # Initially, SO3 is not present
        get_mat("SO2", 2),  # Assuming starting with 2 moles of SO2
        get_mat("O2", 1),
                # Assuming starting with 1 mole of O2
        ])
        SO2_material = material.SO2(mol=0) 
        O2_material = material.O2(mol=0) 
        SO3_material = material.SO3(mol=0.000001)  
        noSO2 = material.SO2(mol=0)
        noO2 = material.O2(mol=0)
        self.shelf[0].material_dict = {SO3_material._name: SO3_material, SO2_material._name: noSO2, O2_material._name: noO2}
        self.previous_product = 0
        return "HELLOOO"
    
    def default_reward(self, vessels,targ):
            sum_=0
            for vessel in vessels:
                mats=vessel.material_dict
                all_mat = sum(mats[a].mol for a in mats)
                if all_mat>1e-12:
                    sum_+=(mats[targ].mol if targ in mats else 0)**2/all_mat
            return sum_

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        for i in range(len(action)):
            if action[i] == 1:
                currAction = self.actions[i]
                if currAction.event_name == "heat contact":
                    self.shelf[0].temperature += 1 # NOTE: RIGHT NOW IM HARDCODING A 1 DEGREE TEMP INCREASE
                elif currAction.event_name == "pour by percent":
                    materialLeft = 0
                    if (i == 1):
                        materialLeft = self.shelf[i].material_dict["SO2"].mol
                        change = min(materialLeft, .1)
                        self.shelf[0].material_dict["SO2"].mol += change
                        self.shelf[1].material_dict["SO2"].mol -= change
                    elif i == 2:
                        materialLeft = self.shelf[i].material_dict["O2"].mol
                        change = min(materialLeft, .1)
                        self.shelf[0].material_dict["O2"].mol += change
                        self.shelf[2].material_dict["O2"].mol -= change

                    # if materialLeft > 0:
                    #     self.shelf[i]._pour_by_percent(self.default_dt, self.shelf[0], .1) # NOTE: RIGHT NOW IM POURING 10% at a time every time
        self.reaction.update_concentrations(self.shelf[0])   
        self.product = self.shelf[0].material_dict["SO3"].mol
        terminated = False
        if (self.product > self.previous_product - .0001 and self.product < self.previous_product + .0001 and self.shelf[1].material_dict["SO2"].mol < .0001 and self.shelf[2].material_dict["O2"].mol < .0001):
            terminated = True
        reward = self.default_reward(self.shelf,self.target_material) if terminated else 0
        self.previous_product = self.product
        # NOTE: CURRENTLY FOR MY REWARD, IM RETURNING -.1 FOR EACH ITERATION BECAUSE WE DON'T WANT IT TO BE REALLY SLOW            

        observation = self.shelf[0].material_dict
        self.currStep += 1
        return observation, reward, terminated, False, dict()
    
    def get_keys_to_action(self):
        # Control with the numpad or number keys.
        keys = dict()
        keys["1"] = np.array([1,0,0])
        keys["2"] = np.array([0,1,0])
        keys["3"] = np.array([0,0,1])
        keys["12"] = np.array([1,1,0])
        keys["13"] = np.array([1,0,1])
        keys["23"] = np.array([0,1,1])
        keys["123"] = np.array([1,1,1])
        return keys
