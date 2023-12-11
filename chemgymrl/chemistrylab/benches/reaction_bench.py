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
        self.numVessels = 4  # Number of vessels used in the env
        self.default_dt=0.01
        self.shelf = self.make_shelf()
        self.previous_shelf = None

       
        self.chems = ["SO3", "SO2", "O2"]
        self.target_material = "SO3"
        self.max_steps = 500
        self.currStep = 0
        
        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "SO3": spaces.Box(0,2,(4,), dtype=np.float32),
                "SO2": spaces.Box(0,2,(4,), dtype=np.float32),
                "O2": spaces.Box(0,2,(4,), dtype=np.float32),
            }
        )
        self.product = 0
        self.previous_product = 0
        self.action_space = spaces.Box(low=np.array([-500, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                                                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                                       high=np.array([500, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                                             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                                             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,]), 
                                       shape=(37,), dtype=np.float32) 
        
        print(self.action_space) #TODO: REMOVE THIS

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
    def _get_obs(self):
        obs = {"SO3": [], "SO2": [], "O2": []}
        for i in range(4):
            for chem in self.chems:
                obs[chem].append(self.shelf[i].material_dict[chem].mol)
        for chem in self.chems:
            obs[chem] = np.array(obs[chem], dtype=np.float32)
        return obs

    def reset(self, seed=None, options=None):
        self.product = 0
        self.shelf = self.make_shelf()
        self.previous_shelf = None
        self.previous_product = 0
        self.currStep = 0
        return (self._get_obs(), {})
    
    def default_reward(self, targ):
            return self.shelf[3].material_dict[targ].mol


    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        reward = 0
        reward += self.processAction(action) # Update the concentrations of each vessel + the heat of the rxn vessel
        self.reaction.update_concentrations(self.shelf[0])   # Update the concentrations of each vessel
        self.reaction.update_concentrations(self.shelf[1])   # Update the concentrations of each vessel
        self.reaction.update_concentrations(self.shelf[2])
        self.reaction.update_concentrations(self.shelf[3])
        self.product = self.shelf[0].material_dict["SO3"].mol
        terminated = False
        terminated = self.hasBeenNoChange()
        if (self.currStep > self.max_steps):
            terminated = True
        reward += self.default_reward(self.target_material)
        self.previous_product = self.product
        self.copyShelf()          
        observation = self._get_obs()
        self.currStep += 1
        return observation, reward, terminated, False, dict()
    
        #Check all vessels from previous iteration vs now. If there has been no change, return False
    def hasBeenNoChange(self):
        totalDiff = 0
        if self.previous_shelf != None:
            for i in range(4):
                for chem in self.chems:
                    prev = self.previous_shelf[i].material_dict[chem].mol
                    curr = self.shelf[i].material_dict[chem].mol
                    totalDiff += abs(prev - curr)
                    if totalDiff > 0.000002:
                        return False
        else:
            return False
        
        print("**No more change in the dataset detected**")
        return True
    
    def copyShelf(self):
        self.previous_shelf = self.make_shelf()
        for i in range(4):
            for chem in self.chems:
                self.previous_shelf[i].material_dict[chem].mol = self.shelf[i].material_dict[chem].mol

    def printData(self):
        print(f"REACTANT: {self.shelf[0].get_material_dataframe()}")
        print(f"SO2 VESSEL: {self.shelf[1].get_material_dataframe()}")
        print(f"O2 VESSEL: {self.shelf[2].get_material_dataframe()}")
        print(f"PRODUCT VESSEL: {self.shelf[3].get_material_dataframe()}")


    # reward is -100 if the amount is more than what is in the vessel. Reward is -.0001 if it moves materials due to work cost
    # Still need to make reward function for the temperature change
    def processAction(self, action):
        actionReward = 0

        self.shelf[0].temperature += action[0]
        #TODO: DECREMENT THE REWARD TO ACCOUNT FOR TEMP
        #return THE REWARD
        for i in range(3):
            chemical = self.chems[i]
            for j in range(12):
                value = action[i*12 + j + 1]
                if value:
                    if j == 0:
                        actionReward+= self.pour(0, 3, value, chemical)
                    elif j ==1:
                        actionReward+= self.pour(0, 1, value, chemical)
                    elif j == 2:
                        actionReward+= self.pour(0, 2, value, chemical)
                    elif j == 3:
                        actionReward+= self.pour(3, 0, value, chemical)
                    elif j == 4:
                        actionReward+= self.pour(3, 1, value, chemical)
                    elif j == 5:
                        actionReward+= self.pour(3, 2, value, chemical)
                    elif j == 6:
                        actionReward+= self.pour(1, 0, value, chemical)
                    elif j == 7:
                        actionReward+= self.pour(1, 3, value, chemical)
                    elif j == 8:
                        actionReward+= self.pour(1, 2, value, chemical)
                    elif j == 9:
                        actionReward+= self.pour(2, 0, value, chemical)
                    elif j == 10:
                        actionReward+= self.pour(2, 3, value, chemical)
                    elif j == 11:
                        actionReward+= self.pour(2, 1, value, chemical)
        return actionReward
    

    # Inits the 4 different vessels in a shelf
    def make_shelf(self):
        prodSO2 = material.SO2(mol=0)
        prodO2 = material.O2(mol=0)
        prodSO3 = material.SO3(mol=0)
        productVessel = vessel.Vessel("Product Vessel")
        productVessel.material_dict = {prodSO2._name: prodSO2, prodO2._name: prodO2, prodSO3._name: prodSO3}

        reactant1Vessel = vessel.Vessel("Reactant 1 Vessel")
        react1SO2 = material.SO2(mol=2)
        react1O2 = material.O2(mol=0)
        react1SO3 = material.SO3(mol=0)
        reactant1Vessel.material_dict = {react1SO2._name: react1SO2, react1O2._name: react1O2, react1SO3._name: react1SO3}

        react2Vessel = vessel.Vessel("Reactant 2 Vessel")
        react2SO2 = material.SO2(mol=0)
        react2O2 = material.O2(mol=1)
        react2SO3 = material.SO3(mol=0)
        react2Vessel.material_dict = {react2SO2._name: react2SO2, react2O2._name: react2O2, react2SO3._name: react2SO3}

        reactionVessel = vessel.Vessel("Reaction Vessel")
        reactSO2 = material.SO2(mol=0)
        reactO2 = material.O2(mol=0)
        reactSO3 = material.SO3(mol=0.000001)
        reactionVessel.material_dict = {reactSO2._name: reactSO2, reactO2._name: reactO2, reactSO3._name: reactSO3}
        newShelf = Shelf([ #Using the shelf to store things, but we will manually update the contents. 
        reactionVessel, 
        reactant1Vessel,
        react2Vessel,
        productVessel
        ])
        return newShelf
        


    def pour(self, source, dest, amount, chemical):
        # If the agent tries to move too much, move all you can and reward -100
        if self.shelf[source].material_dict[chemical].mol < amount:
            print(f"Transferring {self.shelf[source].material_dict[chemical].mol} of {chemical} from {self.shelf[source].label} to {self.shelf[dest].label}")
            self.shelf[dest].material_dict[chemical].mol += self.shelf[source].material_dict[chemical].mol
            self.shelf[source].material_dict[chemical].mol = 0
            return -100.0
        
        print(f"Transferring {amount} of {chemical} from {self.shelf[source].label} to {self.shelf[dest].label}")
        #else, move the requested amount and reward -.0001
        self.shelf[source].material_dict[chemical].mol -= amount
        self.shelf[dest].material_dict[chemical].mol += amount
        return -0.0001
