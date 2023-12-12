import stable_baselines3
from stable_baselines3.common.monitor import Monitor
import torch



policy = "/Users/katie/Documents/JuniorFall/CS238_Final_Project/CS238_Project/FirstPPOOutputBest100/policy.pth"
policyOptimizer = "/Users/katie/Documents/JuniorFall/CS238_Final_Project/CS238_Project/FirstPPOOutputBest100/policy.optimizer.pth"
pytorchVars = "/Users/katie/Documents/JuniorFall/CS238_Final_Project/CS238_Project/FirstPPOOutputBest100/pytorch_variables.pth"
policyData = torch.load(policy)
policyOptimizerData = torch.load(policyOptimizer)
pytorchVarsData = torch.load(pytorchVars)
# Load the monitor file

# results = stable_baselines3.common.monitor.load_results(directory)
print("POLICY")
print(policyData)
print("POLICY OPTIMIZER")
print(policyOptimizerData)
print("PYTORCH VARS")
print(pytorchVarsData)