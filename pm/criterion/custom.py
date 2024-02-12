from torch.nn import SmoothL1Loss, MSELoss
from pm.registry import CRITERION

CRITERION.register_module(name= "SmoothL1Loss", force=False, module=SmoothL1Loss)
CRITERION.register_module(name="MSELoss", force=False, module=MSELoss)