from torch.optim import Adam, AdamW, Adagrad, Adadelta, SGD
from pm.registry import OPTIMIZER

OPTIMIZER.register_module(name="Adam",force=False, module=Adam)
OPTIMIZER.register_module(name = "AdamW", force=False, module=AdamW)
OPTIMIZER.register_module(name = "Adagrad", force=False, module=Adagrad)
OPTIMIZER.register_module(name = "Adadelta", force=False, module=Adadelta)
OPTIMIZER.register_module(name = "SGD",force=False, module=SGD)