from .base import *
from .attn_bank_memories import BankMemoryConfig

random_test = BankMemoryConfig(32, 100, 10, 10)
layer = make_memory_unit(128, torch.float32, torch.device("cpu"), random_test)
layer = torch.jit.script(layer)