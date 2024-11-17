import torch
import unittest
from torch.utils.data import Dataset, DataLoader


from src.main import arcAGI2024

class TestDistributedTraining(unittest.TestCase):
    """
    Test a simple distributed training case.
    """