import unittest
from typing import List, Tuple
import torch
from torch.nn import functional as F
from src.model.core_classes.core_processer import BatchAssembly, BatchDisassembly, CoreSyncProcessor

### Batch Assembly testing ###

class MockBatchAssembly(BatchAssembly):
    """
    A mocked-up version of a batch assembly
    device. This one will expect to see 1d
    tensors in channels, and will pad those to a common
    length then return them, plus the unpadded length
    """
    def make_batch(self,
                   name: str,
                   cases: List[torch.Tensor]
                   ) ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Creates the batch. Expects a list of 1d tensors.
        Returns batch, and case.

        :param name: The name of the channel
        :param cases: The cases across the differing batches.
        :return: The batch, and the shapes
        """
        shape_length = [case.shape[0] for case in cases]
        max_length = max(shape_length)
        cases = [F.pad(case, (0, max_length - case.shape[0])) for case in cases]
        batch = torch.stack(cases, dim=0)
        shapes = torch.tensor(shape_length)
        return batch, shapes

class TestBatchAssembly(unittest.TestCase):
    """
    Test for batch assembly, unit tests
    """
    def create_environment(self):
