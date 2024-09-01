"""
The data module.

Concerns important details on how we augment our training data, split
the training data up into an eval and validation set, and similar important
related actions.

It also goes into how we prepare our data for consumption in the block multimodal
encoding format
"""

from typing import List, Dict, Any, Callable, Optional
import torch
from dataclasses import dataclass

@dataclass
class StateTracker:
    destination: str
    operation: str
    context: torch.Tensor
    inputs: List[Dict[str, Any]]
    outputs: List[Dict[str, Any]]
    callback: Callable
    timeout: float

    def update(self,
               destination: str,
               context: Optional[torch.Tensor] = None,
               inputs: List[Dict[str, Any]] = None,
               outputs: List[Dict[str, Any]] = None
               )->"StateTracker":
        """
        Updates the state of the statetracker. Returns a new instance
        :param destination:
        :param context:
        :param inputs:
        :param outputs:
        :return:
        """

        context = context if context is not None else self.context
        inputs = inputs if inputs is not None else self.inputs
        outputs = outputs if outputs is not None else self.outputs

        return StateTracker(
            destination,
            self.operation,
            context,
            inputs,
            outputs,
            self.callback,
            self.timeout
        )

@dataclass
class ActionRequest:
    state_tracker: StateTracker
    subtask_details: Dict[str, torch.Tensor]


