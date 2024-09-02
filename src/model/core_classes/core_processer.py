import textwrap
from abc import ABC, abstractmethod
import torch
from typing import Dict, Tuple, List, Optional
from torch import nn
from .types import (
    LoggingCallback,
    BatchCaseBuffer,
    BatchEntry
)
from dataclasses import dataclass
from ..config import Config

##
# Batch assembly and disassembly classes.
#
# These actually complete the job of creating a batch, or taking
# batched content and separating it back into individual pieces.

class BatchAssembly(ABC):
    """
    The BatchAssembly class is an abstract class whose interface is being defined.

    Its responsibility is solely to take a collection of unbatched tensors and assemble
    them into batched tensors in a sane manner. This will usually mean needing to complete
    some level of padding.

    It expects to be provided with a list of the selected cases to build the
    batch out of, and the batch cases buffer. It gets the cases out of the buffer,
    runs the user-provided merge routines
    """

    @abstractmethod
    def make_batch(self,
                   name: str,
                   cases: List[torch.Tensor]
                   )->Tuple[torch.Tensor, torch.Tensor]:
        """
        The function requiring implementation.

        This function does the majority of the work. It should examine
        the tensors which have been collected across the differing cases,
        and figure out how to pad and combine them into a common batch.

        :param name: The name of the feature we are batching together.
        :param cases: The tensors found across all the different cases.
        :return:
            - batch: The batch that has been constructed. It better have batch length of cases
            - nonpadding_mask: A mask indicating what elements are NOT padding. It should be a bool
              tensor with the same shape as the batch
        """
        pass

    def __init__(self,
                 channel_names: List[str]
                 ):
        """

        :param channel_names:
                The names of the channels that should be active that we will see
                in the dictionaries we will be passed. Anything missing these
                channels will NOT be processed.

        """
        super().__init__()
        self.channel_names = channel_names
    def __call__(self,
                 selected_cases: List[str],
                 cases_buffer: BatchCaseBuffer,
                 logging_callback: LoggingCallback
                 )->Tuple[List[str],
                          Dict[str, Exception],
                          Dict[str, BatchEntry]]:
        """
        Performs the action of actually assembling a batch. This
        includes flushing them from the buffer and putting them all
        together

        :param selected_cases: The cases to select out of the buffer
        :param cases_buffer: The cases buffer
        :param logging_callback: The logging callback
        :return:
            - Metadata: A list of AssemblyMetadataEntry objects. These will later be
                        consumed when disassembling the batch.
            - Batch: A dictionary of str to a tensor, mask tuple
        :effect: Deletes the selected cases from the case buffer
        """

        # Get the cases out of the case buffer. This also
        # will modify the case buffer.
        batch_cases = []
        batch_metadata = []
        batch_failed_metadata = {}
        for key in selected_cases:
            if key not in cases_buffer:
                msg = f"""
                Warning! Uuid '{key}' was not found in cases buffer, despite being selected
                for extraction. This might be a sign of a race condition. It is definitely
                a bug. Ignoring this case.
                """
                msg = textwrap.dedent(msg)
                exception = KeyError(msg)
                logging_callback(exception, 0)
                batch_failed_metadata[key] = exception
                continue

            logging_callback(f"Popping '{key}' out of case buffer'", 4)
            case = cases_buffer.pop(key)
            if set(case.keys()) != set(self.channel_names):
                msg = f"""
                Warning! uuid case '{key}' was expected to be a dictionary with
                keys {self.channel_names}. 
                However, actually got {case.keys()}
                This can be a sign of mixing evaluation and training pipelines. 
                Ignoring '{key}'
                """
                msg = textwrap.dedent(msg)
                exception = KeyError(msg)
                logging_callback(exception, 0)
                batch_failed_metadata[key] = exception
                continue

            batch_cases.append(cases_buffer.pop(key))
            batch_metadata.append(key)

        ##
        # Go extract the data for each channel all into lists
        ##
        data = {key : [] for key in self.channel_names}
        for key in self.channel_names:
            for i, batch_case in batch_cases:
                data[key].append(batch_case[key])

        ###
        # Process each of the lists. We end up with the batch
        ###
        batch = {}
        for name, tensors in data.items():
            logging_callback(f"Making batch out of features {name}", 4)
            batch[name] = self.make_batch(name, tensors)

        return batch_metadata, batch_failed_metadata, batch



class BatchDisassembly:
    """
    The BatchDissassebly class is designed with the primary
    responsibility of splitting up batched data back into
    unbatched data then dispatching that data to the destination
    callbacks.

    It takes a dictionary containing batched tensors that are the
    result of running a model, then takes those dictionaries apart
    into individual subcases. These can then be dispatched back
    through their callbacks.
    """
    def __call__(self,
                 original_uuids: List[str],
                 exception_data: Dict[str, Exception],
                 model_data: Dict[str, torch.Tensor],
                 logging_callback: LoggingCallback
                 )->Dict[str, Dict[str, torch.Tensor] | Exception]:

        # Take apart the incoming data by the batch dimensions
        #
        # Create batch independent cases
        cases = {id : {} for id in original_uuids}
        for key, tensor in model_data.items():
            if tensor.shape[0] == len(original_uuids):
                msg = f"""
                Unrecoverable error encountered. It was expected that the tensors returned by
                the model would have the same batch shape going in. For feature of name 
                '{key}' this was not the case. 
                """
                exception = RuntimeError(msg)
                logging_callback(exception, 0)
                raise exception
            for uuid, subtensor in zip(original_uuids, tensor.unbind(0)):
                cases[uuid][key] = subtensor

        ##
        # Insert per-case exceptions data into the stream
        ##

        for key, exception in exception_data.items():
            cases[key] = exception
        return cases

##
#
# Core syncronous processor
#
# This section does the neural heavy lifting,
# and also deletes expired entries from
#

class CoreSyncProcessor(nn.Module):
    """
    Core processor for the async processing mechanism.
    This is entirely synchronous, a fact that aids
    greatly while debugging.

    This class performs the extraction, batching,
    machine learning processing, and disassembly
    of targets into and from batches
    """
    def __init__(self,
                 batch_assembler: BatchAssembly,
                 core_model: nn.Module,
                 batch_disassembler: BatchDisassembly,
                 ):
        """

        :param batch_assembler:
            A mechanism to assemble a batch out of the target ids and the batches lying in the buffer.
        :param core_model:
            A core machine learning model to process the batch with. Should expect and return dictionaries
        :param batch_disassembler:
            A mechanism that retunrs
        """
        super().__init__()
        self.batch_assembler = batch_assembler
        self.core_model = core_model
        self.batch_disassembler = batch_disassembler

    def forward(self,
                selected_cases: List[str],
                case_buffer: BatchCaseBuffer,
                logging_callback: LoggingCallback
                )->Dict[str, Dict[str, torch.Tensor] | Exception]:
        """
        :param selected_cases:
            A list of string-based uuids. Each should uniquely identify a case that has been selected
            for inclusion as part of a batch
        :param case_buffer:
            The case buffer. This should contain within it a list of groups of tensor cases
            such as shapes, context, and
        :return:
        """
        used_ids, exception_data, batch = self.batch_assembler(selected_cases,
                                                               case_buffer,
                                                               logging_callback
                                                               )
        try:
            batch = self.core_model(batch)
        except Exception as err:
            # Setup exception
            msg = f"""
            Issue encountered across multiple uuids while attempting to run batch using core
            model. This affected cases:
            {selected_cases}
            The exact batch that caused this issue, if any, is not known.
            """
            msg = textwrap.dedent(msg)
            exception = RuntimeError(msg)
            exception.__cause__ = err

            # Log exception, modify outputs so we will except on all entries
            logging_callback(exception, 0)
            used_ids = []
            exception_data = {id : exception for id in selected_cases}

        output = self.batch_disassembler(selected_cases, exception_data, batch)
        return output

