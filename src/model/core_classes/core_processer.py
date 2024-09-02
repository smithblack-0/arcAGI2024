import textwrap
from abc import ABC, abstractmethod
import torch
from typing import Dict, Tuple, List
from torch import nn
from .types import (
    LoggingCallback,
    TerminationCallback,
    DataCaseBuffer,
    ExceptionAugmentedResponse,
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

    The batch assembly mechanism is generally expected to be implemented in a manner that
    has to do with the spacial combination of data - for instance, different varieties might
    be used to combine images while preserving dimensions vs flatten images for usage in
    a transformer. It is responsible for padding and providing masks.

    In order for a valid batch assembly case to be created, the make_batch function must
    be implemented. This function accepts tensors from all channels of all batches that
    are being collected together, and must return two things. The first is the combined
    batch. The second is, per batch dimension, a 2d shape tensor specifying the extend of
    the nonbatched content in that dimension.

    The content that is sent on to the model will depend somewhat on the parameters provided
    on initialization.
    """

    @abstractmethod
    def make_batch(self,
                   name: str,
                   cases: List[torch.Tensor]
                   )->Tuple[torch.Tensor, torch.Tensor]:
        """
        The user-defined function requiring implementation.
        The function requiring implementation.

        This function does the majority of the work. It should examine
        the tensors which have been collected across the differing cases,
        and figure out how to pad and combine them into a common batch.

        :param name: The name of the feature we are batching together.
        :param cases: The tensors found across all the different cases.
        :return:
            - batch: The batch that has been constructed. It had better have the same length as cases,
                     or it will throw an error.
            - nonpadding_shapes:
                a 2d int tensor containing information on the extend of the nonpadding content in the batch.
                For instance, a tensor containing [[2, 3],[4,6]] might represent a batch that contains a first image
                of shape [2, 3], and a second image of shape [4, 6].
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
                 cases_buffer: DataCaseBuffer,
                 logging_callback: LoggingCallback,
                 termination_callback: TerminationCallback,
                 )->Tuple[List[str],
                          Dict[str, Exception],
                          Dict[str, torch.Tensor],
                          Dict[str, torch.Tensor]
                          ]:
        """
        Performs the action of actually assembling a batch. This
        includes flushing them from the buffer and putting them all
        together

        :param selected_cases: The cases to select out of the buffer
        :param cases_buffer: The cases buffer
        :param logging_callback: The logging callback
        :return:
            - Nonexception_uuid: A list of AssemblyMetadataEntry objects. These will later be
                        consumed when disassembling the batch.
            - Exception_info: uuids which had an exception associated, and the exception
            - Batch: A dictionary of channel to assembled batches
            - Nonpadded_Shapes: A dictionary of channel to nonpadded shape specification.
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
                logging_callback(exception, 1)
                batch_failed_metadata[key] = exception
                continue

            logging_callback(f"Popping '{key}' out of case buffer'", 3)
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
                logging_callback(exception, 1)
                batch_failed_metadata[key] = exception
                continue

            batch_cases.append(case)
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
        batches = {}
        shapes = {}
        for name, tensors in data.items():
            logging_callback(f"Making batch out of features {name}", 3)
            batch, shape = self.make_batch(name, tensors)
            if batch.shape[0] != len(tensors) or shape.shape[0] != len(tensors):
                msg = f"""
                A terminal error has been encountered
                
                An issue was detected with the implemention of the make batch callback.
                Either the returned batch tensor did not have the correct batch length,
                or the returned shape tensor did not have the correct batch length.
                
                This kind of issue is likely not associated with a particular case, and 
                thus cannot be recovered from.
                
                The expected batch shape was: {len(tensors)}
                The constructed batch dim was: {batch.shape[0]}
                The constructed shape batch dim was: {shape.shape[0]}
                """
                msg = textwrap.dedent(msg)
                exception = RuntimeError(msg)
                logging_callback(exception, 0)
                termination_callback(True)
                raise exception

            batches[name] = batch
            shapes[name] = shape

        return batch_metadata, batch_failed_metadata, batches, shapes



class BatchDisassembly:
    """
    The batch disassembly class has three primary responsibilities, but
    they all boil down to working to produce a single datastructure.

    One of the responsibilities is to split up the results produced
    by the machine learning model and reassociate each of the batch
    dimensions with a uuid counterpart. We keep whatever dict channels were
    provided by the model.

    Another responsibility is to break up the shape
    padding details and again associate them back with uuids. This information
    may be used to remove excess padding in downstream layers.

    Finally, there is exception handling. If an exception has been handled, and we think
    it will not hose the whole generation process, we can insert an exception
    at that location for the uuid output. It will then be propogated back
    into an associated future.

    The result of the class being run is something in which all batch information has
    been removed.
    """
    def __call__(self,
                 run_uuids: List[str],
                 exception_data: Dict[str, Exception],
                 shapes_data: Dict[str, torch.Tensor],
                 model_data: Dict[str, torch.Tensor],
                 logging_callback: LoggingCallback,
                 termination_callback: TerminationCallback,
                 )->Dict[str, ExceptionAugmentedResponse]:
        """
        Runs the batch dissassembly process. One detail

        :param run_uuids: The uuids associated with the batches that were run through the model
        :param exception_data: The uuids associated with batches that had exceptions detected.
        :param shapes_data: The mapping of dict channel to the unpadded shape.
        :param model_data:
            The mapping of dict channel to the results of running the model.
            Note: Likely different from shapes channels.
        :param logging_callback:
            The logging callback. Used to make certain decisions.
        :return:
            - uuid channel responses:
                An exception augmented dictionary mapping uuids to relevant information
                for that particular batch. This generally means either:
                    1) A tuple of dictionaries, each mapping over channels. One of them
                       contains the input padding shapes, one of them the output response.
                    2) An exception. In which case the batch could not be properly processed
        """

        # Take apart the padding shape data and reassociate each batch dimension
        # with a data case based on the run_uuids

        shapes_dict = {id : {} for id in run_uuids}
        for channel, tensor in shapes_data.items():
            if tensor.shape[0] != len(run_uuids):
                msg = f"""
                Terminal error encountered
                
                Something has gone wrong with batch_dim-batch_uuid matching. 
                The number of selected uuids to run was {len(run_uuids)}. 
                However, the batch dimension has a shape of {tensor.shape[0]}
                
                Since we no longer know what uuid goes to what batch, this is unrecoverable.
                Shutting down processing.
                """
                msg = textwrap.dedent(msg)
                exception = RuntimeError(msg)
                logging_callback(exception, 0)
                termination_callback(True)
                raise exception
            for id, subtensors in zip(run_uuids, tensor.unbind(0)):
                shapes_dict[id][channel] = subtensors


        ## Take apart the response produced by invoking the model,
        # and associated each element back with it's uuid case.

        response_cases = {id : {} for id in run_uuids}
        for key, tensor in model_data.items():
            if tensor.shape[0] == len(run_uuids):
                msg = f"""
                Unrecoverable error encountered. It was expected that the tensors returned by
                the model would have the same batch shape as what went in. For feature of name 
                '{key}' this was not the case. 
                
                expected_shape: {len(run_uuids)}
                seen_shape: {tensor.shape[0]}
                """
                exception = RuntimeError(msg)
                logging_callback(exception, 0)
                termination_callback(True)
                raise exception
            for uuid, subtensor in zip(run_uuids, tensor.unbind(0)):
                response_cases[uuid][key] = subtensor
        ## Merge the dictionaries.
        #
        # Then, insert the exceptions for the cases we could
        # not successfully process.

        output = {id: (shapes_dict[id], response_cases[id]) for id in run_uuids}
        for key, exception in exception_data.items():
            output[key] = exception
        return output

## Model core.
#
# Contract which compatible models must satisfy
class ContractedModule(nn.Module):
    """
    Any torch module which wants to be compatible
    with this system MUST implement its forward method
    in the way contracted here.
    """
    def forward(self,
                padding_shapes: Dict[str, torch.Tensor],
                input_data: Dict[str, torch.Tensor],
                logging_callback: LoggingCallback,
                )->Dict[str, torch.Tensor]:
        """

        :param padding_shapes: Shapes padding data, per channel. 2d tensor indicating
                       how much of each dimension in data was NOT padding,
                       You may or may not use this.
        :param input_data:
            The actual batched data. Channels will be common with padding shapes. You
        :return: The responses, per dictionary channel. May have differing
                 channels than input data.
        """
        raise NotImplementedError("The forward method needs to be implemented with the specified contract")

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
                 core_model: ContractedModule,
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
                case_buffer: DataCaseBuffer,
                logging_callback: LoggingCallback,
                termination_callback: TerminationCallback,
                )->Dict[str, ExceptionAugmentedResponse]:
        """
        :param selected_cases:
            A list of string-based uuids. Each should uniquely identify a case that has been selected
            for inclusion as part of a batch
        :param case_buffer:
            The case buffer. This should contain within it a list of groups of tensor cases
            such as shapes, context, and
        :return: A dictionary of uuids to nonbatched info.
        """

        used_ids, exception_data, batch, shapes = self.batch_assembler(selected_cases,
                                                                       case_buffer,
                                                                       logging_callback,
                                                                       termination_callback)


        try:
            model_output = self.core_model(shapes, batch, logging_callback)
        except Exception as err:
            # Setup exception
            msg = f"""
            An issue was encountered while trying to use a neural core to process a batch.
            The issue could not be tracked down to a single batch. The entire batch
            is being discarded.
            
            This issue affected data entries:
            
            {selected_cases}
            
            It occurred while using torch layer:
            """
            msg = textwrap.dedent(msg)
            msg = msg + str(self.core_model)
            exception = RuntimeError(msg)
            exception.__cause__ = err

            # Log exception
            logging_callback(exception, 1)

            # Remove any attempt the model
            used_ids = []
            exception_data.update({id : exception for id in selected_cases})

        output = self.batch_disassembler(used_ids, exception_data, shapes,
                                         model_output, logging_callback, termination_callback)
        return output

