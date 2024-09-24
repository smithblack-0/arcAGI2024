"""
A loader, and also saving, mechanism. The data converter module contains things designed to "tokenize"
or "detokenize" the multimodal content which is being presented. Content is transformed in such a way
that downstream it is straightforward to recover, for each piece of data being processed, what the context
is based on the metadata entries on each tensor


"""
from .types import Block
class MetadataManager:
    """

    """

    def __init__(self,
                 control_spec:
                 ):
    def create_metadata(self,
                        mode:


                        ):

        output = []
        output_append(control.tensor("start"))

class BlockModeEncoder:
    def __init__(self

                 ):

    @abstractmethod
    def implement_encoding(self, payload: ):

    def encode(self,
               header_factory: Callable[]
               channel_manager,
               block: Block
               ):
        header = self.make_header



    def decode(self):