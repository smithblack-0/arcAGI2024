"""
Exceptions file, to contain the various exceptions we might use
"""


import traceback
import asyncio
from config import Config
class CoreException(Exception):
    """
    All exceptions thrown by the
    project share this subtype
    """
    def __init__(self,
                 message,
                 **parameters):
        """
        :param message: The message
        :param parameters: Any extra parameters to keep around
        """
        self.parameters = parameters
        super(CoreException, self).__init__(message, **parameters)

class AsyncException(CoreException):
    """
    An async specific exception, associated
    with the async batch processor mechanism.
    """

class AsyncTerminalException(CoreException):
    """
    An async specific exception. Any exception
    which ends up behaving like this will immediately
    result in the halting of the entire computational process.
    """
    def __init__(self, message, **parameters)
        super().__init__(message, **parameters)


class ModelException(Exception):
    """
    Any error which may be invoked from
    within the async batch processor layers
    """