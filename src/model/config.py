"""
Centrally located config object.

I figure if I ever feel like it I can make this load from
a proper config and just display the right properties
"""

from enum import Enum

class Config(Enum):
    """
    The config object. Contains a bunch of config info
    """
    # Data channel names
    #
    # This mainly controls what things are called as they
    # flow through dictionaries.

    SHAPES_NAME = "shape"
    TARGETS_NAME = "targets"
    CONTEXT_NAME = "context"

    ### Error handling



    ### Logging stuff

    LOGGING = True # Whether logging is enabled
    LOGGING_VERBOSITY_THRESHOLD = 3 # Messages with this and below verbosity are displayed
    LOGGING_DESTINATION = "console" # Where to send the messages