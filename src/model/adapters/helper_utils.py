import textwrap
from typing import Any, Type, get_origin, get_args


## Validation mechanism for the config language
#
# A config entry can be one of the python primitives, or
# a dictionary, list, or tuple generic. Generics can be nested,
# but MUST terminate in a python primitive type. No
# unions are allowed.


supported_primitives = (int, float, str, bool)

def validate_config(config: Any, type_spec: Any):
    """
    Validates that a given config has a typing pattern that matches
    the given type spec. This means that, for nongenerics, the leaves must
    align with a particular type, and for generics the pattern must match.

    ---- primitives ----

    Several primitive leaves are supported. At the time
    of writing this documentation, the leaves supported are:

    * int
    * float
    * bool
    * str

    Typespec primitives behave different depending on if they
    are provided in terms of their class or an instance. For example,
    a typespec given in terms of the "int" class will allow any
    "int" like 3 or 5, but not a float like 3.7.

    However, if you were to provide a typespec string of "foo" then
    it will only match if the string "foo" is found in that spot on
    the config.

    ---- instanced pytrees ----

    At the time of writing, three instanced pytrees are supported.
    These are:

    * dict
    * list
    * tuple

    When an instanced pytree is used as a config, it is interpeted as
    a directive to use that exact pytree shape, with the given
    spefication. For instance, a dictionary of the following

    {"embedding_dim" : int, "batch_size" : int}

    Specifies that we expect a dictionary of length two, with
    a feature called "embedding dim" that must be an int, and
    another called "batch_size" that is an int.

    ---- generic pytrees and variable lengths----

    Two generic pytrees are also supported. These are
    the Dict and List generic. They may be variable length.

    Unlike an instanced pytree typespec, a generic typespec
    must not be of mixed type. For instance, Dict[str, int]
    would be allowed, but Dict[str, int|float] would not.
    """
    # The following code is strutured to lock onto
    # various type spec cases, then see if the config
    # case matches.
    #
    # If NO typespec case matches, it is assumed to
    # be unsupported and throws an error at the end of
    # the function.


    # Handle the literal primitives
    #
    # A literal primitive must match exactly with the
    # typespec primitive. For instance, both must be
    # "embeddding_dim"
    if isinstance(type_spec, supported_primitives):
        if type_spec != config:
            msg = f"""
            Issue detected when validating config for enforcement of literal value.
            Literal value was supposed to be {type_spec}, but was actually {config}
            """
            msg = textwrap.dedent(msg)
            raise TypeError(msg)
        return

    # We need to handle the dictionary instances
    # here.
    #
    # We check the length and keys are the same, and
    # then check the values match the typespec.
    if isinstance(type_spec, dict):
        if not isinstance(config, dict):
            msg = f"""
            Issue was detected when validating config for dictionary instance match.
            The config was expected to have a dictionary entry. However, we actually 
            found {type(config)}
            """
            msg = textwrap.dedent(msg)
            raise TypeError(msg)
        if len(config) != len(type_spec):
            msg = f"""
            Issue was detected when validating config for dictionary instance match.
            Config was expected to be a dictionary of length '{len(type_spec)}'. 
            However, we actually got a dictionary of length '{len(type_spec)}'
            """
            msg = textwrap.dedent(msg)
            raise TypeError(msg)
        if set(type_spec.keys()) != set(config.keys()):
            msg = f"""
            Issue was detecting when validating config for dictionary instance match.
            Config was expected to have keys '{type_spec.keys()}'. 
            However, actually observed were keys '{config.keys()}'
            """
            msg = textwrap.dedent(msg)
            raise TypeError(msg)
        for key in type_spec.keys():
            try:
                validate_config(config[key], type_spec[key])
            except TypeError as e:
                msg = f"""
                Issue was detecting when validating config for dictionary instance match.
                The issue occurred when considering key named '{key}'
                """
                msg = textwrap.dedent(msg)
                raise TypeError(msg) from e
        return

    # We handle lists here.
    #
    # We check the lists have the same length, then
    # check each individual value.
    if isinstance(type_spec, list):
        if not isinstance(config, list):
            msg = f"""
            Issue was detected when validating config for list instance match.
            Config was expected to be of type list. However, 
            config was found to actually be of type '{type(config)}'
            """
            msg = textwrap.dedent(msg)
            raise TypeError(msg)
        if len(config) != len(type_spec):
            msg = f"""
            Issue was detected when validating config for list instance match.
            Config was expected to be a list of length '{len(type_spec)}'. 
            However, we actually got a list of length '{len(type_spec)}'
            """
            msg = textwrap.dedent(msg)
            raise TypeError(msg)

        for i, (config_item, spec_item) in enumerate(zip(config, type_spec)):
            try:
                validate_config(config_item, spec_item)
            except TypeError as e:
                msg = f"""
                Issue was detected when validating config for list instance match.
                Issue occurred while validating list index '{i}'
                """
                msg = textwrap.dedent(msg)
                raise TypeError(msg) from e
        return

    # We handle tuples here. The tuples must
    # be the same length, and have the same types
    # at the same location.
    if isinstance(type_spec, tuple):
        if not isinstance(config, tuple):
            msg = f"""
            Issue was detected while validating config for tuple instance match.
            Config was expected to be of type tuple. 
            However, it was actually found to be of type '{type(config)}'.
            """
            msg = textwrap.dedent(msg)
            raise TypeError(msg)
        if len(config) != len(type_spec):
            msg = f"""
            Issue was detected while validating config for tuple instance match
            Config was expected to be tuple of length '{len(type_spec)}',
            however it was actually found to be length '{len(config)}'            
            """
            msg = textwrap.dedent(msg)
            raise TypeError(msg)
        for i , (config_item, spec_item) in enumerate(zip(config, type_spec)):
            try:
                validate_config(config_item, spec_item)
            except TypeError as e:
                msg = f"""
                Issue was detected while validating config for tuple instance match.
                The issue occurred at tuple index '{i}'      
                """
                msg = textwrap.dedent(msg)
                raise TypeError(msg) from e
        return


    # Handle the generic primitives. These need
    # only match in terms of type, and are indicated
    # by providing a type, For instance, typespec of
    # 'int' matches config of '5'

    if not get_origin(type_spec) and issubclass(type_spec, supported_primitives):
        if not isinstance(config, type_spec):
            msg = f"""
            Issue detected when validating config for correct types of literal values. 
            Config was expected to have literal of type '{type_spec.__name__}',
            but was found to actually have type {type(config)}
            """
            msg = textwrap.dedent(msg)
            raise TypeError(msg)
        return

    # Anything left over, if it is valid, MUST be one of
    # the generic types. We handle those next.

    origin = get_origin(type_spec)
    args = get_args(type_spec)

    if origin is list:
        if not isinstance(config, list):
            msg = f"""
            Issue occurred while validating conformance of config to generic list
            Config should have been a list. However, we actually recieved a type of {type(config)}
            """
            msg = textwrap.dedent(msg)
            raise TypeError(msg)
        for i, item in enumerate(config):
            try:
                validate_config(item, args[0])
            except TypeError as e:
                msg = f"""
                Issue occurred while validating conformance of config to generic list. 
                Issue occurred at index '{i}'
                """
                msg = textwrap.dedent(msg)
                raise TypeError(msg) from e
        return

    elif origin is dict:
        if not isinstance(config, dict):
            msg = f"""
            Issue occurred while validating conformance of config to generic dict.
            Config was expected to be a dictionary, but we actually got '{type(config)}'
            """
            msg = textwrap.dedent(msg)
            raise ValueError(msg)
        key_type, value_type = args
        for key, value in config.items():
            # Catch key issues


            try:
                validate_config(key, key_type)
                validate_config(value, value_type)
            except TypeError as e:
                msg = f"""
                Issue occurred while validating conformance of config to generic dict.
                The issue occurred with key named '{key}'.
                """
                msg = textwrap.dedent(msg)
                raise TypeError(msg) from e
        return

    msg = f"""
    Corrupt config specification detected. 
    The following was provided in the 'type_spec' parameter, but does not
    match any expected validation cases. It is thus not supported:
    
    {type_spec}'
    """
    msg = textwrap.dedent(msg)
    raise RuntimeError(msg)

def pack_tensors()


def unpack_to_multidimensional()