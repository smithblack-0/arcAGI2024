from typing import List, Any, Tuple, Callable

# Define schema and vocabulary types.
Schema = List[int]

# Define logging types.

LoggingCallback = Callable[[str|Exception, int], None]
