import inspect

class ExampleClass:
    def __init__(self, a, b: int, c: float = 1.0):
        pass

# Get the constructor signature
constructor_params = inspect.signature(ExampleClass.__init__).parameters

# Output the signature mapping
for name, param in constructor_params.items():
    print(f"Name: {name}, Annotation: {param.annotation}, Default: {param.default}")
