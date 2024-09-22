import torch

class CBTensor(torch.Tensor):
    def __new__(cls, tensor: torch.Tensor, spec: dict):
        # Create a new instance of the CBTensor class

        return object.__new__(cls, tensor, spec)

    def __init__(self, tensor: torch.Tensor, spec: dict):
        self.spec = spec  # Store additional metadata
        self.tensor = tensor  # Underlying tensor

    # Custom method unique to CBTensor
    def custom_method(self):
        return "Custom method in CBTensor"

    # Overriding __getattribute__ to handle attribute access and delegation
    def __getattribute__(self, item):
        try:
            # Try to access the attribute directly from CBTensor instance
            return object.__getattribute__(self, item)
        except AttributeError:
            # If not found, delegate to the underlying tensor
            return getattr(self.tensor, item)


# Test Script
def test_cbtensor():
    # Create a standard PyTorch tensor
    tensor = torch.randn(3, 4)

    # Create an instance of CBTensor
    cb_tensor = CBTensor(tensor, {"channel1": 4})

    # Test access to custom method
    assert cb_tensor.custom_method() == "Custom method in CBTensor", "Failed: custom_method should be available in CBTensor"

    # Test access to torch.Tensor methods (delegated to tensor)
    assert cb_tensor.mean().item() == tensor.mean().item(), "Failed: mean() should be delegated to the underlying tensor"

    # Test access to tensor shape
    assert cb_tensor.shape == tensor.shape, "Failed: shape attribute should be delegated to the underlying tensor"

    # Test that the metadata (spec) is accessible
    assert cb_tensor.spec == {"channel1": 4}, "Failed: 'spec' attribute should be part of CBTensor"

    print("All tests passed!")

if __name__ == "__main__":
    test_cbtensor()
