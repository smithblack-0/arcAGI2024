import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# Generate data
def generate_data(num_samples=1000):
    x1 = np.random.uniform(low=-10, high=10.0, size=(num_samples, 1))  # First feature
    x2 = np.random.uniform(low=-10, high=10.0, size=(num_samples, 1))  # Second feature
    y = x1 * x2  # Target is the product of x1 and x2
    X = np.hstack((x1, x2))  # Stack features together
    return torch.Tensor(X), torch.Tensor(y)


# Define the Headed Bilinear Layer
class HeadedBilinearLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads, bias=True):
        super(HeadedBilinearLayer, self).__init__()
        self.num_heads = num_heads
        self.bilinear = nn.Bilinear(in_features // num_heads, in_features // num_heads, out_features // num_heads,
                                    bias=bias)

    def forward(self, x):
        batch_size, input_dim = x.shape
        head_dim = input_dim // self.num_heads

        # Reshape the input into heads (batch_size, num_heads, head_dim)
        x_reshaped = x.view(batch_size, self.num_heads, head_dim)

        # Perform bilinear interaction across all heads in parallel
        output_heads = self.bilinear(x_reshaped, x_reshaped)  # Bilinear across heads

        # Reshape back to original shape (batch_size, out_dim)
        output = output_heads.view(batch_size, -1)
        return output


# Define the composite model with LinearReLU, Headed BilinearReLU, and final LinearReLU
class CompositeMultiplicationModel(nn.Module):
    def __init__(self):
        super(CompositeMultiplicationModel, self).__init__()
        self.fc1 = nn.Linear(2, 64)  # First linear layer
        self.headed_bilinear = HeadedBilinearLayer(64, 64, num_heads=8)  # Headed bilinear layer with 8 heads
        self.fc2 = nn.Linear(64, 1)  # Final linear layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # First LinearReLU layer
        x = torch.relu(self.headed_bilinear(x))  # Headed BilinearReLU layer
        x = self.fc2(x)  # Final Linear layer
        return x


# Train the model
def train_model(model, X_train, y_train, epochs=1000, lr=0.001):
    criterion = nn.MSELoss()  # Mean Squared Error Loss
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []
    for epoch in range(epochs):
        model.train()

        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Store loss and print it every 100 epochs
        losses.append(loss.item())
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    return losses


# Test the model
def test_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        mse = nn.MSELoss()(predictions, y_test)
        print(f'Test Mean Squared Error: {mse.item():.4f}')
    return predictions


# Test model on out-of-domain inputs
def test_out_of_domain(model, x1, x2):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.Tensor([[x1, x2]])
        predicted_product = model(input_tensor).item()
        actual_product = x1 * x2
        print(f'Out-of-domain Test: {x1} * {x2} = {actual_product:.4f} (Predicted: {predicted_product:.4f})')


# Main script to train and test the model
def main():
    # Generate training and test data
    X_train, y_train = generate_data(1000)  # Training data
    X_test, y_test = generate_data(100)  # Test data

    # Initialize the model
    model = CompositeMultiplicationModel()

    # Train the model
    print("Training the model...")
    losses = train_model(model, X_train, y_train, epochs=2000, lr=0.001)

    # Plot the training loss over time
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()

    # Test the model on in-domain data
    print("Testing the model...")
    predictions = test_model(model, X_test, y_test)

    # Compare a few test cases
    for i in range(5):
        x1, x2 = X_test[i]
        actual = y_test[i].item()
        predicted = predictions[i].item()
        print(f"Test {i + 1}: {x1.item()} * {x2.item()} = {actual:.4f} (Predicted: {predicted:.4f})")

    # Test the model on out-of-domain data
    print("\nTesting on out-of-domain inputs:")
    test_out_of_domain(model, 100, -309)  # Test with 100 * -309


if __name__ == "__main__":
    main()
