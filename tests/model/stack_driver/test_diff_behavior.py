import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.main.model.subroutine_driver import SubroutineDriver, SubroutineCore
import unittest


# Load Titanic dataset
titanic = pd.read_csv('titanic.csv')

# Drop unnecessary columns and handle missing data
titanic = titanic.drop(['Name', 'Ticket', 'Cabin'], axis=1)
titanic['Age'].fillna(titanic['Age'].median(), inplace=True)
titanic['Embarked'].fillna(titanic['Embarked'].mode()[0], inplace=True)

# Convert categorical data to numeric
titanic['Sex'] = titanic['Sex'].map({'male': 0, 'female': 1})
titanic['Embarked'] = titanic['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Define features and labels
X = titanic.drop('Survived', axis=1)
y = titanic['Survived']

# Scale the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)


class TitanicSubroutineCore(SubroutineCore):
    """
    A simple feedforward core for decision making.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, d_model)
        )

    def setup_state(self, tensor: torch.Tensor):
        # This model doesn't require a complex state, just return empty
        return {}

    def forward(self, tensor: torch.Tensor, states: dict):
        """
        Perform forward pass for the model.
        """
        out = self.ffn(tensor)
        return out, states


class TestSubroutineDriver(unittest.TestCase):

    def create_driver(self, d_model, stack_depth, min_iterations, max_iterations):
        """
        Helper to create SubroutineDriver.
        """
        core = TitanicSubroutineCore(d_model)
        driver = SubroutineDriver(d_model, stack_depth, core)
        return driver

    def test_subroutine_driver_with_accumulation(self):
        """
        Test if the model uses the stack to extend computation.
        """
        d_model = X_train.shape[1]  # Use the input features count as d_model
        stack_depth = 5
        min_iterations_before_destack = 10
        max_iterations_before_flush = 20

        # Create SubroutineDriver
        driver = self.create_driver(d_model, stack_depth, min_iterations_before_destack, max_iterations_before_flush)

        # Forward through driver
        tensor_accumulators, state_accumulators = driver(X_train, max_computation_iterations=20, state=None)

        # Check that tensor_accumulators match the expected dimensions
        self.assertEqual(tensor_accumulators.shape, (X_train.shape[0], 1))

    def test_residual_bypass_model(self):
        """
        Test a control model with residual bypass and LayerNorm.
        """
        d_model = X_train.shape[1]
        core = TitanicSubroutineCore(d_model)

        # Simply perform 10 residual bypass steps with LayerNorm
        layernorm = nn.LayerNorm(d_model)

        output = X_train
        for _ in range(10):
            output, _ = core(output, None)
            output = layernorm(output + X_train)

        self.assertEqual(output.shape, (X_train.shape[0], 1))


# If this is your test entry point
if __name__ == '__main__':
    unittest.main()

