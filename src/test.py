import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer


# Custom LSTM cell with digital memory update using Gumbel-Softmax sampling
class DigitalMemoryLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, memory_range=(-5, 5), temp=0.5):
        super(DigitalMemoryLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_range = memory_range
        self.temp = temp

        # Linear layers for input and hidden transformations
        self.input_transform = nn.Linear(input_size, hidden_size)
        self.hidden_transform = nn.Linear(hidden_size, hidden_size)

        # Output logits for memory update (for discrete values)
        self.memory_update_logits = nn.Linear(hidden_size, memory_range[1] - memory_range[0] + 1)

    def forward(self, x, hidden):
        h, memory = hidden

        # Combine input and hidden transformations
        combined = torch.tanh(self.input_transform(x) + self.hidden_transform(h))

        # Predict memory updates
        logits = self.memory_update_logits(combined)
        memory_update = F.gumbel_softmax(logits, tau=self.temp, hard=True)

        # Calculate update amount (weighted sum of discrete values)
        discrete_values = torch.arange(self.memory_range[0], self.memory_range[1] + 1, device=x.device)
        update_amount = (memory_update * discrete_values).sum(dim=1, keepdim=True)

        # Update the memory
        memory = memory + update_amount

        return combined, (combined, memory)


# Stacked LSTM with digital memory
class StackedDigitalMemoryLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(StackedDigitalMemoryLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.cells = nn.ModuleList(
            [DigitalMemoryLSTMCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        for i, cell in enumerate(self.cells):
            outputs = []
            hx, memory = hidden[i]
            for j in range(x.size(1)):
                out, (hx, memory) = cell(x[:, j, :], (hx, memory))
                outputs.append(out)
            x = torch.stack(outputs, dim=1)
            hidden[i] = (hx, memory)
        outputs = self.fc(x)
        return outputs, hidden

    def init_hidden(self, batch_size):
        return [(torch.zeros(batch_size, self.hidden_size), torch.zeros(batch_size, 1)) for _ in range(self.num_layers)]


# Stacked Standard LSTM
class StackedStandardLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(StackedStandardLSTM, self).__init__()
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        lstm_out, hidden = self.lstm(x, hidden)
        outputs = self.fc(lstm_out)
        return outputs, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))


# Load dataset and tokenizer
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


# Tokenize dataset
def tokenize_data(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128, return_tensors="pt")


tokenized_dataset = dataset["train"].map(tokenize_data, batched=True, remove_columns=["text"])


# Prepare DataLoader with correct handling for tokenized data
def collate_fn(batch):
    inputs = torch.stack([torch.tensor(b["input_ids"]) for b in batch])
    targets = torch.stack([torch.tensor(b["input_ids"])[1:] for b in batch])  # Shift targets by 1 for next-token prediction
    inputs = F.one_hot(inputs[:, :-1], num_classes=tokenizer.vocab_size).float()  # One-hot encode the inputs
    return inputs, targets

dataloader = DataLoader(tokenized_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# Hyperparameters
input_size = tokenizer.vocab_size
hidden_size = 64
output_size = tokenizer.vocab_size
max_batches = 4000  # Set the maximum number of batches to train on
num_epochs = 5
num_layers = 3  # Number of LSTM layers

# Initialize models
digital_lstm = StackedDigitalMemoryLSTM(input_size, hidden_size, output_size, num_layers)
standard_lstm = StackedStandardLSTM(input_size, hidden_size, output_size, num_layers)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
digital_optimizer = torch.optim.Adam(digital_lstm.parameters(), lr=0.001)
standard_optimizer = torch.optim.Adam(standard_lstm.parameters(), lr=0.001)


# Training loop with batch limit
def train(model, optimizer, dataloader, epochs, model_name, max_batches):
    model.train()
    batch_count = 0
    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in dataloader:
            if batch_count >= max_batches:
                break

            hidden = model.init_hidden(inputs.size(0))
            optimizer.zero_grad()
            outputs, hidden = model(inputs, hidden)

            loss = criterion(outputs.view(-1, output_size), targets.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        print(f'{model_name} Epoch [{epoch}/{epochs}], Loss: {total_loss / len(dataloader):.4f}')
        if batch_count >= max_batches:
            break


# Sentence continuation
def generate_sentence(model, tokenizer, seed_text, max_length=20):
    model.eval()
    tokens = tokenizer.encode(seed_text, return_tensors="pt")
    tokens = F.one_hot(tokens[:, :-1], num_classes=tokenizer.vocab_size).float()

    hidden = model.init_hidden(1)
    for _ in range(max_length):
        with torch.no_grad():
            output, hidden = model(tokens, hidden)
            next_token_logits = output[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).item()
            next_word = tokenizer.decode([next_token])
            seed_text += " " + next_word

            next_token_one_hot = F.one_hot(torch.tensor([[next_token]]), num_classes=tokenizer.vocab_size).float()
            tokens = torch.cat((tokens, next_token_one_hot), dim=1)

    return seed_text


# Train both models
print("Training StackedDigitalMemoryLSTM...")
train(digital_lstm, digital_optimizer, dataloader, num_epochs, "StackedDigitalMemoryLSTM", max_batches)

print("\nTraining StackedStandardLSTM...")
train(standard_lstm, standard_optimizer, dataloader, num_epochs, "StackedStandardLSTM", max_batches)

# Generate sentence continuations
seed_text = "The history of science"
print("\nStackedDigitalMemoryLSTM sentence continuation:")
print(generate_sentence(digital_lstm, tokenizer, seed_text))

print("\nStackedStandardLSTM sentence continuation:")
print(generate_sentence(standard_lstm, tokenizer, seed_text))
