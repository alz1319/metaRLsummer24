import torch
import torch.nn as nn  # For building neural networks
import torch.optim as optim  # For optimization algorithms
import torch.nn.functional as F  # For parameterless functions, like activations
from torch.utils.data import DataLoader  # For managing data batches
import torchvision.datasets as datasets  # Standard datasets for vision
import torchvision.transforms as transforms  # Data transformations
from tqdm import tqdm  # For progress bar

# Set device for computations (GPU if available, otherwise CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define hyperparameters for the network
input_size = 28  # Input size (each image is 28x28 pixels)
sequence_length = 28  # Sequence length (28 rows per image)
num_layers = 2  # Number of LSTM layers
hidden_size = 256  # Number of hidden units in LSTM
num_classes = 10  # Number of classes (digits 0-9)
learning_rate = 3e-4  # Learning rate for optimizer
batch_size = 64  # Batch size for training
num_epochs = 2  # Number of training epochs

# Create a bidirectional LSTM neural network
class BRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # LSTM layer: input size, hidden size, number of layers, batch_first=True, bidirectional=True
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        # Fully connected layer: maps from hidden size to number of classes
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # Initialize hidden state and cell state for LSTM
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)

        # Forward propagate through LSTM
        out, _ = self.lstm(x, (h0, c0))
        # Pass the output of the last time step through the fully connected layer
        out = self.fc(out[:, -1, :])
        return out

# Function to save a model checkpoint
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

# Function to load a model checkpoint
def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

# Load MNIST dataset for training and testing
train_dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root="dataset/", train=False, transform=transforms.ToTensor(), download=True)

# Data loaders for creating batches
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize the neural network
model = BRNN(input_size, hidden_size, num_layers, num_classes).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer

# Train the network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        # Move data to the appropriate device and remove unnecessary dimensions
        data = data.to(device=device).squeeze(1)
        targets = targets.to(device=device)

        # Forward pass: compute predicted outputs
        scores = model(data)
        loss = criterion(scores, targets)

        # Backward pass: compute gradient of the loss with respect to model parameters
        optimizer.zero_grad()
        loss.backward()

        # Perform a single optimization step (parameter update)
        optimizer.step()

# Function to check the accuracy of the model
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()  # Set the model to evaluation mode

    # Disable gradient calculation (faster and saves memory)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)  # Get the index of the max log-probability
            num_correct += (predictions == y).sum()  # Count correct predictions
            num_samples += predictions.size(0)  # Count total samples

        print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}")

    model.train()  # Set the model back to training mode

# Save the current model checkpoint
checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
save_checkpoint(checkpoint)

# Uncomment to load a saved checkpoint
# load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

# Check accuracy on training and test sets
check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
