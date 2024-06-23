import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class Network(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Network, self).__init__()
        # Define the first fully connected layer
        self.fc1 = nn.Linear(input_size, 50)
        # Define the second fully connected layer
        self.fc2 = nn.Linear(50, num_classes)
    # Define the forward pass
    def forward(self, x):
        # Apply ReLU activation to the output of the first layer
        x = F.relu(self.fc1(x))
        # Output layer
        x = self.fc2(x)
        return x
    
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


# Set the device to GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define hyperparameters
input_size = 784 # 28x28 images flattened
num_classes = 10 # Number of output classes (digits 0-9)
learning_rate = 0.005
batch_size = 64
num_epochs = 1

# Load the training dataset
train_dataset = datasets.MNIST(root='dataset/', train = True, transform=transforms.ToTensor(), download = True)
# Create a data loader for the training dataset
train_loader = DataLoader(dataset=train_dataset, batch_size = batch_size, shuffle = True)
# Load the test dataset
test_dataset = datasets.MNIST(root='dataset/', train = False, transform = transforms.ToTensor(), download = True)
# Create a data loader for the test dataset
test_loader = DataLoader(dataset= test_dataset, batch_size = batch_size, shuffle = True)

# Initialize the neural network
model = Network(input_size = input_size, num_classes = num_classes).to(device)

# Define the loss function (Cross Entropy Loss)
criterion = nn.CrossEntropyLoss()
# Define the optimizer (Adam Optimizer)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Move data and targets to the device (GPU or CPU)
        data = data.to(device)
        targets = targets.to(device)

        # Reshape the data to fit the input size of the network (batch_size, input_size)
        data = data.reshape(data.shape[0], -1)

        # Forward pass: compute predicted outputs by passing inputs to the model
        scores = model(data)
        # Compute the loss
        loss = criterion(scores, targets)

        # Backward pass: compute gradient of the loss with respect to model parameters
        optimizer.zero_grad()  # Clear existing gradients
        loss.backward()
        optimizer.step()  # Update parameters based on gradients

# Function to check the accuracy of the model
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Disable gradient calculation
        for x, y in loader:
            # Move data and targets to the device (GPU or CPU)
            x = x.to(device)
            y = y.to(device)
            # Reshape the data to fit the input size of the network (batch_size, input_size)
            x = x.reshape(x.shape[0], -1)

            # Forward pass: compute predicted outputs by passing inputs to the model
            scores = model(x)
            # Get the predicted class with the highest score
            _, predictions = scores.max(1)
            # Count correct predictions
            num_correct += (predictions == y).sum()
            # Count total samples
            num_samples += predictions.size(0)

        # Print the accuracy
        print(f'Accuracy: {float(num_correct) / float(num_samples) * 100:.2f}%')
    
    model.train()  # Set the model back to training mode

# Check accuracy on training data
check_accuracy(train_loader, model)
# Check accuracy on test data
check_accuracy(test_loader, model)