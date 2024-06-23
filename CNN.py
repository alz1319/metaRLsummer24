import torch
import torch.nn.functional as F  
import torchvision.datasets as datasets  
import torchvision.transforms as transforms  
from torch import optim  
from torch import nn  
from torch.utils.data import DataLoader 
from tqdm import tqdm 

# Define a simple Convolutional Neural Network (CNN)
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        # First convolutional layer: takes 'in_channels' and produces 8 feature maps
        self.conv1 = nn.Conv2d(in_channels, out_channels=8, kernel_size=3, stride=1, padding=1)
        # Max pooling layer to reduce the spatial dimensions
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Second convolutional layer: takes 8 input channels and produces 16 feature maps
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        # Fully connected layer: flattens the feature maps into a single vector
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # Apply ReLU activation after the first convolution
        x = self.pool(x)  # Apply max pooling
        x = F.relu(self.conv2(x))  # Apply ReLU activation after the second convolution
        x = self.pool(x)  # Apply max pooling again
        x = x.view(x.size(0), -1)  # Flatten the tensor into a vector
        x = self.fc1(x)  # Apply the fully connected layer
        return x
    
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define hyperparameters
in_channels = 1  # Number of input channels (1 for grayscale images)
num_classes = 10  # Number of output classes (digits 0-9)
learning_rate = 3e-4  # Learning rate for the optimizer
batch_size = 64  # Number of samples per batch
num_epochs = 3  # Number of times to iterate over the entire dataset

# Load the MNIST training and test datasets
train_dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root="dataset/", train=False, transform=transforms.ToTensor(), download=True)

# Create data loaders for the training and test datasets
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize the CNN model
model = CNN(in_channels=in_channels, num_classes=num_classes).to(device)

# Define the loss function (cross-entropy loss) and the optimizer (Adam)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        data, targets = data.to(device), targets.to(device)  # Move data to the appropriate device

        # Forward pass: compute the model output
        scores = model(data)
        loss = criterion(scores, targets)  # Compute the loss

        # Backward pass: compute the gradients
        optimizer.zero_grad()  # Zero the gradients before backpropagation
        loss.backward()  # Perform backpropagation

        # Update the model parameters using the computed gradients
        optimizer.step()

# Function to evaluate the model's accuracy
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Disable gradient computation
        for x, y in loader:
            x, y = x.to(device), y.to(device)  # Move data to the appropriate device

            scores = model(x)  # Compute the model output
            _, predictions = scores.max(1)  # Get the index of the highest score
            num_correct += (predictions == y).sum()  # Count correct predictions
            num_samples += predictions.size(0)  # Count total samples

    model.train()  # Set the model back to training mode
    return num_correct / num_samples

checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
save_checkpoint(checkpoint)

#load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

# Print the accuracy on the training and test datasets
print(f"Accuracy on training set: {check_accuracy(train_loader, model) * 100:.2f}%")
print(f"Accuracy on test set: {check_accuracy(test_loader, model) * 100:.2f}%")
