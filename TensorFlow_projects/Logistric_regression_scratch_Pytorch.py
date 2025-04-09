import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Define transformations for MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

# Load MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

# Binary classification: Filter only digits 0 and 1
train_dataset.data = train_dataset.data[(train_dataset.targets == 0) | (train_dataset.targets == 1)]
train_dataset.targets = train_dataset.targets[(train_dataset.targets == 0) | (train_dataset.targets == 1)]

test_dataset.data = test_dataset.data[(test_dataset.targets == 0) | (test_dataset.targets == 1)]
test_dataset.targets = test_dataset.targets[(test_dataset.targets == 0) | (test_dataset.targets == 1)]

# Create data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # Output is a single probability for binary classification

    def forward(self, x):
        return torch.sigmoid(self.linear(x))  # Apply sigmoid activation for probability output

input_dim = 28 * 28  # MNIST images are 28x28 pixels
model = LogisticRegression(input_dim)

# Define loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent with learning rate of 0.01

epochs = 10

for epoch in range(epochs):
    model.train()
    for images, labels in train_loader:
        # Flatten images to vectors of size [batch_size, input_dim]
        images = images.view(-1, input_dim)

        # Convert labels to float (required for BCELoss)
        labels = labels.float().unsqueeze(1)

        # Forward pass: Compute predictions and loss
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass: Compute gradients and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.view(-1, input_dim)
        labels = labels.float().unsqueeze(1)

        outputs = model(images)
        predictions = (outputs >= 0.5).float()  # Threshold at 0.5 for binary classification

        total += labels.size(0)
        correct += (predictions == labels).sum().item()

accuracy = correct / total * 100
print(f'Accuracy on test set: {accuracy:.2f}%')


model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.view(-1, input_dim)
        labels = labels.float().unsqueeze(1)

        outputs = model(images)
        predictions = (outputs >= 0.5).float()  # Threshold at 0.5 for binary classification

        total += labels.size(0)
        correct += (predictions == labels).sum().item()

accuracy = correct / total * 100
print(f'Accuracy on test set: {accuracy:.2f}%')
