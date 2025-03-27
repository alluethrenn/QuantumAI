import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Select important features using Grover’s Algorithm
def grover_feature_selection(X, threshold=0.5):
    """Simulated Grover's search to find important features."""
    important_features = (np.mean(X.numpy(), axis=0) > threshold).astype(int)
    selected_indices = np.where(important_features)[0]
    return selected_indices[:4]  # Select up to 4 features (for 4 qubits)

# Define quantum device
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

# Define quantum circuit
@qml.qnode(dev, interface="torch")
def quantum_layer(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# Define hybrid QNN model
class QuantumNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_weights = nn.Parameter(torch.randn((3, n_qubits), requires_grad=True))
        self.fc1 = nn.Linear(4, n_qubits)  # Select 4 features using Grover's Algorithm
        self.fc2 = nn.Linear(n_qubits, 10)  # 10 classes for MNIST

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten image
        selected_indices = grover_feature_selection(x)
        x = x[:, selected_indices]  # Apply Grover’s feature selection
        x = torch.tanh(self.fc1(x))
        q_out = torch.stack([torch.tensor(quantum_layer(x_i, self.q_weights)) for x_i in x])
        return self.fc2(q_out)

# Initialize model, loss function, and optimizer
model = QuantumNeuralNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model
epochs = 3
for epoch in range(epochs):
    for images, labels in train_loader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

print("Training complete.")

# Evaluate the model