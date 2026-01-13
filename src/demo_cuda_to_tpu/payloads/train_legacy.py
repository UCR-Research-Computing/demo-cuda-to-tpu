import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define a simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def main():
    # Hyperparameters
    batch_size = 2048  # Large batch to saturate A100
    learning_rate = 0.001
    epochs = 5

    # Data Loading (Synthetic to avoid download issues/time)
    print("Generating synthetic data...")
    # 60k images, 1 channel, 28x28
    train_data = torch.randn(60000, 1, 28, 28)
    train_labels = torch.randint(0, 10, (60000,))
    dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Starting Training...")
    model.train()
    
    start_total = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Time: {epoch_time:.4f}s")

    total_time = time.time() - start_total
    print(f"Total Training Time: {total_time:.4f}s")

if __name__ == "__main__":
    main()
