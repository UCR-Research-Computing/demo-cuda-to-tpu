import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def main():
    # Hyperparameters for "Heavy" Load
    batch_size = 128
    steps = 300  # Targeted for ~3 mins on A100
    learning_rate = 0.001

    # Model: ResNet-50 (Standard ImageNet workhorse)
    print("Initializing ResNet-50...")
    model = models.resnet50(weights=None).to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Data: Synthetic ImageNet (Batch, 3, 224, 224)
    print(f"Allocating Synthetic Data ({batch_size}, 3, 224, 224)...")
    # Pre-allocate to measure pure compute throughput, not data loading
    inputs = torch.randn(batch_size, 3, 224, 224, device=device)
    labels = torch.randint(0, 1000, (batch_size,), device=device)

    print("Starting Training Comparison...")

    # Warmup
    for _ in range(5):
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    print("Warmup complete.")

    start_time = time.time()

    for step in range(1, steps + 1):
        step_start = time.time()

        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        # Sync for accurate timing (remove in prod for speed, keeping here for demo pacing)
        # We won't sync every step to allow overlap, but we sync for print
        if step % 10 == 0:
            torch.cuda.synchronize()
            elapsed = time.time() - step_start
            img_sec = batch_size / elapsed
            print(
                f"Step {step}/{steps} | Loss: {loss.item():.4f} | {img_sec:.1f} img/s"
            )

    torch.cuda.synchronize()
    total_time = time.time() - start_time
    throughput = (steps * batch_size) / total_time

    print(f"Total Time: {total_time:.2f}s")
    print(f"Average Throughput: {throughput:.1f} images/sec")


if __name__ == "__main__":
    main()
