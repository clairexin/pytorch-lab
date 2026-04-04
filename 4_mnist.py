import json
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms

# Download and load MNIST
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root="data/mnist_raw", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="data/mnist_raw", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# Simple CNN
model = nn.Sequential(
    nn.Conv2d(1, 16, 3, padding=1),   # 28x28 -> 28x28
    nn.ReLU(),
    nn.MaxPool2d(2),                    # 28x28 -> 14x14
    nn.Conv2d(16, 32, 3, padding=1),   # 14x14 -> 14x14
    nn.ReLU(),
    nn.MaxPool2d(2),                    # 14x14 -> 7x7
    nn.Flatten(),
    nn.Linear(32 * 7 * 7, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 5
train_losses = []
train_accuracies = []

for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)
    print(f"Epoch {epoch}/{epochs} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2%}")

# Evaluation on test set
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        preds = outputs.argmax(1)
        all_preds.append(preds)
        all_labels.append(labels)

all_preds = torch.cat(all_preds)
all_labels = torch.cat(all_labels)
test_accuracy = (all_preds == all_labels).float().mean().item()
print(f"\nTest Accuracy: {test_accuracy:.2%}")

# Confusion matrix (10x10)
confusion = [[0] * 10 for _ in range(10)]
for true, pred in zip(all_labels.tolist(), all_preds.tolist()):
    confusion[true][pred] += 1

# Model summary
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")
layer_info = []
for layer in model:
    if hasattr(layer, "weight"):
        params = layer.weight.numel() + layer.bias.numel()
        print(f"  {layer} -> {params} params")
        layer_info.append({"name": str(layer), "params": params})

# Sample predictions (16 random test images)
torch.manual_seed(42)
indices = torch.randperm(len(test_dataset))[:16]
sample_predictions = []
with torch.no_grad():
    for idx in indices:
        image, true_label = test_dataset[idx]
        output = model(image.unsqueeze(0))
        probs = torch.softmax(output, dim=1).squeeze()
        pred_label = probs.argmax().item()
        confidence = probs[pred_label].item()
        sample_predictions.append({
            "image": image.squeeze().tolist(),  # 28x28 pixel values
            "true_label": true_label,
            "predicted_label": pred_label,
            "confidence": round(confidence, 4),
        })

# Write JSON
os.makedirs("data", exist_ok=True)
data = {
    "training": {
        "epochs": list(range(1, epochs + 1)),
        "losses": train_losses,
        "accuracies": train_accuracies,
    },
    "evaluation": {
        "test_accuracy": test_accuracy,
        "confusion_matrix": confusion,
        "class_labels": list(range(10)),
    },
    "architecture": {
        "layers": layer_info,
        "total_params": total_params,
    },
    "sample_predictions": sample_predictions,
}
with open("data/4_mnist.json", "w") as f:
    json.dump(data, f)
print("\nWrote data/4_mnist.json")
