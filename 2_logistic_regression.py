import json
import os
import torch
import torch.nn as nn

# Generate toy 2D dataset: two clusters
torch.manual_seed(42)
n_samples = 100

# Class 0: centered around (-1, -1), Class 1: centered around (1, 1)
X = torch.cat([
    torch.randn(n_samples, 2) * 0.5 + torch.tensor([-1.0, -1.0]),
    torch.randn(n_samples, 2) * 0.5 + torch.tensor([1.0, 1.0]),
])
y = torch.cat([torch.zeros(n_samples), torch.ones(n_samples)]).unsqueeze(1)

# Model: linear layer + sigmoid
model = nn.Sequential(
    nn.Linear(2, 1),
    nn.Sigmoid(),
)
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Training loop
losses = []
accuracies = []
for epoch in range(1, 201):
    predictions = model(X)
    loss = criterion(predictions, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    accuracy = ((predictions >= 0.5).float() == y).float().mean().item()
    losses.append(loss.item())
    accuracies.append(accuracy)

    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | Accuracy: {accuracy:.2%}")

# Results
linear = model[0]
w1, w2 = linear.weight[0].tolist()
b = linear.bias.item()
print(f"\nLearned weights: w1={w1:.4f}, w2={w2:.4f}, b={b:.4f}")

# Sample predictions
samples = torch.tensor([[-2.0, -2.0], [0.0, 0.0], [2.0, 2.0]])
with torch.no_grad():
    probs = model(samples).squeeze()
for point, prob in zip(samples, probs):
    label = 1 if prob >= 0.5 else 0
    print(f"  ({point[0]:+.0f}, {point[1]:+.0f}) -> p={prob:.4f} -> class {label}")

# Decision boundary line: w1*x1 + w2*x2 + b = 0
x1_range = torch.linspace(-3, 3, 100)
x2_boundary = -(w1 * x1_range + b) / w2

# Decision grid for heatmap
grid_size = 100
gx = torch.linspace(-3, 3, grid_size)
gy = torch.linspace(-3, 3, grid_size)
xx, yy = torch.meshgrid(gx, gy, indexing="xy")
grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
with torch.no_grad():
    probs_grid = model(grid).squeeze().reshape(grid_size, grid_size)

# Write JSON for web dashboard
os.makedirs("data", exist_ok=True)
data = {
    "dataset": {
        "x1": X[:, 0].tolist(),
        "x2": X[:, 1].tolist(),
        "labels": y.squeeze().tolist(),
    },
    "training": {
        "epochs": list(range(1, 201)),
        "losses": losses,
        "accuracies": accuracies,
    },
    "decision_boundary": {
        "x1": x1_range.tolist(),
        "x2": x2_boundary.tolist(),
    },
    "decision_grid": {
        "x_range": gx.tolist(),
        "y_range": gy.tolist(),
        "probabilities": probs_grid.tolist(),
    },
    "results": {
        "w1": w1,
        "w2": w2,
        "bias": b,
    },
    "predictions": [
        {"point": s.tolist(), "prob": p.item(), "label": int(p >= 0.5)}
        for s, p in zip(samples, probs)
    ],
}
with open("data/2_logistic_regression.json", "w") as f:
    json.dump(data, f)
print("\nWrote data/2_logistic_regression.json")
