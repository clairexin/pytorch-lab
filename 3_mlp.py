import json
import os
import torch
import torch.nn as nn

# Generate toy dataset: concentric circles (not linearly separable)
torch.manual_seed(42)
n_samples = 200

# Inner circle (class 0): radius ~1
angles_inner = torch.rand(n_samples) * 2 * 3.14159
r_inner = torch.randn(n_samples) * 0.1 + 1.0
X_inner = torch.stack([r_inner * angles_inner.cos(), r_inner * angles_inner.sin()], dim=1)

# Outer circle (class 1): radius ~3
angles_outer = torch.rand(n_samples) * 2 * 3.14159
r_outer = torch.randn(n_samples) * 0.1 + 3.0
X_outer = torch.stack([r_outer * angles_outer.cos(), r_outer * angles_outer.sin()], dim=1)

X = torch.cat([X_inner, X_outer])
y = torch.cat([torch.zeros(n_samples), torch.ones(n_samples)]).unsqueeze(1)

# MLP: 2 -> 16 -> 8 -> 1
model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid(),
)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
losses = []
accuracies = []
for epoch in range(1, 301):
    predictions = model(X)
    loss = criterion(predictions, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    accuracy = ((predictions >= 0.5).float() == y).float().mean().item()
    losses.append(loss.item())
    accuracies.append(accuracy)

    if epoch % 30 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | Accuracy: {accuracy:.2%}")

# Model summary
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params}")
layer_info = []
for i, layer in enumerate(model):
    if hasattr(layer, "weight"):
        params = layer.weight.numel() + layer.bias.numel()
        print(f"  {layer} -> {params} params")
        layer_info.append({
            "name": str(layer),
            "params": params,
        })

# Sample predictions at different radii
print("\nSample predictions (distance from origin):")
samples = torch.tensor([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [0.0, 4.0]])
with torch.no_grad():
    probs = model(samples).squeeze()
for point, prob in zip(samples, probs):
    r = point.norm().item()
    label = 1 if prob >= 0.5 else 0
    print(f"  r={r:.1f} -> p={prob:.4f} -> class {label}")

# Decision grid for heatmap
grid_size = 100
gx = torch.linspace(-4, 4, grid_size)
gy = torch.linspace(-4, 4, grid_size)
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
        "epochs": list(range(1, 301)),
        "losses": losses,
        "accuracies": accuracies,
    },
    "architecture": {
        "layers": layer_info,
        "total_params": total_params,
    },
    "decision_grid": {
        "x_range": gx.tolist(),
        "y_range": gy.tolist(),
        "probabilities": probs_grid.tolist(),
    },
    "predictions": [
        {"point": s.tolist(), "radius": s.norm().item(), "prob": p.item(), "label": int(p >= 0.5)}
        for s, p in zip(samples, probs)
    ],
}
with open("data/3_mlp.json", "w") as f:
    json.dump(data, f)
print("\nWrote data/3_mlp.json")
