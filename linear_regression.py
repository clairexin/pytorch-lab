import json
import os
import torch
import torch.nn as nn

# Generate synthetic data: y = 2x + 1 + noise
torch.manual_seed(42)
X = torch.randn(100, 1)
y = 2 * X + 1 + 0.1 * torch.randn(100, 1)

# Model, loss, optimizer
model = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
losses = []
for epoch in range(1, 201):
    predictions = model(X)
    loss = criterion(predictions, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f}")

# Results
weight = model.weight.item()
bias = model.bias.item()
print(f"\nLearned: w={weight:.4f}, b={bias:.4f}")
print(f"Expected: w=2.0000, b=1.0000")

# Sample prediction
x_test = torch.tensor([[5.0]])
print(f"\nPrediction for x=5: {model(x_test).item():.4f} (expected ~11.0)")

# Fit line for visualization
x_line = torch.linspace(X.min().item(), X.max().item(), 100).unsqueeze(1)
with torch.no_grad():
    y_line = model(x_line)

# Write JSON for web dashboard
os.makedirs("data", exist_ok=True)
data = {
    "dataset": {
        "x": X.squeeze().tolist(),
        "y": y.squeeze().tolist(),
    },
    "training": {
        "epochs": list(range(1, 201)),
        "losses": losses,
    },
    "results": {
        "weight": weight,
        "bias": bias,
        "fit_line_x": x_line.squeeze().tolist(),
        "fit_line_y": y_line.squeeze().tolist(),
    },
    "predictions": {
        "x": 5.0,
        "y_pred": model(x_test).item(),
        "y_expected": 11.0,
    },
}
with open("data/linear_regression.json", "w") as f:
    json.dump(data, f)
print("\nWrote data/linear_regression.json")
