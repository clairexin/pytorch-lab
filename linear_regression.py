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
for epoch in range(1, 201):
    predictions = model(X)
    loss = criterion(predictions, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

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
