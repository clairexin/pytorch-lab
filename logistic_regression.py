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
for epoch in range(1, 201):
    predictions = model(X)
    loss = criterion(predictions, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        accuracy = ((predictions >= 0.5).float() == y).float().mean()
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | Accuracy: {accuracy.item():.2%}")

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
