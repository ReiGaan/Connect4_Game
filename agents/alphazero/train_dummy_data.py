import torch
from torch.utils.data import DataLoader
import numpy as np
from network import Connect4Net, BoardDataset, CustomLoss

# Generate dummy data (replace this with self-play later)
def generate_dummy_data(n=100):
    states = np.random.randint(0, 2, (n, 3, 6, 7))  # 3 channels, 6x7 board
    policies = np.random.dirichlet(np.ones(7), size=n)  # 7 moves
    values = np.random.uniform(-1, 1, (n, 1))  # scalar value
    data = np.array(list(zip(states, policies, values)), dtype=object)
    return data

# Load dummy dataset
data = generate_dummy_data(500)
dataset = BoardDataset(data)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model and training components
model = Connect4Net()
loss_fn = CustomLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(5):  # Start with just 5 epochs
    total_loss = 0
    for batch in loader:
        states, target_policy, target_value = batch
        states = states.float()
        target_policy = target_policy.float()
        target_value = target_value.float()

        pred_policy, pred_value = model(states)
        loss = loss_fn(target_value, pred_value, target_policy, pred_policy)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Save model
torch.save(model.state_dict(), "model.pt")
print("Model saved as model.pt")
