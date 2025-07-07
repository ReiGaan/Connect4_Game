import torch
from torch.utils.data import DataLoader
import numpy as np
from network import Connect4Net, BoardDataset, CustomLoss


# Generate dummy data (replace with self-play later)
def generate_dummy_data(n=100):
    """
    Generates dummy training data for a Connect4 AlphaZero agent.

    Parameters:
        n (int): Number of data samples to generate. Default is 100.

    Returns:
        np.ndarray: An array of length n, where each element is a tuple containing:
            - states (np.ndarray): A (3, 6, 7) array representing the board state with 3 channels.
            - policies (np.ndarray): A (7,) array representing the move probabilities (policy vector).
            - values (float): A scalar value in the range [-1, 1] representing the value of the state.
    """
    states = np.random.randint(0, 2, (n, 3, 6, 7))  
    policies = np.random.dirichlet(np.ones(7), size=n)  
    values = np.random.uniform(-1, 1, (n, 1)) 
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
