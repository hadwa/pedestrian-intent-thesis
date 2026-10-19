import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import matplotlib.pyplot as plt




# Load pre-processed sequence data
y_seq = np.load("y_seq.npy")   # shape (N,)

X_seq = np.load("X_seq.npy")   # shape (N, 10, 40)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Define the LSTM model
class BIUKeypointLSTM(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=64, output_dim=1):
        super(BIUKeypointLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        out = self.fc(last_hidden)
        return self.sigmoid(out)

# Initialize model, loss function and optimizer
model = BIUKeypointLSTM()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 30
batch_size = 64
losses = []

for epoch in range(num_epochs):
    model.train()
    perm = torch.randperm(X_train_tensor.size(0))
    epoch_loss = 0

    for i in range(0, X_train_tensor.size(0), batch_size):
        indices = perm[i:i+batch_size]
        batch_x = X_train_tensor[indices]
        batch_y = y_train_tensor[indices]

        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg = epoch_loss / (X_train_tensor.size(0) // batch_size)
    losses.append(avg)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg:.4f}")

# Evaluate
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor).squeeze().numpy()
    y_class = (y_pred > 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, y_class))
print("AUC:", roc_auc_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_class))

# Plot training loss
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.show()
