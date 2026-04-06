import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

from pathlib import Path
DATA_DIR = Path.cwd() / "data"

# Hyperparameters -------------------------- #
batch_size = 64
learning_rate = 0.001
epochs = 20
hidden_dim = 256
# ------------------------------------------ #

dataset = torch.load(DATA_DIR / "processed_graphs_v1.pt", weights_only=False)
train_loader = DataLoader(dataset[:int(len(dataset)*0.8)], batch_size=batch_size, shuffle=True)
train_eval_loader = DataLoader(dataset[:int(len(dataset)*0.8)], batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset[int(len(dataset)*0.8):], batch_size=batch_size, shuffle=False)

class CrystalGNN(nn.Module):
    def __init__(self, node_in_dim=4, hidden_dim=64):
        super(CrystalGNN, self).__init__()

        # Processing the graph
        self.conv1 = GCNConv(node_in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        # Embedding layer
        self.sys_embedding = nn.Embedding(7, 8) # 7 vectors of length 8

        # Molecule global parameters (2 - > [Density, nelements])
        self.global_mlp = nn.Linear(2, 8)

        # Fusion (combine all layers)
        layers = [
            nn.Linear(hidden_dim + 8 + 8, 32), nn.ReLU(), # Only one of the embedding vectors will be used
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 2) # Outpute has 2 values [Band Gap, Formation Energy]
        ]
        self.fus = nn.Sequential(*layers)

    def forward(self, data):
        # Graph logic
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        # Individul features
        graph_feat = global_mean_pool(x, data.batch)
        sys_feat = self.sys_embedding(data.u_cat)
        global_feat = F.relu(self.global_mlp(data.u_cont)).squeeze(1)

        # Combine them
        combined = torch.cat([graph_feat, sys_feat, global_feat], dim=1)

        # Get the output
        return self.fus(combined)

model = CrystalGNN(hidden_dim=hidden_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Evaluation
def model_eval(loader):
    model.eval()
    with torch.no_grad():
        all_labels = []
        all_out = []

        for batch in loader:
            batch = batch.to(device)
            out = model(batch)

            all_labels.append(batch.y.cpu())
            all_out.append(out.cpu())
        
        all_labels = torch.cat(all_labels, dim=0)
        all_out = torch.cat(all_out, dim=0)

        # Slice the results (Column 0: Band Gap, Column 1: Formation Energy)
        bg_true = all_labels[:, 0]
        bg_pred = all_out[:, 0]

        fe_true = all_labels[:, 1]
        fe_pred = all_out[:, 1]

        # 1. Calculate Band Gap Metrics
        bg_mse = torch.mean((bg_true - bg_pred) ** 2)
        bg_rmse = torch.sqrt(bg_mse)
        bg_mae = torch.mean(torch.abs(bg_true - bg_pred))

        # 2. Calculate Formation Energy Metrics
        fe_mse = torch.mean((fe_true - fe_pred) ** 2)
        fe_rmse = torch.sqrt(fe_mse)
        fe_mae = torch.mean(torch.abs(fe_true - fe_pred))

        return bg_rmse, bg_mae, fe_rmse, fe_mae

model.train()
for epoch in range(1, epochs+1):
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        out = model(batch)

        loss = criterion(out, batch.y)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if epoch % 5 == 0:
        print(f"Epoch #{epoch}")

        bg_rmse, bg_mae, fe_rmse, fe_mae = model_eval(train_eval_loader)
        print("Train")
        print(f"--- Band Gap Evaluation ---")
        print(f"MAE: {bg_mae:.4f} eV | RMSE: {bg_rmse:.4f} eV")
        print(f"\n--- Formation Energy Evaluation ---")
        print(f"MAE: {fe_mae:.4f} eV/atom | RMSE: {fe_rmse:.4f} eV/atom")

        bg_rmse, bg_mae, fe_rmse, fe_mae = model_eval(test_loader)
        print("Test")
        print(f"--- Band Gap Evaluation ---")
        print(f"MAE: {bg_mae:.4f} eV | RMSE: {bg_rmse:.4f} eV")
        print(f"\n--- Formation Energy Evaluation ---")
        print(f"MAE: {fe_mae:.4f} eV/atom | RMSE: {fe_rmse:.4f} eV/atom")

        print()

print("Training Complete\n")
