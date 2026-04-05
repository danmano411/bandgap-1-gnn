import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool
from torch_geometric.loader import DataLoader

import json
import pandas as pd
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

from pathlib import Path
DATA_DIR = Path.cwd() / "data"

# Hyperparameters -------------------------- #
batch_size = 64
learning_rate = 0.005
epochs = 200
hidden_dim = 128
# ------------------------------------------ #

dataset = torch.load(DATA_DIR / "processed_graphs_v2.pt", weights_only=False)
train_loader = DataLoader(dataset[:int(len(dataset)*0.8)], batch_size=batch_size, shuffle=True)
train_eval_loader = DataLoader(dataset[:int(len(dataset)*0.8)], batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset[int(len(dataset)*0.8):], batch_size=batch_size, shuffle=False)

class CrystalGINE(nn.Module):
    def __init__(self, node_in_dim=4, edge_in_dim=1, hidden_dim=128):
        super(CrystalGINE, self).__init__()

        def gine_mlp():
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        
        self.node_embed = nn.Linear(node_in_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_in_dim, hidden_dim)

        # Processing the graph
        self.conv1 = GINEConv(gine_mlp())
        self.conv2 = GINEConv(gine_mlp())
        self.conv3 = GINEConv(gine_mlp())

        # Embedding layer
        self.sys_embedding = nn.Embedding(7, 16) # 7 vectors of length 8

        # Molecule global parameters (2 - > [Density, nelements])
        self.global_mlp = nn.Linear(2, 16)

        # Fusion (combine all layers)
        self.fus = nn.Sequential(
            nn.Linear(hidden_dim + 16 + 16, 64), 
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(64, 32), 
            nn.ReLU(),
            nn.Linear(32, 2) # Outpute has 2 values [Band Gap, Formation Energy]
        )

    def forward(self, data):
        # Graph logic
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.node_embed(x)
        edge_attr = self.edge_embed(edge_attr)

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = F.relu(self.conv3(x, edge_index, edge_attr))

        # Individul features
        graph_feat = global_mean_pool(x, data.batch)
        sys_feat = self.sys_embedding(data.u_cat)
        global_feat = F.relu(self.global_mlp(data.u_cont)).squeeze(1)

        # Combine them
        combined = torch.cat([graph_feat, sys_feat, global_feat], dim=1)

        # Get the output
        return self.fus(combined)

model = CrystalGINE(hidden_dim=hidden_dim).to(device)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"═══ Model Statistics ═══")
print(f"Total Parameters:     {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}")
print(f"════════════════════════")

# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(f"{name:20} | {param.numel():>10,}")

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=15,
)

def weighted_mse_loss(pred, target):
    # Weights: 0.7 for Band Gap, 0.3 for Formation Energy
    # Band gap is harder to learn, so we give it more "importance"
    weights = torch.tensor([0.7, 0.3]).to(device)
    loss = (pred - target) ** 2
    weighted_loss = loss * weights
    return weighted_loss.mean()

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
        bg_mean = torch.mean(bg_true)
        ss_tot_bg = torch.sum((bg_true - bg_mean) ** 2)
        ss_res_bg = torch.sum((bg_true - bg_pred) ** 2)
        bg_r2 = 1 - (ss_res_bg / ss_tot_bg)

        # 2. Calculate Formation Energy Metrics
        fe_mse = torch.mean((fe_true - fe_pred) ** 2)
        fe_rmse = torch.sqrt(fe_mse)
        fe_mae = torch.mean(torch.abs(fe_true - fe_pred))
        fe_mean = torch.mean(fe_true)
        ss_tot_fe = torch.sum((fe_true - fe_mean) ** 2)
        ss_res_fe = torch.sum((fe_true - fe_pred) ** 2)
        fe_r2 = 1 - (ss_res_fe / ss_tot_fe)
 
        return {
            "bg": {
                "rmse": bg_rmse.item(),
                "mae": bg_mae.item(),
                "r2": bg_r2.item()
            },
            "fe": {
                "rmse": fe_rmse.item(),
                "mae": fe_mae.item(),
                "r2": fe_r2.item()
            }
        }

history = {
    "test_bg": {"rmse": [], "mae": [], "r2": []},
    "test_fe": {"rmse": [], "mae": [], "r2": []},
    "train_loss": []
}

all_results = []

model.train()
pbar = tqdm(range(1, epochs + 1), desc="Training CrystalGINE")

for epoch in pbar:
    epoch_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        out = model(batch)

        loss = weighted_mse_loss(out, batch.y)
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    # Evaluation Step
    metrics = model_eval(test_loader)
    for key in ["rmse", "mae", "r2"]:
        history["test_bg"][key].append(metrics["bg"][key])
        history["test_fe"][key].append(metrics["fe"][key])
    
    epoch_results = {
        "epoch": epoch,
        "bg_rmse": metrics["bg"]['rmse'],
        "bg_mae": metrics["bg"]['mae'],
        "bg_r2": metrics["bg"]['r2'],
        "fe_rmse": metrics["fe"]['rmse'],
        "fe_mae": metrics["fe"]['mae'],
        "fe_r2": metrics["fe"]['r2'],
    }
    all_results.append(epoch_results)

    # Step the scheduler based on the Test MAE
    scheduler.step(metrics['bg']['mae'])
    
    # --- LIVE UPDATE ---
    # This updates the text to the right of the progress bar every epoch
    pbar.set_postfix({
        "Loss": f"{epoch_loss/len(train_loader):.4f}",
        "BG_MAE": f"{metrics['bg']['mae']:.3f}",
        "FE_MAE": f"{metrics['fe']['mae']:.3f}",
        "LR": f"{optimizer.param_groups[0]['lr']:.1e}"
    })

print("Training Complete\n")


# Conclusion ------------------------------------------- #

metrics_train = model_eval(train_eval_loader)
print("Train")
print(f"--- Band Gap Evaluation ---")
print(f"MAE: {metrics_train["bg"]["mae"]:.4f} eV | RMSE: {metrics_train["bg"]["rmse"]:.4f} eV | R2: {metrics_train["bg"]["r2"]:.4f}")
print(f"\n--- Formation Energy Evaluation ---")
print(f"MAE: {metrics_train["fe"]["mae"]:.4f} eV | RMSE: {metrics_train["fe"]["rmse"]:.4f} eV | R2: {metrics_train["fe"]["r2"]:.4f}\n\n")

metrics_test = model_eval(test_loader)
print("Test")
print(f"--- Band Gap Evaluation ---")
print(f"MAE: {metrics_test["bg"]["mae"]:.4f} eV | RMSE: {metrics_test["bg"]["rmse"]:.4f} eV | R2: {metrics_test["bg"]["r2"]:.4f}")
print(f"\n--- Formation Energy Evaluation ---")
print(f"MAE: {metrics_test["fe"]["mae"]:.4f} eV | RMSE: {metrics_test["fe"]["rmse"]:.4f} eV | R2: {metrics_test["fe"]["r2"]:.4f}\n")

# Save Reults --------------------------------- #

base_folder = Path.cwd() / "test"
base_name = "model_v2_test_{}"

counter = 1
while (base_folder / base_name.format(counter)).exists():
    counter += 1

SAVE_PATH = base_folder / base_name.format(counter)
SAVE_PATH.mkdir(parents=True, exist_ok=True)

df = pd.DataFrame(all_results)
df.to_csv(SAVE_PATH / "test.csv", index=False)

metadata = {
    "epochs": epochs,
    "batch_size": batch_size,
    "lr": learning_rate,
    "hidden_dim": hidden_dim,
    "total_params": total_params,
    "trainable_params": trainable_params,
    "note": "Switched to GINEConv with distance-aware edges"
}

with open(SAVE_PATH / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print(f"Saved")