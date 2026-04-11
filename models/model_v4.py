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

base_folder = Path.cwd() / "test"
base_name = "model_v4_test_{}"

counter = 1
while (base_folder / base_name.format(counter)).exists():
    counter += 1

SAVE_PATH = base_folder / base_name.format(counter)
SAVE_PATH.mkdir(parents=True, exist_ok=True)

# Hyperparameters -------------------------- #
batch_size = 256
learning_rate = 0.0015
weight_decay = 1e-4
# dropout = 0.4
epochs = 200

hidden_dim = 256
gnn_layers = 5
# ------------------------------------------ #

dataset = torch.load(DATA_DIR / "processed_graphs_v4.pt", weights_only=False)
train_loader = DataLoader(dataset[:int(len(dataset)*0.8)], batch_size=batch_size, shuffle=True)
train_eval_loader = DataLoader(dataset[:int(len(dataset)*0.8)], batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset[int(len(dataset)*0.8):], batch_size=batch_size, shuffle=False)

class CrystalGINE(nn.Module):
    def __init__(self, hidden_dim, gnn_layers, node_in_dim=7, edge_in_dim=1):
        super(CrystalGINE, self).__init__()
        
        ## Graph
        """
        1.) The group and period and valence electrons are embedded as vectors
        2.) These vectors are passed through a linear layer with the rest
        of the node's features
        3.) The edge features are passed through linear layers, and them two combined
        with the edge map as passed through the GNN as graph features
        """

        # Embed the groups and periods and valence electrons
        self.group_embed = nn.Embedding(19, 32)
        self.period_embed = nn.Embedding(8, 32)
        self.valence_embed = nn.Embedding(16, 32) # 0 (void case) - 15 (4f13 6s2)

        # Nodes and Edges
        self.node_embed = nn.Linear(node_in_dim + 32 + 32 + 32, hidden_dim) # All node characteristics
        
        self.edge_embed = nn.Sequential(
            nn.Linear(edge_in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        ) 

        # Processing the graphs
        def gine_mlp():
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        
        self.convs = nn.ModuleList([GINEConv(gine_mlp()) for _ in range(gnn_layers)])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(gnn_layers)])

        ## Global features 
        """
        1.) The crystal shape is embedded as a vector
        2.) The two global characteristics go through a layer
        """
        self.sys_embedding = nn.Embedding(7, 32) # 7 crystal vectors
        self.global_mlp = nn.Linear(2, 32) # (2 - > [Density, nelements])

        ## Fusion (combine all layers)
        """
        Graph layer, crystal shape layer, global mlp layer
        """
        self.fus = nn.Sequential(
            nn.Linear(hidden_dim + 32 + 32, 256), 
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(256, 128), 
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            
            nn.Linear(64, 32), 
            nn.ReLU(),
            
            nn.Linear(32, 2)
        )

    def forward(self, data):
        # Graph logic
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        g_emb = self.group_embed(data.group)
        p_emb = self.period_embed(data.period)
        v_emb = self.valence_embed(data.valence)
        x = torch.cat([x, g_emb, p_emb, v_emb], dim=1)

        x = self.node_embed(x)
        edge_attr = self.edge_embed(edge_attr)
        
        for conv, bn in zip(self.convs, self.batch_norms):
            # Store the "identity" before the convolution so node features
            # do not get as homogenized through many GNN layers
            identity = x
            x = conv(x, edge_index, edge_attr)

            # Normalize layers to keep them "centered" over layers
            x = bn(x)
            x = F.relu(x + identity) # The Residual Step

        # Individual features
        graph_feat = global_mean_pool(x, data.batch)
        sys_feat = self.sys_embedding(data.u_cat)
        global_feat = F.relu(self.global_mlp(data.u_cont)).squeeze(1)

        # Combine them
        combined = torch.cat([graph_feat, sys_feat, global_feat], dim=1)

        # Get the output
        return self.fus(combined)

model = CrystalGINE(hidden_dim=hidden_dim, gnn_layers=gnn_layers).to(device)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"═══ Model Statistics ═══")
print(f"Total Parameters:     {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}")
print(f"════════════════════════")

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=15,
    min_lr=1e-6
)

def weighted_mse_loss(pred, target):
    # Band gap is harder to learn, so we give it more "importance"
    weights = torch.tensor([0.85, 0.15]).to(device)
    loss = (pred - target) ** 2
    weighted_loss = loss * weights
    return weighted_loss.mean()

criterion = weighted_mse_loss

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

all_results = []

model.train()
pbar = tqdm(range(1, epochs + 1), desc="Training CrystalGINE")

best_score = float('inf')
best_epoch = 0
patience_early_stop = 30
epochs_no_improve = 0
stopped_early = False

for epoch in pbar:
    epoch_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        out = model(batch)

        loss = criterion(out, batch.y)
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    # Evaluation Step
    train_metrics = model_eval(train_eval_loader)
    test_metrics = model_eval(test_loader)
    avg_loss = epoch_loss/len(train_loader)
    
    epoch_results = {
        "epoch": epoch,
        "train_bg_rmse": train_metrics["bg"]['rmse'],
        "train_bg_mae": train_metrics["bg"]['mae'],
        "train_bg_r2": train_metrics["bg"]['r2'],
        "train_fe_rmse": train_metrics["fe"]['rmse'],
        "train_fe_mae": train_metrics["fe"]['mae'],
        "train_fe_r2": train_metrics["fe"]['r2'],
        "test_bg_rmse": test_metrics["bg"]['rmse'],
        "test_bg_mae": test_metrics["bg"]['mae'],
        "test_bg_r2": test_metrics["bg"]['r2'],
        "test_fe_rmse": test_metrics["fe"]['rmse'],
        "test_fe_mae": test_metrics["fe"]['mae'],
        "test_fe_r2": test_metrics["fe"]['r2'],
        "loss": avg_loss
    }
    all_results.append(epoch_results)

    test_bg_rmse = test_metrics['bg']['rmse']
    test_fe_rmse = test_metrics['fe']['rmse']
    unified_score = 0.85 * test_bg_rmse + 0.15 * test_fe_rmse # Both already in the same units

    # Step the scheduler based on the Band Gap Test MAE
    scheduler.step(unified_score)

    if unified_score < best_score:
        best_score = unified_score
        best_epoch = epoch
        epochs_no_improve = 0
        torch.save(model.state_dict(), SAVE_PATH / "best_model.pth")
    else:
        epochs_no_improve += 1 # No improvement, increment counter

    if epochs_no_improve >= patience_early_stop:
        stopped_early = True
        print(f"\n--- Early Stopping Triggered at Epoch {epoch} ---")
        print(f"No improvement for {patience_early_stop} epochs. Best epoch was {best_epoch}.")
        break # Exit the training loop early
    
    # --- LIVE UPDATE ---
    # This updates the text to the right of the progress bar every epoch
    pbar.set_postfix({
        "Loss": f"{avg_loss:.4f}",
        "TRAIN_BG_RMSE": f"{train_metrics['bg']['rmse']:.3f}",
        "TEST_BG_RMSE": f"{test_bg_rmse:.3f}",
        "LR": f"{optimizer.param_groups[0]['lr']:.1e}"
    })

torch.save(model.state_dict(), SAVE_PATH / "final_model.pth")
print("Training Complete\n")


# Conclusion ------------------------------------------- #
model.load_state_dict(torch.load(SAVE_PATH / "best_model.pth"))

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

# Save Results --------------------------------- #

df = pd.DataFrame(all_results)
df.to_csv(SAVE_PATH / "test.csv", index=False)

metadata = {
    "epochs": epochs,
    "batch_size": batch_size,
    "lr": learning_rate,
    "hidden_dim": hidden_dim,
    "gnn_layers": gnn_layers,
    "total_params": total_params,
    "trainable_params": trainable_params,
    "best_unified_score": best_score,
    "best_epoch": best_epoch, 
    "stopped_early": stopped_early,
    "note": "shifted weights, widened tail, lowered dropout, increased lr"
}

with open(SAVE_PATH / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)


# --- Save Predictions vs Actuals for Analysis --- #
model.eval()
train_results = []
test_results = []

with torch.no_grad():
    for batch in train_eval_loader:
        batch = batch.to(device)
        out = model(batch)

        # Move to CPU and convert to numpy
        y_true = batch.y.cpu().numpy()
        y_pred = out.cpu().numpy()
        
        for i in range(len(y_true)):
            train_results.append({
                "bg_actual": y_true[i, 0],
                "bg_predicted": y_pred[i, 0],
                "fe_actual": y_true[i, 1],
                "fe_predicted": y_pred[i, 1]
            })

    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch)
        
        # Move to CPU and convert to numpy
        y_true = batch.y.cpu().numpy()
        y_pred = out.cpu().numpy()
        
        for i in range(len(y_true)):
            test_results.append({
                "bg_actual": y_true[i, 0],
                "bg_predicted": y_pred[i, 0],
                "fe_actual": y_true[i, 1],
                "fe_predicted": y_pred[i, 1]
            })

# Save to CSV
df_test_preds = pd.DataFrame(test_results)
df_test_preds.to_csv(SAVE_PATH / "test_predictions_vs_actuals.csv", index=False)

df_train_preds = pd.DataFrame(train_results) 
df_train_preds.to_csv(SAVE_PATH / "train_predictions_vs_actuals.csv", index=False)

print(f"Saved")