import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader

from sklearn.model_selection import train_test_split

import json
import pandas as pd
import numpy as np
from tqdm import tqdm

## import the C++ optimizer
import sys
import os
# Go up one level to find the compiled .pyd extension
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import fused_graph_ops # type: ignore

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

from pathlib import Path
DATA_DIR = Path.cwd() / "data"

base_folder = Path.cwd() / "test"
base_name = "model_v5_optimized_test_{}"

counter = 1
while (base_folder / base_name.format(counter)).exists():
    counter += 1

SAVE_PATH = base_folder / base_name.format(counter)
SAVE_PATH.mkdir(parents=True, exist_ok=True)

# Hyperparameters -------------------------- #
train_ratio = 0.8
batch_size = 256
learning_rate = 0.0015
weight_decay = 1e-4
epochs = 200

target_weights = [0.90, 0.05, 0.05]
metal_reg_weight = 2.5
metal_pos_weight = 1.2

hidden_dim = 512
gnn_layers = 5
# ------------------------------------------ #

dataset = torch.load(DATA_DIR / "processed_graphs_v5.pt", weights_only=False)

# --- 1. JOINT STRATIFICATION KEY ---
# Create a key combining Metal/Semi (0/1) and Crystal System (0-6)
# Format: "MetalID_SystemID" (e.g., "1_6" for Cubic Metal)
strat_keys = [f"{int(d.y_metal.item())}_{int(d.u_cat.item())}" for d in dataset]


# --- 2. THE STRATIFIED SPLIT ---
train_idx, test_idx = train_test_split(
    range(len(dataset)),
    train_size=train_ratio,
    stratify=strat_keys, # Ensures identical structure/metal ratios in both sets
    random_state=42
)

train_dataset = [dataset[i] for i in train_idx]
test_dataset = [dataset[i] for i in test_idx]

# --- 3. BALANCED SAMPLING (FOR TRAINING ONLY) ---
# We recalculate weights for the training set only to ensure 50/50 Metal/Semi batches
train_metal_labels = np.array([int(d.y_metal.item()) for d in train_dataset])
class_sample_count = np.array([len(np.where(train_metal_labels == t)[0]) for t in np.unique(train_metal_labels)])
weight = 1. / np.sqrt(class_sample_count)
samples_weight = torch.from_numpy(weight[train_metal_labels]).double()

sampler = WeightedRandomSampler(samples_weight.tolist(), len(samples_weight))

# --- 4. LOADERS ---
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

# We use the same train_dataset but NO sampler and NO shuffle 
# so we can compare predictions to actuals in order.
train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
# We don't sample the test set; we want to see real-world performance
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

pos_weight = torch.tensor([metal_pos_weight]).to(device)

class CrystalGINE(nn.Module):
    def __init__(self, hidden_dim, gnn_layers, node_in_dim=7, edge_in_dim=1):
        super(CrystalGINE, self).__init__()
        
        self.group_embed = nn.Embedding(19, 32)
        self.period_embed = nn.Embedding(8, 32)
        self.valence_embed = nn.Embedding(16, 32)
        self.node_embed = nn.Linear(node_in_dim + 96, hidden_dim)
        
        self.edge_embed = nn.Sequential(
            nn.Linear(edge_in_dim, 32), nn.ReLU(),
            nn.Linear(32, hidden_dim), nn.BatchNorm1d(hidden_dim)
        ) 

        self.gine_mlps = nn.ModuleList([self.gine_mlp(hidden_dim) for _ in range(gnn_layers)])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(gnn_layers)])

        self.sys_embedding = nn.Embedding(7, 32)
        self.global_mlp = nn.Linear(2, 32)

        ## UPDATED FUSION LAYER ##
        self.shared_fusion = nn.Sequential(
            nn.Linear(hidden_dim + 32 + 32, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        # Head 1: Regression (Band Gap, Formation Energy)
        self.reg_head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 2) 
        )

        # Head 2: Classification (Is Metal)
        self.metal_head = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 1) # Outputting logits for BCEWithLogitsLoss
        )

    def gine_mlp(self, hidden_dim):
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
            nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        g_emb, p_emb, v_emb = self.group_embed(data.group), self.period_embed(data.period), self.valence_embed(data.valence)
        x = torch.cat([x, g_emb, p_emb, v_emb], dim=1)

        x = self.node_embed(x)
        edge_attr = self.edge_embed(edge_attr)
        num_nodes = x.shape[0]
        
        for mlp, bn in zip(self.gine_mlps, self.batch_norms):
            identity = x

            msg = x[edge_index[0]] + edge_attr
            x = fused_graph_ops.fused_aggregate(msg, edge_index, num_nodes)

            x = F.relu(bn(mlp(x)) + identity)

        graph_feat = global_mean_pool(x, data.batch)
        sys_feat = self.sys_embedding(data.u_cat).squeeze(1)
        global_feat = F.relu(self.global_mlp(data.u_cont)).squeeze(1)

        combined = torch.cat([graph_feat, sys_feat, global_feat], dim=1)
        shared = self.shared_fusion(combined)

        reg_out = self.reg_head(shared)
        metal_logits = self.metal_head(shared)
        
        return reg_out, metal_logits

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
    factor=0.6,
    patience=20,
    threshold=0.001,
    min_lr=1e-6
)

def multi_task_loss(reg_pred, metal_logits, target_reg, target_metal):
    # --- HUBER LOSS FOR BAND GAP ---
    # We use a delta of 1.0. 
    # This helps with the "overshooting" by dampening gradients from large outliers.
    mse_bg_raw = F.huber_loss(reg_pred[:, 0], target_reg[:, 0], delta=1.0, reduction='none')
    
    # --- SAMPLE WEIGHTING ---
    # We increase the penalty for errors on materials that are ACTUALLY metals (gap == 0)
    # This forces the model to respect the "0.0" boundary more.
    sample_weights = torch.ones_like(target_reg[:, 0])
    sample_weights[target_reg[:, 0] == 0] = metal_reg_weight # Extra penalty for missing a metal
    
    weighted_bg_loss = (mse_bg_raw * sample_weights).mean()
    
    # --- STANDARD MSE FOR FORMATION ENERGY ---
    mse_fe = F.mse_loss(reg_pred[:, 1], target_reg[:, 1])
    
    # --- BCE FOR METAL ---
    metal_loss = F.binary_cross_entropy_with_logits(
        metal_logits.view(-1), 
        target_metal, 
        pos_weight=pos_weight
    )
    
    return (target_weights[0] * weighted_bg_loss) + \
           (target_weights[1] * mse_fe) + \
           (target_weights[2] * metal_loss)

criterion = multi_task_loss

# Evaluation
def model_eval(loader):
    model.eval()
    all_reg_true, all_reg_pred = [], []
    all_metal_true, all_metal_logits = [], []
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            reg_out, metal_logits = model(batch)
            
            # 1. Collect targets and outputs
            all_reg_true.append(batch.y.cpu())
            all_reg_pred.append(reg_out.cpu())
            all_metal_true.append(batch.y_metal.cpu())
            all_metal_logits.append(metal_logits.cpu())

            # 2. Calculate batch loss
            # This allows us to track the "true" objective function on unseen data
            target_reg = batch.y
            target_metal = batch.y_metal.view(-1).float()

            loss = criterion(reg_out, metal_logits, target_reg, target_metal)
            total_loss += loss.item()
        
        # Concatenate everything for metric calculation
        all_reg_true = torch.cat(all_reg_true, dim=0)
        all_reg_pred = torch.cat(all_reg_pred, dim=0)
        all_metal_true = torch.cat(all_metal_true, dim=0).view(-1).float()
        all_metal_logits = torch.cat(all_metal_logits, dim=0).view(-1)

        # Slice results
        bg_true, bg_pred = all_reg_true[:, 0], all_reg_pred[:, 0]
        fe_true, fe_pred = all_reg_true[:, 1], all_reg_pred[:, 1]

        # Regression Metrics (BG & FE)
        bg_rmse = torch.sqrt(F.mse_loss(bg_pred, bg_true))
        bg_mae = torch.mean(torch.abs(bg_true - bg_pred))
        bg_r2 = 1 - (torch.sum((bg_true - bg_pred)**2) / torch.sum((bg_true - bg_true.mean())**2))

        fe_rmse = torch.sqrt(F.mse_loss(fe_pred, fe_true))
        fe_mae = torch.mean(torch.abs(fe_true - fe_pred))
        fe_r2 = 1 - (torch.sum((fe_true - fe_pred)**2) / torch.sum((fe_true - fe_true.mean())**2))

        # Classification Metrics (Metal)
        metal_probs = torch.sigmoid(all_metal_logits)
        metal_preds_binary = (metal_probs > 0.5).float()
        metal_acc = (metal_preds_binary == all_metal_true).sum().item() / len(all_metal_true)

        # Calculate R2 for true semiconductors to observe isolated R2
        semi_mask = all_reg_true[:, 0] > 0
        bg_r2_semi = torch.tensor(0.0).to(device)
        if semi_mask.sum() > 0:
            bg_true_semi = all_reg_true[semi_mask, 0]
            bg_pred_semi = all_reg_pred[semi_mask, 0]
            
            # Calculate R2 for ONLY semiconductors
            ss_res = torch.sum((bg_true_semi - bg_pred_semi)**2)
            ss_tot = torch.sum((bg_true_semi - bg_true_semi.mean())**2)
            bg_r2_semi = 1 - (ss_res / ss_tot)
 
        return {
            "loss": total_loss / len(loader), # Average Loss across batches
            "bg": {
                "rmse": bg_rmse.item(),
                "mae": bg_mae.item(),
                "r2": bg_r2.item(),
                "r2-semi": bg_r2_semi.item()
            },
            "fe": {
                "rmse": fe_rmse.item(),
                "mae": fe_mae.item(),
                "r2": fe_r2.item()
            },
            "metal": {
                "acc": metal_acc
            },
            "data": {
                "reg_pred": all_reg_pred,
                "reg_true": all_reg_true,
                "metal_logits": all_metal_logits,
                "metal_true": all_metal_true
            }
        }

all_results = []

model.train()
pbar = tqdm(range(1, epochs + 1), desc="Training CrystalGINE")

best_score = float('inf')
best_epoch = 0
patience_early_stop = 40
epochs_no_improve = 0
stopped_early = False

for epoch in pbar:
    for batch in train_loader:
        batch = batch.to(device)
        reg_out, metal_out = model(batch)

        # reg_pred, metal_logits, target_reg, target_metal
        loss = criterion(reg_out, metal_out, batch.y, batch.y_metal.view(-1).float())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    # Evaluation Step
    train_metrics = model_eval(train_eval_loader)
    test_metrics = model_eval(test_loader)
    
    epoch_results = {
        "epoch": epoch,
        "train_bg_rmse": train_metrics["bg"]['rmse'],
        "train_bg_mae": train_metrics["bg"]['mae'],
        "train_bg_r2": train_metrics["bg"]['r2'],
        "train_bg_r2-semi": train_metrics['bg']['r2-semi'],
        "train_fe_rmse": train_metrics["fe"]['rmse'],
        "train_fe_mae": train_metrics["fe"]['mae'],
        "train_fe_r2": train_metrics["fe"]['r2'],
        "train_metal_acc": train_metrics["metal"]["acc"],
        "test_bg_rmse": test_metrics["bg"]['rmse'],
        "test_bg_mae": test_metrics["bg"]['mae'],
        "test_bg_r2": test_metrics["bg"]['r2'],
        "test_bg_r2-semi": test_metrics["bg"]['r2-semi'],
        "test_fe_rmse": test_metrics["fe"]['rmse'],
        "test_fe_mae": test_metrics["fe"]['mae'],
        "test_fe_r2": test_metrics["fe"]['r2'],
        "test_metal_acc": test_metrics["metal"]["acc"],
        "loss": test_metrics["loss"]
    }
    all_results.append(epoch_results)

    # Calculate score
    unified_score = F.huber_loss(test_metrics['data']['reg_pred'][:, 0], test_metrics['data']['reg_true'][:, 0], delta=1.0).item()

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
        "Loss": f"{test_metrics['loss']:.4f}",
        "Tr_BG_R2": f"{train_metrics['bg']['r2']:.3f}",
        "Te_BG_R2": f"{test_metrics['bg']['r2']:.3f}",
        "Tr_BG_R2-Semi": f"{train_metrics['bg']['r2-semi']:.3f}",
        "Te_BG_R2-Semi": f"{test_metrics['bg']['r2-semi']:.3f}",
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
print(f"MAE: {metrics_train["fe"]["mae"]:.4f} eV | RMSE: {metrics_train["fe"]["rmse"]:.4f} eV | R2: {metrics_train["fe"]["r2"]:.4f}")
print(f"\n--- Metal Classification Evaluation ---")
print(f"Accuracy: {metrics_train["metal"]["acc"]:.4f}%\n\n")


metrics_test = model_eval(test_loader)
print("Test")
print(f"--- Band Gap Evaluation ---")
print(f"MAE: {metrics_test["bg"]["mae"]:.4f} eV | RMSE: {metrics_test["bg"]["rmse"]:.4f} eV | R2: {metrics_test["bg"]["r2"]:.4f}")
print(f"\n--- Formation Energy Evaluation ---")
print(f"MAE: {metrics_test["fe"]["mae"]:.4f} eV | RMSE: {metrics_test["fe"]["rmse"]:.4f} eV | R2: {metrics_test["fe"]["r2"]:.4f}")
print(f"\n--- Metal Classification Evaluation ---")
print(f"Accuracy: {metrics_test["metal"]["acc"]:.4f}%\n\n")

# Save Results --------------------------------- #

df = pd.DataFrame(all_results)
df.to_csv(SAVE_PATH / "test.csv", index=False)

metadata = {
    "train_size": train_ratio,
    "epochs": epochs,
    "batch_size": batch_size,
    "lr": learning_rate,
    "bg_weight": target_weights[0],
    "fe_weight": target_weights[1],
    "metal_weight": target_weights[2],
    "metal_reg_weight": metal_reg_weight,
    "metal_pos_weight": metal_pos_weight,
    "hidden_dim": hidden_dim,
    "gnn_layers": gnn_layers,
    "total_params": total_params,
    "trainable_params": trainable_params,
    "best_unified_score": best_score,
    "best_epoch": best_epoch, 
    "stopped_early": stopped_early,
    "note": "shifted to optimized model to obtain optimized model weights"
}

with open(SAVE_PATH / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

# --- Save Predictions vs Actuals for Analysis --- #
model.eval()
train_results = []
test_results = []

# Confusion Matrix Counters for Test Set
cm = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

with torch.no_grad():
    # Helper to process loaders
    def process_loader(loader, results_list, is_test=False):
        for batch in loader:
            batch = batch.to(device)
            reg_out, metal_logits = model(batch)

            # Move everything to CPU/Numpy
            y_reg_true = batch.y.cpu().numpy()
            y_reg_pred = reg_out.cpu().numpy()
            
            y_metal_true = batch.y_metal.cpu().numpy().flatten()
            # Convert logits to probability, then to 0/1 prediction
            metal_probs = torch.sigmoid(metal_logits).cpu().numpy().flatten()
            y_metal_pred = (metal_probs > 0.5).astype(int)
            
            for i in range(len(y_reg_true)):
                res = {
                    "bg_actual": y_reg_true[i, 0],
                    "bg_predicted": y_reg_pred[i, 0],
                    "fe_actual": y_reg_true[i, 1],
                    "fe_predicted": y_reg_pred[i, 1],
                    "metal_actual": int(y_metal_true[i]),
                    "metal_predicted": int(y_metal_pred[i]),
                    "metal_prob": float(metal_probs[i])
                }
                results_list.append(res)
                
                # If this is the test set, update our 2x2 counters
                if is_test:
                    actual = int(y_metal_true[i])
                    pred = int(y_metal_pred[i])
                    if actual == 1 and pred == 1: cm["tp"] += 1
                    elif actual == 0 and pred == 0: cm["tn"] += 1
                    elif actual == 0 and pred == 1: cm["fp"] += 1
                    elif actual == 1 and pred == 0: cm["fn"] += 1

    process_loader(train_eval_loader, train_results, is_test=False)
    process_loader(test_loader, test_results, is_test=True)

# Save to CSV
pd.DataFrame(test_results).to_csv(SAVE_PATH / "test_predictions_vs_actuals.csv", index=False)
pd.DataFrame(train_results).to_csv(SAVE_PATH / "train_predictions_vs_actuals.csv", index=False)

# Print 2x2 Confusion Matrix for Test Set
print("\n═══ Metal Classification: 2x2 Confusion Matrix (Test) ═══")
print(f"Total Samples: {len(test_results)}")
print(f"{'':<15} | Predicted Metal | Predicted Semi")
print("-" * 50)
print(f"{'Actual Metal':<15} | {cm['tp']:<15} | {cm['fn']:<15}")
print(f"{'Actual Semi':<15} | {cm['fp']:<15} | {cm['tn']:<15}")
print("═" * 53)

# Add confusion matrix to your metadata for safe keeping
metadata["metal_confusion_matrix"] = cm
with open(SAVE_PATH / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print(f"Saved predictions and updated metadata.")