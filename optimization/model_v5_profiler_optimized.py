import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity

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

# Hyperparameters -------------------------- #
batch_size = 256
learning_rate = 0.0015
weight_decay = 1e-4

target_weights = [0.90, 0.05, 0.05]
metal_reg_weight = 2.5
metal_pos_weight = 1.2

hidden_dim = 512
gnn_layers = 5

# ------------------------------------------ #

dataset = torch.load(DATA_DIR / "processed_graphs_v5.pt", weights_only=False)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

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
        pos_weight=torch.tensor([metal_pos_weight]).to(device)
    )
    
    return (target_weights[0] * weighted_bg_loss) + \
           (target_weights[1] * mse_fe) + \
           (target_weights[2] * metal_loss)

criterion = multi_task_loss

all_results = []

with profile(
    activities=[ProfilerActivity.CPU],
    schedule=torch.profiler.schedule(wait=1, warmup=2, active=15, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/matgine_profile'),
    record_shapes=True,
    with_stack=True
) as prof:

    model.train()
    for i, batch in enumerate(train_loader):
        # Stop after 'active' steps in the schedule are complete
        if i >= 18: break
        
        batch = batch.to(device)
        
        with record_function("model_forward"):
            reg_out, metal_logits = model(batch)
            
        with record_function("loss_computation"):
            # Simplified loss for profiling overhead
            loss_reg = F.mse_loss(reg_out, batch.y)
            loss_metal = F.binary_cross_entropy_with_logits(metal_logits.view(-1), batch.y_metal.view(-1).float())
            loss = loss_reg + loss_metal

        with record_function("model_backward"):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        prof.step()
        print(f"Step {i} completed")

# --- Final Analysis Table ---
print("\n" + "="*50)
print("PROFILING COMPLETE: TOP CPU OPERATIONS")
print("="*50)
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))

# aten:scatter_add_ and aten::gather are the most CPU intensive other than aten::mm
# optimization needs to be done on the graph components of the NN, gathering nearby node data and aggregating them efficiently