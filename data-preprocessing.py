import torch
import pandas as pd
from torch_geometric.data import Data
from tqdm import tqdm

from pathlib import Path
DATA_DIR = Path.cwd() / "data"

# CONSTANTS & MAPPINGS
SYSTEM_MAP = {
    'triclinic': 0, 'monoclinic': 1, 'orthorhombic': 2, 
    'tetragonal': 3, 'trigonal': 4, 'hexagonal': 5, 'cubic': 6
}

# Approximate physical limits for normalization
# We use these to scale features between 0 and 1
SCALING = {
    'z': {'min': 1, 'max': 118},
    'mass': {'min': 1.0, 'max': 294.0},
    'en': {'min': 0.7, 'max': 4.0},      # Pauling Electronegativity
    'radius': {'min': 0.2, 'max': 3.0}   # Atomic radius in Angstroms
}

def get_node_features(struct):
    """
    Extracts and normalizes intrinsic atomic properties for every atom.
    Returns a tensor of shape (N, 4)
    """
    node_feats = []
    for site in struct:
        # 1. Extract raw values
        z = float(site.specie.Z)
        mass = float(site.specie.atomic_mass)
        en = float(site.specie.X) if getattr(site.specie, "X", None) is not None else 2.0
        radius = float(site.specie.atomic_radius) if getattr(site.specie, "atomic_radius", None) is not None else 1.5
        
        # 2. Min-Max Scaling (Normalizing to 0-1 range)
        z_norm = (z - SCALING['z']['min']) / (SCALING['z']['max'] - SCALING['z']['min'])
        mass_norm = (mass - SCALING['mass']['min']) / (SCALING['mass']['max'] - SCALING['mass']['min'])
        en_norm = (en - SCALING['en']['min']) / (SCALING['en']['max'] - SCALING['en']['min'])
        rad_norm = (radius - SCALING['radius']['min']) / (SCALING['radius']['max'] - SCALING['radius']['min'])
        
        node_feats.append([z_norm, mass_norm, en_norm, rad_norm])
        
    return torch.tensor(node_feats, dtype=torch.float)

def structure_to_graph(struct, density, nelements, system_str, target_bg, target_fe, stats):
    # A. NODES (N, 4)
    x = get_node_features(struct)

    # B. EDGES (2, E)
    # 5.0A is standard for capturing the first and second coordination shells
    all_neighbors = struct.get_all_neighbors(r=5.0)

    # edge_index = []
    # for i, neighbors in enumerate(all_neighbors):
    #     for nb in neighbors:
    #         edge_index.append([i, nb.index])

    # Inside structure_to_graph in File 2:
    edge_index = []
    edge_attr = [] # NEW: Edge features
    for i, neighbors in enumerate(all_neighbors):
        for nb in neighbors:
            edge_index.append([i, nb.index])
            # Get distance and normalize (assuming 5A max)
            dist = nb.nn_distance / 5.0 
            edge_attr.append([dist])

    if not edge_index: # Handle edge case for isolated atoms
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # C. GLOBAL MULTIMODAL FEATURES
    # Normalize continuous globals (Approximates)
    norm_density = (density - stats['density_mean']) / stats['density_std']
    norm_nelem = (nelements - stats['nelements_mean']) / stats['nelements_std']
    u_cont = torch.tensor([[norm_density, norm_nelem]], dtype=torch.float)
    
    # Categorical Crystal System
    sys_idx = SYSTEM_MAP.get(system_str.lower(), 0)
    u_cat = torch.tensor([sys_idx], dtype=torch.long)

    # D. TARGETS (1, 2)
    y = torch.tensor([[target_bg, target_fe]], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, u_cont=u_cont, u_cat=u_cat, y=y)

print("Loading raw dataframe...")
df = pd.read_pickle(DATA_DIR / "materials_dataset_v1.pkl")

stats = {
    'density_mean': df['density'].mean(),
    'density_std': df['density'].std(),
    'nelements_mean': df['nelements'].mean(),
    'nelements_std': df['nelements'].std()
}

processed_graphs = []

print("Converting structures to Multimodal Graphs...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    try:
        graph = structure_to_graph(
            struct=row['structure'],
            density=row['density'],
            nelements=row['nelements'],
            system_str=row['crystal_system'],
            target_bg=row['band_gap'],
            target_fe=row['formation_energy'],
            stats=stats
        )
        processed_graphs.append(graph)
    except Exception as e:
        continue # Skip corrupted entries

torch.save(processed_graphs, DATA_DIR / "processed_graphs_v2.pt")
print(f"Saved {len(processed_graphs)} graphs")