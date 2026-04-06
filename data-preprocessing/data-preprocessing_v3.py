import torch
import re
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from pymatgen.core.periodic_table import Element
from tqdm import tqdm
from pathlib import Path

DATA_DIR = Path.cwd() / "data"

# CONSTANTS & MAPPINGS
SYSTEM_MAP = {
    'triclinic': 0, 'monoclinic': 1, 'orthorhombic': 2, 
    'tetragonal': 3, 'trigonal': 4, 'hexagonal': 5, 'cubic': 6
}

# EXTRACT RAW PHYSICS (First Pass for Normalization)
print("Calculating global stats for 13k nodes...")
df = pd.read_pickle(DATA_DIR / "materials_dataset_v1.pkl")
print(df.shape)

all_z, all_mass, all_en, all_rad, all_ie, all_ea = [], [], [], [], [], []

struct_none = 0

for struct in tqdm(df['structure'], desc="Gathering node stats"):
    for site in struct:
        symbol = site.specie.symbol if hasattr(site.specie, "symbol") else str(site.specie)
        el = Element(symbol)

        all_z.append(float(el.Z))
        all_mass.append(float(el.atomic_mass))
        all_en.append(float(el.X))

        # en_raw = getattr(el, "X", None)
        # all_en.append(float(en_raw) if en_raw is not None else 2.0)

        rad_raw = getattr(el, "atomic_radius", None)
        all_rad.append(float(rad_raw) if rad_raw is not None else 1.5)

        # ie_list = getattr(el, "ionization_energies", [])
        # ie_val = float(ie_list[0]) if ie_list else 10.0
        # all_ie.append(float(ie_val))

        all_ie.append(float(el.ionization_energies[0]))

        # ea_raw = getattr(el, "electron_affinity", None)
        # all_ea.append(float(ea_raw) if ea_raw is not None else 0.0)

        all_ea.append(float(el.electron_affinity))

# Global stats for Z-score normalization
node_stats = {
    'z': {'mean': np.mean(all_z), 'std': np.std(all_z)},
    'mass': {'mean': np.mean(all_mass), 'std': np.std(all_mass)},
    'en': {'mean': np.mean(all_en), 'std': np.std(all_en)},
    'rad': {'mean': np.mean(all_rad), 'std': np.std(all_rad)},
    'ie': {'mean': np.mean(all_ie), 'std': np.std(all_ie)},
    'ea': {'mean': np.mean(all_ea), 'std': np.std(all_ea)}
}

global_stats = {
    'density': {'mean': df['density'].mean(), 'std': df['density'].std()},
    'nelements': {'mean': df['nelements'].mean(), 'std': df['nelements'].std()}
}

stats = {**node_stats, **global_stats}

def normalize(raw_value, key):
    mean, std = stats[key]['mean'], stats[key]['std']
    return (float(raw_value) - mean) / std

def get_baseline_valence_by_group(group):
    if group <= 2:
        return group
    elif group >= 13:
        return group - 10
    else:
        return 2 # The "Transition Baseline" hack

def get_physics_valence(structure_str, group):
    """
    Parses valence electrons while treating filled d10/f14 shells as core.
    Example: 
    Iron [Ar].3d6.4s2 -> 8
    Arsenic [Ar].3d10.4s2.4p3 -> 5
    """
    if (not isinstance(structure_str, str)) or ("[" not in structure_str):
        return get_baseline_valence_by_group(group)  # Default baseline
    
    # Extract everything after the core [NobleGas]
    after_core = structure_str.split(']')[-1]
    
    # Regex to find patterns like '3d6', '4s2', '4p3'
    # (\d) is the shell, ([spdf]) is the type, (\d+) is the count
    pattern = re.compile(r'(\d)([spdf])(\d+)')
    matches = pattern.findall(after_core)
    
    if not matches:
        return get_baseline_valence_by_group(group)

    corrected_valence = 0
    for shell, orbital_type, count in matches:
        num = int(count)
        
        # Rule: Skip fully filled d and f shells as they don't 
        # contribute to the Band Gap like valence electrons do.
        if (orbital_type == 'd' and num == 10) or (orbital_type == 'f' and num == 14):
            continue
        else:
            corrected_valence += num

    if corrected_valence == 0 and "4d10" in after_core:
        return 10

    # Return at least 1 (to avoid index 0 in embedding) and cap at 15
    return float(max(1, min(corrected_valence, 15)))

def structure_to_complex_graph(struct, row):
    # A. NODES & CHEMICAL INDICES
    node_feats = []
    groups = []
    periods = []
    valence = []
    
    # x group structure -> [atomic number, mass, EN, radius, ionization energy, electron affinity, is_ghost]
    # is_ghost is a binary column that is 0.0 for all nodes except for imaginary ghost node
    for site in struct:
        symbol = site.specie.symbol if hasattr(site.specie, "symbol") else str(site.specie)
        el = Element(symbol)
        
        z = normalize(el.Z, 'z')
        m = normalize(el.atomic_mass, 'mass')
        
        # e_raw = getattr(el, "X", None)
        # e = normalize(e_raw if e_raw is not None else stats['en']['mean'], 'en')
        e = normalize(el.X, 'en')
        
        r_raw = getattr(el, "atomic_radius", None)
        r = normalize(r_raw if r_raw is not None else stats['rad']['mean'], 'rad')

        ### New Data points

        # 1. Ionization Energy
        ie = normalize(el.ionization_energies[0], 'ie')

        # 2. Electron Affinity
        ea = normalize(el.electron_affinity, 'ea')

        ###
        
        node_feats.append([z, m, e, r, ie, ea, 0.0]) 
        groups.append(int(el.group))
        periods.append(int(el.row))
        valence.append(get_physics_valence(el.electronic_structure, int(el.group)))

    # B. ADD GHOST NODE
    num_real_nodes = len(node_feats)
    # Ghost node gets average features (0.0 after Z-score) and is_ghost = 1
    node_feats.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    groups.append(0) # The 'void' group
    periods.append(0) # The 'void' period
    valence.append(0) # The 'void' valence
    
    x = torch.tensor(node_feats, dtype=torch.float)
    group_tensor = torch.tensor(groups, dtype=torch.long)
    period_tensor = torch.tensor(periods, dtype=torch.long)
    valence_tensor = torch.tensor(valence, dtype=torch.long)

    # C. EDGES (Real + Ghost)
    edge_index = []
    edge_attr = []
    all_neighbors = struct.get_all_neighbors(r=5.0) # 5.0A max
    
    # Real Edges
    for i, neighbors in enumerate(all_neighbors):
        for nb in neighbors:
            edge_index.append([i, nb.index])
            edge_attr.append([nb.nn_distance / 5.0]) # Normalize by cut-off
            
    # Ghost Edges (Connect everything to the last index)
    ghost_idx = num_real_nodes
    for i in range(num_real_nodes):
        # Bi-directional ghost communication
        edge_index.append([i, ghost_idx])
        edge_index.append([ghost_idx, i])
        # Ghost edges get a unique distance tag (e.g., 0.0)
        edge_attr.append([0.0])
        edge_attr.append([0.0])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # D. GLOBALS & TARGETS
    u_cont = torch.tensor([[normalize(row['density'], 'density'), normalize(row['nelements'], 'nelements')]], dtype=torch.float)
    
    u_cat = torch.tensor([SYSTEM_MAP.get(row['crystal_system'].lower(), 0)], dtype=torch.long)
    y = torch.tensor([[row['band_gap'], row['formation_energy']]], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, 
                u_cont=u_cont, u_cat=u_cat, y=y, 
                group=group_tensor, period=period_tensor, valence=valence_tensor)

# Final Loop
processed_graphs = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Building Graphs"):
    g = structure_to_complex_graph(row['structure'], row)
    processed_graphs.append(g)
print(len(processed_graphs))
torch.save(processed_graphs, DATA_DIR / "processed_graphs_v4.pt")