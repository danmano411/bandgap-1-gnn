import torch
import sys
import os

# Ensure the path to the compiled extension is correct
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import fused_graph_ops # type: ignore

def test_optimized_logic():
    # 1. Setup a tiny graph: 3 nodes, 2 edges
    # Edge 0: Node 0 -> Node 1
    # Edge 1: Node 1 -> Node 2
    nodes = torch.tensor([[10.0], [20.0], [30.0]], dtype=torch.float32)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    num_nodes = nodes.size(0)

    print("--- Simulation of Optimized Model Logic ---")
    
    # STEP A: Pre-index messages (This happens in Python/PyTorch)
    # This creates a tensor of size (NumEdges, HiddenDim)
    # msg[0] is features of Node 0 (source of edge 0)
    # msg[1] is features of Node 1 (source of edge 1)
    messages = nodes[edge_index[0]] 
    print(f"Messages (Edge-space):\n{messages}")

    # STEP B: Run C++ Fused Aggregation
    # C++ now only looks at edge_index[1] to decide where to sum these messages
    output = fused_graph_ops.fused_aggregate(messages, edge_index, num_nodes)
    print(f"C++ Aggregated Result (Node-space):\n{output}")

    # STEP C: Verification
    # Node 0: Should be 0.0 (no edges point to it)
    # Node 1: Should be 10.0 (received message from Node 0)
    # Node 2: Should be 20.0 (received message from Node 1)
    expected = torch.tensor([[0.0], [10.0], [20.0]])
    
    if torch.allclose(output, expected):
        print("\n✅ Math is Perfect!")
        print("The C++ engine is correctly scattering edge-messages to node-destinations.")
    else:
        print("\n❌ Math Mismatch!")
        print(f"Expected:\n{expected}")

if __name__ == "__main__":
    test_optimized_logic()