import sys
import os
import torch
import time
import numpy as np

# Ensure the path to the compiled extension is correct
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import fused_graph_ops # type: ignore
except ImportError:
    print("Error: Could not import fused_graph_ops. Ensure it is compiled and in the path.")
    sys.exit(1)

def run_benchmark(num_nodes=100000, num_edges=500000, feat_dim=512, iterations=10):
    print(f"🚀 Benchmarking: {num_nodes} nodes, {num_edges} edges, {feat_dim} features")
    
    # 1. Prepare Data
    # Use contiguous() to ensure the best possible performance for C++ accessors
    x = torch.randn(num_nodes, feat_dim).float().contiguous()
    edge_index = torch.randint(0, num_nodes, (2, num_edges)).long().contiguous()

    # --- WARMUP PHASE ---
    # Essential to "spin up" OpenMP threads and load the binary into cache
    for _ in range(3):
        _ = fused_graph_ops.fused_aggregate(x[edge_index[0]], edge_index, num_nodes)
        _ = torch.zeros_like(x).index_add_(0, edge_index[1], x[edge_index[0]])

    # 2. Benchmark PyTorch Baseline
    pt_times = []
    for _ in range(iterations):
        # We pre-index here to match the logic used in our optimized model_v5
        msg = x[edge_index[0]] 
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        
        out_pt = torch.zeros(num_nodes, feat_dim).index_add_(0, edge_index[1], msg)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        pt_times.append(time.perf_counter() - start)
    
    avg_pt = np.mean(pt_times)
    print(f"✅ PyTorch Avg: {avg_pt:.6f}s")

    # 3. Benchmark Custom C++ Engine
    cpp_times = []
    for _ in range(iterations):
        msg = x[edge_index[0]]
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        
        out_cpp = fused_graph_ops.fused_aggregate(msg, edge_index, num_nodes)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        cpp_times.append(time.perf_counter() - start)

    avg_cpp = np.mean(cpp_times)
    print(f"✅ Custom C++ Avg: {avg_cpp:.6f}s")

    # 4. Final Validation
    match = torch.allclose(out_pt, out_cpp, atol=1e-5) #type: ignore
    speedup = avg_pt / avg_cpp
    
    print("-" * 30)
    print(f"Results Match: {match}")
    print(f"Final Speedup: {speedup:.2f}x")
    
    if speedup < 1.1 and num_nodes < 50000:
        print("💡 Tip: Try increasing nodes/edges. C++ overhead is more visible on small graphs.")

if __name__ == "__main__":
    # Use the scale where your previous test showed the best results
    run_benchmark(num_nodes=100000, num_edges=500000, feat_dim=512)