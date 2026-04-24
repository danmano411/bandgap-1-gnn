#include <torch/extension.h>
#include <omp.h>

torch::Tensor fused_aggregate(torch::Tensor edge_features, torch::Tensor edge_index, int64_t num_nodes) {
    // Ensure data is in a predictable memory layout
    auto edge_feats_con = edge_features.contiguous();
    auto edge_idx_con = edge_index.contiguous();
    
    int64_t num_edges = edge_feats_con.size(0);
    int64_t feature_dim = edge_feats_con.size(1);

    // Initialize the output (Node-space)
    auto output = torch::zeros({num_nodes, feature_dim}, edge_feats_con.options());

    auto feat_acc = edge_feats_con.accessor<float, 2>();
    auto edge_acc = edge_idx_con.accessor<int64_t, 2>();
    auto out_acc = output.accessor<float, 2>();

    #pragma omp parallel for
    for (int64_t i = 0; i < num_edges; i++) {
        // We only need the destination row (row 1) from edge_index
        int64_t dst = edge_acc[1][i]; 
        
        // Safety check to prevent the "hanging" crash
        if (dst >= 0 && dst < num_nodes) {
            for (int64_t j = 0; j < feature_dim; j++) {
                #pragma omp atomic
                out_acc[dst][j] += feat_acc[i][j];
            }
        }
    }
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_aggregate", &fused_aggregate, "Fused Graph Aggregation (C++)");
}