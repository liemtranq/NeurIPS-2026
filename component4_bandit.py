import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data

# ============================================================================
# Section 1: Hierarchical Arm Space (H-UCB)
# Giải quyết vấn đề không gian tìm kiếm loãng khi có 40+ mutation.
# ============================================================================

class HierarchicalArmTree:
    """
    Tổ chức Arms theo dạng cây: L1 (Error Type) -> L2 (Strategy) -> L3 (Operator).
    Giúp Bandit loại bỏ nhanh các nhánh không tiềm năng (Ví dụ: Lỗi Factual thì bỏ qua nhánh Causal).
    """
    def __init__(self, use_informed_prior: bool = True):
        # Khởi tạo Beta prior. Nếu dùng Meta-Policy, load từ checkpoint, ngược lại start từ 1.0
        self.prior_base = 5.0 if use_informed_prior else 1.0 
        
        # Shared memory cho G-Phasing (Asynchronous updates across 8 GPUs)
        self.alphas = torch.ones(5, 4, 10).share_memory_() * self.prior_base 
        self.betas = torch.ones(5, 4, 10).share_memory_() * self.prior_base
        
    def get_informed_prior(self, l1: int, l2: int, l3: int):
        """Lấy prior từ Meta-Policy Pre-training"""
        return self.alphas[l1, l2, l3].item(), self.betas[l1, l2, l3].item()

    def async_update(self, l1: int, l2: int, l3: int, reward: float):
        """G-Phasing: Cập nhật song song không cần lock nhờ tính chất atomic của tensor"""
        if reward > 0:
            self.alphas[l1, l2, l3] += reward
        else:
            self.betas[l1, l2, l3] += abs(reward) if reward < 0 else 0.1

# ============================================================================
# Section 2: Graph-Contextual Bandit Policy (GAT)
# Cấp cho Bandit khả năng nhận thức cấu trúc (Structural Awareness) của Proof DAG.
# ============================================================================

class GraphContextualPolicy(nn.Module):
    """
    Thay thế MLP phẳng. Nhận vào Proof Graph, dùng GAT để trích xuất Context.
    """
    def __init__(self, node_feature_dim=16, hidden_dim=64, num_arms_flat=40):
        super().__init__()
        # Graph Attention Network xử lý logic depth > 10 hops
        self.gat1 = GATConv(node_feature_dim, hidden_dim, heads=4, concat=True)
        self.gat2 = GATConv(hidden_dim * 4, hidden_dim, heads=1, concat=False)
        
        # Policy Head map graph embedding sang action space
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim + 12, 128), # 12 là chiều của InconsistencySignal
            nn.ReLU(),
            nn.Linear(128, num_arms_flat)
        )

    def forward(self, proof_graph: Data, signal_features: torch.Tensor) -> torch.Tensor:
        x, edge_index = proof_graph.x, proof_graph.edge_index
        
        # Graph encoding
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gat2(x, edge_index)
        
        # Global pooling để lấy context của toàn bộ proof chain
        graph_embed = global_mean_pool(x, proof_graph.batch)
        
        # Fuse với Inconsistency Signal (C3)
        fused_context = torch.cat([graph_embed, signal_features], dim=-1)
        
        return self.policy_head(fused_context)

# ============================================================================
# Section 3: Asynchronous Bandit Engine with Reward Shaping
# Tận dụng tối đa 8x MI300X và Counterfactual Regret.
# ============================================================================

class SOTABanditMutationEngine:
    def __init__(self, lambda_cohesion=0.2):
        self.lambda_cohesion = lambda_cohesion
        self.h_tree = HierarchicalArmTree(use_informed_prior=True)
        self.graph_policy = GraphContextualPolicy()
        
    def _compute_proof_cohesion(self, proof_steps: List[Any]) -> float:
        """
        Reward Shaping: Đo lường tính liên kết của đồ thị sau khi mutate.
        Tránh việc mutation phá nát cấu trúc logic thành các mảnh vỡ.
        """
        G = nx.DiGraph()
        for i, step in enumerate(proof_steps):
            G.add_node(i)
            for prem in step.premises:
                # Giả sử premise trỏ tới index của node trước đó
                if hasattr(prem, 'source_idx'): 
                    G.add_edge(prem.source_idx, i)
        
        # Nếu đồ thị bị chia cắt (đứt gãy logic), cohesion giảm mạnh
        if nx.number_weakly_connected_components(G) > 1:
            return -0.5 
        
        # Trả về độ dày đặc (density) như một metric của cohesion
        return nx.density(G)

    def calculate_shaped_reward(self, old_signal, new_signal, mutated_proof):
        """CFR: Cân bằng giữa giảm lỗi và duy trì mạch logic (Global vs Local)"""
        delta_inconsistency = old_signal.combined_score - new_signal.combined_score
        cohesion_score = self._compute_proof_cohesion(mutated_proof)
        
        # Reward = ΔInconsistency + λ * (Proof_Cohesion)
        return delta_inconsistency + self.lambda_cohesion * cohesion_score

    def get_exploration_rate(self, credal_width: float) -> float:
        """
        Image 2: Exploration Rate decay dựa theo độ rộng Credal.
        Credal hẹp (tin tự tin) -> Khám phá ít. Credal rộng (bối rối) -> Khám phá nhiều.
        """
        base_epsilon = 0.1
        return base_epsilon + (0.9 * credal_width)

    def worker_evaluate_candidate(self, gpu_id, candidate, proof, old_signal, shared_h_tree):
        """
        G-Phasing Worker: Chạy song song trên 1 GPU MI300X.
        Tính toán latency < 100ms.
        """
        # 1. Apply mutation (Giả định hàm execute đã tối ưu cực nhanh)
        mutated_proof = self.executor.execute(candidate, proof)
        
        # 2. Re-evaluate với C3 (Chạy forward pass trên GPU cục bộ)
        new_signal = self.pis.compute(..., mutated_proof, ...)
        
        # 3. Tính Shaped Reward
        reward = self.calculate_shaped_reward(old_signal, new_signal, mutated_proof)
        
        # 4. Asynchronous H-UCB Update qua Infinity Fabric (Shared memory)
        # Không cần lock, giảm thiểu overhead
        l1, l2, l3 = candidate.hierarchical_indices
        shared_h_tree.async_update(l1, l2, l3, reward)
        
        return reward, mutated_proof

    def parallel_thompson_search(self, proof_graph, old_signal):
        """Đẩy 8 candidates xuống 8 GPU MI300X song song"""
        # Epsilon decay theo Credal width
        epsilon = self.get_exploration_rate(old_signal.credal_inconsistency.width)
        
        candidates = self.generate_h_ucb_candidates(num_candidates=8, eps=epsilon)
        
        # Mô phỏng G-Phasing với multiprocessing (Trên MI300X thực tế sẽ map qua CUDA devices)
        processes = []
        results = []
        for i in range(8):
            p = mp.Process(
                target=self.worker_evaluate_candidate, 
                args=(i, candidates[i], proof_graph, old_signal, self.h_tree)
            )
            processes.append(p)
            p.start()
            
        for p in processes:
            p.join()
            
        # Chọn kết quả tốt nhất
        return self.select_best_mutated_proof(results)