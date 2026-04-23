import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
import logging

logger = logging.getLogger("mc_nsr.bandit")

# ============================================================================
# Section 1: Hierarchical Arm Space (H-UCB)
# ============================================================================

class HierarchicalArmTree:
    """
    Tổ chức Arms theo dạng cây: L1 (Error Type) -> L2 (Strategy) -> L3 (Operator).
    """
    def __init__(self, use_informed_prior: bool = True):
        self.prior_base = 5.0 if use_informed_prior else 1.0

        # Shared memory cho G-Phasing (Asynchronous updates across 8 GPUs)
        self.alphas = torch.ones(5, 4, 10).share_memory_() * self.prior_base
        self.betas  = torch.ones(5, 4, 10).share_memory_() * self.prior_base

    def get_informed_prior(self, l1: int, l2: int, l3: int):
        return self.alphas[l1, l2, l3].item(), self.betas[l1, l2, l3].item()

    def async_update(self, l1: int, l2: int, l3: int, reward: float):
        """G-Phasing: Cập nhật song song không cần lock."""
        if reward > 0:
            self.alphas[l1, l2, l3] += reward
        else:
            self.betas[l1, l2, l3] += abs(reward) if reward < 0 else 0.1


# ============================================================================
# Section 2: Graph-Contextual Bandit Policy (GAT)
# ============================================================================

class GraphContextualPolicy(nn.Module):
    """
    Thay thế MLP phẳng. Nhận vào Proof Graph, dùng GAT để trích xuất Context.
    """
    def __init__(self, node_feature_dim=16, hidden_dim=64, num_arms_flat=40):
        super().__init__()
        self.gat1 = GATConv(node_feature_dim, hidden_dim, heads=4, concat=True)
        self.gat2 = GATConv(hidden_dim * 4, hidden_dim, heads=1, concat=False)

        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim + 12, 128),
            nn.ReLU(),
            nn.Linear(128, num_arms_flat)
        )

    def forward(self, proof_graph: Data, signal_features: torch.Tensor) -> torch.Tensor:
        x, edge_index = proof_graph.x, proof_graph.edge_index

        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gat2(x, edge_index)

        graph_embed = global_mean_pool(x, proof_graph.batch)
        fused_context = torch.cat([graph_embed, signal_features], dim=-1)
        return self.policy_head(fused_context)


# ============================================================================
# Section 2b: Mutation Operators (nn.ModuleDict)
# [Fix 1] Đây là nơi phát sinh lỗi .get() — được phong tỏa hoàn toàn
# ============================================================================

class RephraseMutationOp(nn.Module):
    """Rephrase: Biến đổi ngôn ngữ của proof step mà giữ nguyên logic."""
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.transform = nn.Linear(hidden_dim, hidden_dim, dtype=torch.bfloat16)

    def forward(self, proof_steps: List[Any]) -> List[Any]:
        # Stub: Production thay bằng T5/LLM paraphrase call
        return proof_steps


class StructuralMutationOp(nn.Module):
    """Structural: Sắp xếp lại thứ tự các bước suy luận."""
    def __init__(self):
        super().__init__()

    def forward(self, proof_steps: List[Any]) -> List[Any]:
        if len(proof_steps) > 1:
            return proof_steps[1:] + [proof_steps[0]]  # Rotate
        return proof_steps


class EvidenceMutationOp(nn.Module):
    """Evidence: Thay thế evidence yếu bằng evidence mạnh hơn từ retriever."""
    def __init__(self):
        super().__init__()

    def forward(self, proof_steps: List[Any]) -> List[Any]:
        # Stub: Production gọi lại retriever với negative constraint
        return proof_steps


class MutationOperatorRegistry:
    """
    [Fix 1] Bọc nn.ModuleDict trong một class với _safe_get.
    Mọi truy cập operator đều đi qua đây — KHÔNG bao giờ gọi .get() trực tiếp.
    """
    def __init__(self, hidden_dim: int = 256):
        # [Fix 1] nn.ModuleDict không hỗ trợ .get(key, default)
        # Tất cả truy cập phải qua _safe_get()
        self._ops = nn.ModuleDict({
            "rephrase":   RephraseMutationOp(hidden_dim),
            "structural": StructuralMutationOp(),
            "evidence":   EvidenceMutationOp(),
        })

    @staticmethod
    def _safe_get(module_dict: nn.ModuleDict, key: str) -> Optional[nn.Module]:
        """
        [Fix 1] An toàn khi truy cập nn.ModuleDict.

        Sai:  op = self._ops.get("rephrase", None)    # Không hỗ trợ — crash hoặc sai kết quả
        Đúng: op = MutationOperatorRegistry._safe_get(self._ops, "rephrase")
        """
        return self._ops[key] if key in self._ops else None

    def get(self, strategy: str) -> Optional[nn.Module]:
        """Public interface — luôn dùng cái này, không truy cập self._ops trực tiếp."""
        op = self._safe_get(self._ops, strategy)
        if op is None:
            logger.warning(f"[MutationOperatorRegistry] Unknown strategy '{strategy}'. Falling back to 'structural'.")
            op = self._safe_get(self._ops, "structural")
        return op

    def list_strategies(self) -> List[str]:
        return list(self._ops.keys())


# ============================================================================
# Section 3: Asynchronous Bandit Engine with Reward Shaping
# ============================================================================

class SOTABanditMutationEngine:
    def __init__(self, lambda_cohesion: float = 0.2, hidden_dim: int = 256):
        self.lambda_cohesion = lambda_cohesion
        self.h_tree = HierarchicalArmTree(use_informed_prior=True)
        self.graph_policy = GraphContextualPolicy()

        # [Fix 1] Operator registry — tập trung toàn bộ ModuleDict access vào đây
        self.operator_registry = MutationOperatorRegistry(hidden_dim=hidden_dim)

    def _compute_proof_cohesion(self, proof_steps: List[Any]) -> float:
        """
        Reward Shaping: Đo lường tính liên kết của đồ thị sau khi mutate.
        """
        G = nx.DiGraph()
        for i, step in enumerate(proof_steps):
            G.add_node(i)
            if hasattr(step, 'premises'):
                for prem in step.premises:
                    if hasattr(prem, 'source_idx'):
                        G.add_edge(prem.source_idx, i)

        if nx.number_weakly_connected_components(G) > 1:
            return -0.5

        return nx.density(G)

    def calculate_shaped_reward(self, old_signal, new_signal, mutated_proof: List[Any]) -> float:
        """CFR: Cân bằng giữa giảm lỗi và duy trì mạch logic."""
        delta_inconsistency = old_signal.combined_score - new_signal.combined_score
        cohesion_score = self._compute_proof_cohesion(mutated_proof)
        return delta_inconsistency + self.lambda_cohesion * cohesion_score

    def get_exploration_rate(self, credal_width: float) -> float:
        """Exploration Rate decay dựa theo độ rộng Credal."""
        base_epsilon = 0.1
        return base_epsilon + (0.9 * credal_width)

    def mutate(
        self,
        signal: Any,
        proof_steps: List[Any],
        evidence: Any,
        strategy: str = "structural"
    ) -> Tuple[List[Any], float]:
        """
        Public API được gọi từ Master Orchestrator.
        Trả về (mutated_proof, reward_estimate).

        [Fix 1] Dùng operator_registry.get() — không bao giờ gọi ModuleDict.get() trực tiếp.
        """
        op = self.operator_registry.get(strategy)

        if op is None:
            # Không có operator nào — trả về proof gốc, reward 0
            logger.error(f"[mutate] No operator found for strategy '{strategy}'. Returning original proof.")
            return proof_steps, 0.0

        try:
            mutated = op(proof_steps)
        except Exception as e:
            logger.error(f"[mutate] Operator '{strategy}' raised: {e}", exc_info=True)
            return proof_steps, 0.0

        # Reward estimate đơn giản cho stub — production dùng shaped reward với C3
        reward_estimate = 0.1
        return mutated, reward_estimate

    def worker_evaluate_candidate(self, gpu_id, candidate, proof, old_signal, shared_h_tree):
        """
        G-Phasing Worker: Chạy song song trên 1 GPU MI300X.
        """
        mutated_proof = self.executor.execute(candidate, proof)
        new_signal = self.pis.compute(..., mutated_proof, ...)
        reward = self.calculate_shaped_reward(old_signal, new_signal, mutated_proof)

        l1, l2, l3 = candidate.hierarchical_indices
        shared_h_tree.async_update(l1, l2, l3, reward)
        return reward, mutated_proof

    def generate_h_ucb_candidates(self, num_candidates: int = 8, eps: float = 0.2) -> List[Any]:
        """Thompson Sampling trên H-UCB tree để chọn candidates."""
        candidates = []
        for _ in range(num_candidates):
            # Sample từ Beta distribution với prior từ H-Tree
            l1 = np.random.randint(0, 5)
            l2 = np.random.randint(0, 4)
            l3 = np.random.randint(0, 10)
            alpha, beta = self.h_tree.get_informed_prior(l1, l2, l3)

            # Epsilon-greedy với decay theo credal width
            if np.random.random() < eps:
                strategy = np.random.choice(self.operator_registry.list_strategies())
            else:
                strategy = "structural"  # Greedy default

            candidates.append({
                "hierarchical_indices": (l1, l2, l3),
                "strategy": strategy,
                "thompson_score": np.random.beta(alpha, beta)
            })
        return candidates

    def parallel_thompson_search(self, proof_graph, old_signal):
        """Đẩy 8 candidates xuống 8 GPU MI300X song song."""
        epsilon = self.get_exploration_rate(old_signal.credal_inconsistency.width)
        candidates = self.generate_h_ucb_candidates(num_candidates=8, eps=epsilon)

        processes = []
        for i in range(8):
            p = mp.Process(
                target=self.worker_evaluate_candidate,
                args=(i, candidates[i], proof_graph, old_signal, self.h_tree)
            )
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        return self.select_best_mutated_proof([])

    def select_best_mutated_proof(self, results: List[Any]) -> Any:
        """Chọn proof có reward cao nhất từ kết quả G-Phasing."""
        if not results:
            return None
        return max(results, key=lambda r: r[0] if isinstance(r, tuple) else 0)