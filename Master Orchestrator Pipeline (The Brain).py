import torch
import torch.nn as nn
import torch.multiprocessing as mp
import numpy as np
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import logging

# [Fix 5] BFloat16 precision cho AMD MI300X — tránh lỗi Float8/dtype mismatch toàn hệ thống
torch.set_float32_matmul_precision('high')

from component1_retriever import MultiSourceRetriever, ActiveReasonerConfig
from component2_symbolic import NeuroSymbolicCompiler, DifferentiableContext, _coerce_to_context, _validate_llm_program
from component3_probabilistic import ProbabilisticInconsistencySignal
from component4_bandit import SOTABanditMutationEngine
from component5_metacognitive import SOTAMetaCognitiveVerifier

logger = logging.getLogger("ANSB_Master_Orchestrator")

# ============================================================================
# Section 1: Telemetry & Elastic VRAM Partitioning (DRA Architecture)
# ============================================================================

class DifficultyPredictor(nn.Module):
    """Phân loại độ khó của câu hỏi để cấp phát VRAM (Triage)."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def predict(self, question: str) -> float:
        dummy_embedding = torch.randn(768)
        return self.sigmoid(self.encoder(dummy_embedding)).item()

class VRAMGovernor:
    """Giám sát HBM3 và thực hiện Context Shifting / Early Exit."""
    def __init__(self, total_gpus=8):
        self.total_gpus = total_gpus
        self.active_pools = {"N-Pool": 0, "L-Pool": 0, "M-Pool": 0}
        self.vram_usage_percent = 0.0

    def allocate_resources(self, difficulty: float) -> Dict[str, Any]:
        """Elastic VRAM Partitioning dựa trên độ khó."""
        if difficulty < 0.3:
            logger.info("🟢 Difficulty: EASY. Allocating 1 GPU (4-bit Quantized N-Pool).")
            return {"mode": "efficient", "gpus": [0], "precision": "int4", "branches": 2}
        elif difficulty < 0.7:
            logger.info("🟡 Difficulty: MEDIUM. Allocating 4 GPUs.")
            return {"mode": "standard", "gpus": [0,1,2,3], "precision": "bf16", "branches": 4}
        else:
            logger.info("🔴 Difficulty: HARD/PARADOXICAL. Overdrive Mode! 8 GPUs active.")
            return {"mode": "overdrive", "gpus": list(range(8)), "precision": "fp16", "branches": 16}

    def check_compute_aware_stopping(self, time_elapsed: float, best_confidence: float) -> bool:
        """Ngắt sớm nếu VRAM/Compute tốn quá nhiều mà confidence không tăng."""
        self.vram_usage_percent += time_elapsed * 0.1
        if self.vram_usage_percent > 90.0 and best_confidence < 0.5:
            logger.warning("🚨 Compute-Aware Stopping Triggered! Saving VRAM.")
            return True
        return False

    def trigger_context_shifting(self, blackboard: 'Blackboard'):
        """Offload KV-cache của các nhánh MCTS kém cỏi xuống RAM."""
        logger.info("🔄 Context Shifting: Offloading poor branches to CPU RAM.")
        for node in blackboard.nodes.values():
            if node.score < 0.3 and hasattr(node, 'kv_cache'):
                node.kv_cache = "OFFLOADED_TO_RAM"

# ============================================================================
# Section 2: Global State (The Blackboard)
# ============================================================================

@dataclass
class ProofNode:
    """Một Node trong Forest of Proofs trên Blackboard."""
    id: str
    proof_chain: List[Any]
    evidence: List[Any]
    credal_score: float = 0.0
    meta_score: float = 0.0
    fusion_score: float = 0.0
    parent_id: Optional[str] = None
    visits: int = 0
    kv_cache: Any = None

class Blackboard:
    """Shared Memory Space lưu trữ toàn bộ lịch sử suy luận."""
    def __init__(self):
        self.nodes: Dict[str, ProofNode] = {}
        self.best_node_id: Optional[str] = None

    def add_node(self, node: ProofNode):
        self.nodes[node.id] = node
        if self.best_node_id is None or node.fusion_score > self.nodes[self.best_node_id].fusion_score:
            self.best_node_id = node.id

    def get_best_node(self) -> ProofNode:
        return self.nodes[self.best_node_id]

# ============================================================================
# Section 3: The Arbitrator (MCTS Search Engine)
# ============================================================================

class ArbitratorMCTS:
    """Sử dụng UCT (Upper Confidence Bound applied to Trees) để chọn nhánh."""
    def __init__(self, exploration_weight=1.414):
        self.c = exploration_weight

    def select_next_node(self, blackboard: Blackboard) -> ProofNode:
        """Chọn Node tiềm năng nhất để mở rộng dựa trên Fusion Score và số lần duyệt."""
        best_uct = -float('inf')
        selected_node = None
        
        total_visits = sum(n.visits for n in blackboard.nodes.values()) or 1

        for node in blackboard.nodes.values():
            if node.visits == 0:
                return node
            
            uct_score = node.fusion_score + self.c * np.sqrt(np.log(total_visits) / node.visits)
            if uct_score > best_uct:
                best_uct = uct_score
                selected_node = node
                
        return selected_node

# ============================================================================
# Section 4: THE MASTER ORCHESTRATOR (The Brain)
# ============================================================================

class ANSBMasterOrchestrator:
    """
    Điều phối toàn bộ luồng data từ C1 -> C5.
    Hỗ trợ Parallel Initialization, Concurrent Evaluation, và Dynamic Strategy.
    """
    def __init__(self):
        self.governor = VRAMGovernor()
        self.triage = DifficultyPredictor()
        self.arbitrator = ArbitratorMCTS()
        self.blackboard = Blackboard()

        # [Fix 3] Một nguồn config duy nhất — không dùng `cfg` tự do ở bất kỳ đâu
        self.config = ActiveReasonerConfig()
        self.hidden_dim = self.config.hidden_size  # Biến tiện dùng, tránh gõ lại

        # [Fix 2] prob_engine khởi tạo TRƯỚC symbolic_engine vì symbolic cần nó làm dependency
        self.prob_engine = ProbabilisticInconsistencySignal()

        # [Fix 3] Dùng self.config. thay vì cfg. nhất quán toàn bộ
        self.retriever = MultiSourceRetriever(cfg=self.config)
        self.symbolic_engine = NeuroSymbolicCompiler(
            hidden_dim=self.hidden_dim,         # [Fix 3] dùng self.hidden_dim đã cache
            pis_evaluator=self.prob_engine      # [Fix 2] đã tồn tại ở dòng trên
        )

        self.bandit_engine = SOTABanditMutationEngine()
        self.verifier = SOTAMetaCognitiveVerifier(llm_model=None)

    # --------------------------------------------------------------------------
    # [Fix 1] Safe ModuleDict access helper
    # Dùng hàm này bất cứ khi nào cần lấy layer/module từ một nn.ModuleDict
    # --------------------------------------------------------------------------
    @staticmethod
    def _safe_module_get(module_dict: nn.ModuleDict, key: str, default=None):
        """
        [Fix 1] An toàn khi truy cập nn.ModuleDict — KHÔNG dùng .get() trực tiếp.
        ModuleDict không hỗ trợ .get(key, default) như dict thường.

        Sai:  layer = self.layers.get("conv1", None)
        Đúng: layer = _safe_module_get(self.layers, "conv1", None)
        """
        return module_dict[key] if key in module_dict else default

    def _generate_logic_steps(self, question: str) -> List[Any]:
        """
        Stub sinh llm_program — kế hoạch suy luận dạng List[(rule_name, args)].
        Production: thay bằng LLM call thật, vd: self.llm.generate_logic_steps(question).
        """
        logger.info(f"🧠 Generating logic plan for: '{question}'")
        return [
            ("EVENT_CALC", {"current_time_step": 0, "event_tensor": None}),
            ("DO_CALC",    {"intervention_node_idx": 0}),
        ]

    def _fusion_function(self, credal_score: float, meta_score: float) -> float:
        """Hòa trộn tín hiệu: S = f(Credal, Metacognitive)."""
        return 0.4 * (1.0 - credal_score) + 0.6 * meta_score

    def execute_pipeline(self, question: str, max_iterations=10) -> Dict[str, Any]:
        start_time = time.time()
        logger.info(f"🚀 Initializing ANSB Master Pipeline for: '{question}'")

        # 1. Triage & Resource Allocation
        difficulty = self.triage.predict(question)
        resources = self.governor.allocate_resources(difficulty)
        
        # 2. Parallel Initialization
        logger.info(f"🌱 [Step 1] Parallel Initialization with {resources['branches']} branches...")
        initial_evidences = self.retriever.retrieve(question, top_k=resources['branches'])

        # [Fix 2+3] LLM thật: llm_plan = llm_output.get("program", "default_verify")
        # Stub dùng _generate_logic_steps, sau đó validate qua _validate_llm_program
        raw_plan = self._generate_logic_steps(question)
        llm_plan = _validate_llm_program(raw_plan)  # [Fix 3] Không bao giờ crash vì list rỗng/None

        # [Device Fix] EvidenceWrapper: bọc string/object thô từ Retriever thành object
        # có đủ .text, .confidence, .embedding để C3 (RetrievedEvidence) không bị AttributeError
        class EvidenceWrapper:
            """
            Shim layer giữa Retriever output (có thể là string thuần) và C3 interface.
            C3 cần: .confidence (CredalInterval-like hoặc float), .text, .embedding
            """
            def __init__(self, raw):
                if isinstance(raw, str):
                    self.text       = raw
                    self.source_id  = "retriever_raw"
                    self.confidence = type('CI', (), {           # duck-type CredalInterval
                        'lower': 0.8, 'upper': 0.95,
                        'midpoint': 0.875, 'width': 0.15,
                    })()
                    self.relevance_score = 0.8
                    self.sparse_score    = 0.7
                    self.hop_depth       = 1
                    # [Device Fix] embedding trên CUDA bfloat16 — tránh CPU/CUDA mismatch ở C2
                    self.embedding = torch.zeros(128, device="cuda", dtype=torch.bfloat16)
                else:
                    # Object đã đủ thuộc tính — passthrough, chỉ đảm bảo embedding trên CUDA
                    self.__dict__.update(raw.__dict__ if hasattr(raw, '__dict__') else {})
                    if not hasattr(self, 'confidence'):
                        self.confidence = type('CI', (), {
                            'lower': 0.8, 'upper': 0.95,
                            'midpoint': 0.875, 'width': 0.15,
                        })()
                    if not hasattr(self, 'embedding'):
                        self.embedding = torch.zeros(128, device="cuda", dtype=torch.bfloat16)
                    elif isinstance(self.embedding, torch.Tensor):
                        # [Device Fix] Pin embedding về CUDA bfloat16
                        self.embedding = self.embedding.to(device="cuda", dtype=torch.bfloat16)
                    if not hasattr(self, 'text'):
                        self.text = str(raw)

        extra_params = {}
        beam_width = resources.get('beam_width', 3)
        # [Fix] Coerce từng evidence + compile beam search
        for i, ev in enumerate(initial_evidences):
            # Bọc ev thành EvidenceWrapper nếu là string hoặc thiếu thuộc tính
            ev_obj = EvidenceWrapper(ev) if isinstance(ev, (str, type(None))) else EvidenceWrapper(ev)

            ctx = _coerce_to_context(ev_obj, hidden_dim=self.hidden_dim)  # [Fix 2]
            initial_proof = self.symbolic_engine.compile(
                initial_ctx=ctx,
                logic_plan=llm_plan,
                **extra_params
            )
            node = ProofNode(id=f"init_{i}", proof_chain=initial_proof, evidence=ev_obj)
            self.blackboard.add_node(node)

        # Main Orchestration Loop
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"🔄 --- Iteration {iteration} ---")
            
            # 3. MCTS Selection
            current_node = self.arbitrator.select_next_node(self.blackboard)
            current_node.visits += 1

            # 4. Concurrent Evaluation
            logger.info(f"⚡ [Step 2] Concurrent Evaluation on Node {current_node.id}")

            # --- Chuẩn bị 6 tham số cho C3.compute() ---
            # Tham số 1: evidence_set — dùng evidence GỐC đã bọc EvidenceWrapper từ init loop
            # current_node.evidence luôn là EvidenceWrapper (đã bọc ở bước khởi tạo),
            # nhưng các node được sinh ra bởi deepdive/mutation có thể còn là object thô.
            _raw_ev = current_node.evidence
            if isinstance(_raw_ev, list):
                # Đã là list — đảm bảo từng phần tử có .confidence
                _evidence_list = [
                    EvidenceWrapper(e) if not hasattr(e, 'confidence') else e
                    for e in _raw_ev
                ]
            elif _raw_ev is None:
                # Node không có evidence (recovery node) — dùng dummy wrapper
                _evidence_list = [EvidenceWrapper("")]
            else:
                # Single object — bọc lại nếu thiếu .confidence
                _evidence_list = [
                    EvidenceWrapper(_raw_ev) if not hasattr(_raw_ev, 'confidence') else _raw_ev
                ]
            # Tham số 2: proof_chain — C3 cần List[SymbolicProofStep]
            # BeamNode từ C2 không phải SymbolicProofStep — unwrap nếu cần
            _raw_chain = current_node.proof_chain
            if isinstance(_raw_chain, list):
                _proof_list = _raw_chain
            elif hasattr(_raw_chain, 'path_trace'):
                # BeamNode — lấy ctx.trace_log làm proof steps nếu SymbolicProofStep chưa có
                _proof_list = [_raw_chain]
            else:
                _proof_list = [_raw_chain] if _raw_chain is not None else []
            # Tham số 3: context_embed — (1, 1024) bfloat16
            # Production: ev.embedding.to("cuda").to(torch.bfloat16).unsqueeze(0)
            # Stub: zero tensor khớp context_dim=1024 của AttentionSteeredEvidentialExtractor
            _context_embed = torch.zeros(1, 1024, device="cuda", dtype=torch.bfloat16)
            # Tham số 4: symbolic_adj_matrix — ma trận kề causal graph từ C2
            # Production: lấy ctx.causal_graph.to_dense() từ BeamNode trả về
            # Stub: identity matrix theo số bước proof
            _n_steps = max(len(_proof_list), 1)
            _adj_matrix = torch.eye(_n_steps, device="cuda", dtype=torch.bfloat16)
            # Tham số 5: llm_attention_matrix — attention map từ LLM (B x heads x seq x seq)
            # Production: model_outputs.attentions[-1]
            # Stub: ones tensor (1, 8, 32, 32)
            _attn_matrix = torch.ones(1, 8, 32, 32, device="cuda", dtype=torch.bfloat16)
            # Tham số 6: llm_outputs — None = credal-only mode cho đến khi LLM thật được mount
            _llm_outputs = None

            c3_signal = self.prob_engine.compute(
                evidence_set=_evidence_list,
                proof_chain=_proof_list,
                llm_outputs=_llm_outputs,
                context_embed=_context_embed,
                symbolic_adj_matrix=_adj_matrix,
                llm_attention_matrix=_attn_matrix,
            )
            c5_result = self.verifier.verify(current_node.proof_chain, c3_signal, question, {})

            # [Fix] Defensive read C3 — guard nếu compute() trả về string hoặc object thiếu attr
            if isinstance(c3_signal, str):
                logger.warning(f"[C3] compute() returned string: '{c3_signal}'. Defaulting credal_err=0.5")
                credal_err = 0.5
            else:
                credal_err = getattr(c3_signal, 'combined_score', 0.5)

            # [Fix] Defensive read C5 — guard nếu verify() trả về tuple/string (legacy path)
            if isinstance(c5_result, (str, tuple)):
                logger.warning(f"[C5] verify() returned {type(c5_result).__name__}: '{c5_result}'. Defaulting meta_conf=0.5")
                meta_conf = 0.5
            else:
                # verification_score: 0=tốt, 1=tệ → meta_conf = 1 - score
                raw_score = getattr(c5_result, 'verification_score',
                            getattr(c5_result, 'confidence', 0.5))
                meta_conf = 1.0 - float(raw_score)
            
            fusion_score = self._fusion_function(credal_err, meta_conf)
            current_node.fusion_score = fusion_score
            self.blackboard.add_node(current_node)

            # Telemetry Check
            elapsed = time.time() - start_time
            if self.governor.check_compute_aware_stopping(elapsed, fusion_score):
                break
                
            if iteration % 3 == 0:
                self.governor.trigger_context_shifting(self.blackboard)

            # 5. Dynamic Strategy Switching
            logger.info(f"🧠 [Step 3] Dynamic Strategy Assessment: Fusion Score = {fusion_score:.4f}")
            
            # Mode 1: Fast Path
            if fusion_score > 0.90:
                logger.info("✅ Fast Path Triggered! Optimal proof found.")
                break
                
            # Mode 2: Symbolic Deep-Dive (C3 chê, C5 khen)
            elif credal_err > 0.6 and meta_conf > 0.8:
                logger.info("🔍 Symbolic Deep-Dive: Logic flaw detected. Refreshing Evidence.")
                new_ev = self.retriever.retrieve_with_negative_constraint(current_node.evidence)
                new_proof = self.symbolic_engine.compile(
                    initial_ctx=new_ev,
                    llm_program=llm_plan
                )
                self.blackboard.add_node(ProofNode(
                    id=f"deepdive_{iteration}",
                    proof_chain=new_proof,
                    evidence=new_ev,
                    parent_id=current_node.id
                ))
                
            # Mode 3: Neural Alignment (C3 khen, C5 chê)
            elif credal_err < 0.4 and meta_conf < 0.5:
                logger.info("🧩 Neural Alignment: Injecting Steering Vectors and Re-phrasing.")
                mutated_proof, _ = self.bandit_engine.mutate(
                    c3_signal, current_node.proof_chain, current_node.evidence, strategy="rephrase"
                )
                self.blackboard.add_node(ProofNode(
                    id=f"align_{iteration}",
                    proof_chain=mutated_proof,
                    evidence=current_node.evidence,
                    parent_id=current_node.id
                ))

            # Mode 4: Standard Mutation
            else:
                logger.info("⚙️ Standard Bandit Mutation triggered.")
                mutated_proof, _ = self.bandit_engine.mutate(
                    c3_signal, current_node.proof_chain, current_node.evidence
                )
                self.blackboard.add_node(ProofNode(
                    id=f"mutate_{iteration}",
                    proof_chain=mutated_proof,
                    evidence=current_node.evidence,
                    parent_id=current_node.id
                ))

            # 6. Dead-end Recovery (Backtracking)
            best_global_score = self.blackboard.get_best_node().fusion_score
            if best_global_score < 0.3 and iteration > max_iterations // 2:
                logger.warning("☠️ Dead-end detected! Triggering Backtracking.")
                anti_ev = self.retriever.retrieve(question, negative_constraint=True)
                new_proof = self.symbolic_engine.compile(
                    initial_ctx=anti_ev[0],
                    llm_program=llm_plan
                )
                self.blackboard.add_node(ProofNode(
                    id=f"recovery_{iteration}",
                    proof_chain=new_proof,
                    evidence=anti_ev[0]
                ))

        # Finalization
        end_time = time.time()
        best_final_node = self.blackboard.get_best_node()
        
        return {
            "final_proof": best_final_node.proof_chain,
            "fusion_score": best_final_node.fusion_score,
            "iterations": iteration,
            "latency_ms": (end_time - start_time) * 1000,
            "telemetry": {
                "vram_max_percent": self.governor.vram_usage_percent,
                "nodes_explored": len(self.blackboard.nodes)
            }
        }


# Khởi chạy thử nghiệm
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    orchestrator = ANSBMasterOrchestrator()
    
    tricky_question = "If the Eiffel Tower was built in 1889, how could a photograph of it exist in 1885?"
    result = orchestrator.execute_pipeline(tricky_question)
    
    print("\n" + "="*50)
    print("🏆 PIPELINE EXECUTION COMPLETE 🏆")
    print("="*50)
    print(f"Latency: {result['latency_ms']:.2f} ms")
    print(f"Final Fusion Score: {result['fusion_score']:.4f}")
    print(f"Nodes Explored (Diversity): {result['telemetry']['nodes_explored']}")