import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s | PIS-Planner | %(message)s")
logger = logging.getLogger("atomic_planner")

# =====================================================================
# 1. DATA STRUCTURES (PIS-ALIGNED)
# =====================================================================

@dataclass
class ConfidenceCI:
    lower: float
    upper: float

    def to_dict(self):
        return {"lower": round(self.lower, 4), "upper": round(self.upper, 4)}

@dataclass
class NeuralSignals:
    token_logprobs: List[float]
    step_entropy: float
    pooled_hidden: Optional[torch.Tensor] = None # Tensor không dump JSON trực tiếp được

    def to_dict(self):
        return {
            "token_logprobs": [round(p, 4) for p in self.token_logprobs],
            "step_entropy": round(self.step_entropy, 4)
        }

@dataclass
class AtomicStep:
    step_id: str
    action: str
    entity_args: Dict[str, Any]
    expected_evidence: str
    source_refs: List[int]
    confidence_ci: ConfidenceCI
    neural_signals: NeuralSignals
    parent_step_ids: List[str]

    def to_dict(self):
        d = asdict(self)
        d['confidence_ci'] = self.confidence_ci.to_dict()
        d['neural_signals'] = self.neural_signals.to_dict()
        return d

@dataclass
class PlanCandidate:
    candidate_id: str
    atomic_steps: List[AtomicStep]
    evidence_set: Dict[int, str]
    trace: List[str]
    
    def to_dict(self):
        return {
            "candidate_id": self.candidate_id,
            "atomic_steps": [step.to_dict() for step in self.atomic_steps],
            "evidence_set": self.evidence_set,
            "trace": self.trace
        }

@dataclass
class PISReport:
    failed_step_id: str
    error_type: str  # e.g., "Hallucination", "Missing_Entity", "Logic_Contradiction"
    constraint: str  # e.g., "Must verify mechanism A before B"

# =====================================================================
# 2. ATOMIC PLANNER-RETRIEVER CORE
# =====================================================================

class AtomicPlannerRetriever:
    def __init__(self, device="cuda", log_file="pis_experiments.jsonl"):
        self.device = device
        self.log_file = log_file
        
        # Real Index Buffer thay cho mock_d_embs
        self.corpus_chunks: Dict[int, str] = {}
        self.chunk_embeddings: Optional[torch.Tensor] = None
        self.chunk_ids_map: List[int] = []
        
        # Encoder (Giả định LateInteractionEncoder đã khởi tạo từ code cũ)
        # self.encoder = LateInteractionEncoder(cfg).to(device)
        logger.info(f"Atomic Planner initialized. Logging to {self.log_file}")

    # ---------------------------------------------------------
    # INDEXING: Real chunk handling
    # ---------------------------------------------------------
    def add_corpus(self, documents: List[Tuple[int, str]]):
        """Thêm dữ liệu vào index"""
        for doc_id, text in documents:
            self.corpus_chunks[doc_id] = text
        logger.info(f"Added {len(documents)} chunks to corpus.")

    def encode_chunks(self, batch_size=32):
        """Mã hóa toàn bộ corpus thành Dense/ColBERT Tensors"""
        logger.info("Encoding corpus chunks to tensor index...")
        self.chunk_ids_map = list(self.corpus_chunks.keys())
        
        # MOCK: Tự tạo embedding thay vì gọi LLM thật để demo
        num_chunks = len(self.chunk_ids_map)
        if num_chunks > 0:
            # Chiều embedding 128 (token_dim)
            self.chunk_embeddings = torch.randn(num_chunks, 128, device=self.device)
            self.chunk_embeddings = F.normalize(self.chunk_embeddings, p=2, dim=-1)

    # ---------------------------------------------------------
    # NEURAL UNCERTAINTY & CONFIDENCE
    # ---------------------------------------------------------
    def extract_neural_signals(self, llm_outputs: Any = None) -> NeuralSignals:
        """
        Lấy tín hiệu thần kinh từ Output của LLM.
        Mock: Nếu chưa có LLM, sinh giá trị entropy và logprobs hợp lý.
        """
        # Giả lập token logprobs dao động từ -0.1 đến -2.0
        mock_logprobs = np.random.uniform(-1.5, -0.01, size=10).tolist()
        # Entropy = -sum(p * log(p)) -> mô phỏng mức độ phân vân của mô hình
        mock_entropy = float(np.abs(np.mean(mock_logprobs)) * 1.5) 
        
        return NeuralSignals(
            token_logprobs=mock_logprobs,
            step_entropy=mock_entropy,
            pooled_hidden=torch.zeros(128, device=self.device) # Mock hidden state
        )

    def _calibrate_confidence(self, base_score: float, entropy: float) -> ConfidenceCI:
        """Chuyển đổi điểm số + entropy thành Confidence Interval (CI) cho Component 3"""
        # Entropy cao -> margin rộng. Entropy thấp -> margin hẹp
        margin = min(0.4, entropy * 0.2) 
        lower = max(0.0, base_score - margin)
        upper = min(1.0, base_score + margin)
        return ConfidenceCI(lower=lower, upper=upper)

    # ---------------------------------------------------------
    # RETRIEVAL FOR SPECIFIC STEP
    # ---------------------------------------------------------
    def retrieve_for_step(self, expected_evidence: str, step_entropy: float, top_k=3) -> Tuple[List[int], ConfidenceCI]:
        """Tìm kiếm evidence cho MỘT step cụ thể, không phải query chung."""
        if self.chunk_embeddings is None or len(self.chunk_ids_map) == 0:
            return [], ConfidenceCI(0.0, 0.0)

        # MOCK: Tạo vector cho expected_evidence
        q_emb = torch.randn(128, device=self.device)
        q_emb = F.normalize(q_emb, p=2, dim=-1)

        # Tính dot product (hoặc maxsim nếu dùng ColBERT)
        scores = torch.matmul(self.chunk_embeddings, q_emb)
        top_scores, top_idx = torch.topk(scores, min(top_k, len(self.chunk_ids_map)))
        
        retrieved_ids = [self.chunk_ids_map[i.item()] for i in top_idx]
        mean_score = float(top_scores.mean().item())
        
        # Calibrate CI dựa trên score và step entropy
        ci = self._calibrate_confidence(mean_score, step_entropy)
        
        return retrieved_ids, ci

    # ---------------------------------------------------------
    # PLAN GENERATION & PIS INTEGRATION
    # ---------------------------------------------------------
    def generate_atomic_plans(self, question: str, n_samples=5, temperature=0.7) -> List[PlanCandidate]:
        """Sinh nhiều giả thuyết kế hoạch khác nhau (Experiment setup)."""
        logger.info(f"Generating {n_samples} plan candidates for: '{question}' (T={temperature})")
        candidates = []
        
        for i in range(n_samples):
            candidate_id = f"plan_{uuid.uuid4().hex[:6]}"
            steps = []
            evidence_set = {}
            
            # Giả lập LLM sinh ra 3 atomic steps
            parent_ids = []
            for step_idx in range(3):
                step_id = f"step_{step_idx+1}"
                expected_ev = f"Concept {step_idx} related to {question}"
                
                # Trích xuất tín hiệu LLM
                neural_sigs = self.extract_neural_signals()
                
                # Tự động retrieve evidence cho step này ngay khi sinh ra
                source_refs, ci = self.retrieve_for_step(expected_ev, neural_sigs.step_entropy)
                
                # Lưu evidence text
                for ref in source_refs:
                    evidence_set[ref] = self.corpus_chunks.get(ref, "")

                step = AtomicStep(
                    step_id=step_id,
                    action="EXTRACT_RELATION" if step_idx == 0 else "VERIFY_LOGIC",
                    entity_args={"subject": "Entity_A", "target": "Entity_B"},
                    expected_evidence=expected_ev,
                    source_refs=source_refs,
                    confidence_ci=ci,
                    neural_signals=neural_sigs,
                    parent_step_ids=parent_ids.copy()
                )
                steps.append(step)
                parent_ids.append(step_id)

            candidate = PlanCandidate(
                candidate_id=candidate_id,
                atomic_steps=steps,
                evidence_set=evidence_set,
                trace=[f"Generated step {s.step_id}" for s in steps]
            )
            candidates.append(candidate)
            
        return candidates

    def replan_from_pis(self, original_plan: PlanCandidate, pis_report: PISReport) -> PlanCandidate:
        """
        Nhận feedback từ PIS. Cắt bỏ từ bước lỗi và chỉ regenerate phần sau.
        """
        logger.warning(f"Replanning candidate {original_plan.candidate_id} due to PIS feedback on {pis_report.failed_step_id}. Error: {pis_report.error_type}")
        
        # Tìm index của bước lỗi
        failed_idx = next((i for i, s in enumerate(original_plan.atomic_steps) if s.step_id == pis_report.failed_step_id), None)
        
        if failed_idx is None:
            logger.error("Failed step ID not found in plan.")
            return original_plan

        # Giữ lại các bước TRƯỚC bước lỗi
        new_steps = original_plan.atomic_steps[:failed_idx]
        new_trace = original_plan.trace[:failed_idx]
        new_trace.append(f"REPLAN_TRIGGERED at {pis_report.failed_step_id} with constraint: {pis_report.constraint}")
        
        # MOCK: Sinh ra 1 bước mới sửa lỗi
        new_step_id = f"{pis_report.failed_step_id}_fixed"
        expected_ev = f"Fixed evidence incorporating constraint: {pis_report.constraint}"
        neural_sigs = self.extract_neural_signals()
        
        # Do có constraint từ PIS, thường mô hình sẽ chắc chắn hơn -> entropy giảm
        neural_sigs.step_entropy *= 0.5 
        
        source_refs, ci = self.retrieve_for_step(expected_ev, neural_sigs.step_entropy)
        
        fixed_step = AtomicStep(
            step_id=new_step_id,
            action="APPLY_CONSTRAINT",
            entity_args={"constraint": pis_report.constraint},
            expected_evidence=expected_ev,
            source_refs=source_refs,
            confidence_ci=ci,
            neural_signals=neural_sigs,
            parent_step_ids=[s.step_id for s in new_steps]
        )
        new_steps.append(fixed_step)
        new_trace.append(f"Generated fixed step {new_step_id}")

        original_plan.atomic_steps = new_steps
        original_plan.trace = new_trace
        for ref in source_refs:
             original_plan.evidence_set[ref] = self.corpus_chunks.get(ref, "")

        return original_plan

    # ---------------------------------------------------------
    # LOGGING CHO PAPER
    # ---------------------------------------------------------
    def log_experiment(self, question: str, candidates: List[PlanCandidate], 
                       selected_candidate_id: str, pis_report: Optional[PISReport], 
                       replan_result: Optional[PlanCandidate]):
        """Dump JSONL để làm dataset huấn luyện hoặc phân tích paper."""
        record = {
            "question": question,
            "candidates": [c.to_dict() for c in candidates],
            "selected_candidate_id": selected_candidate_id,
            "pis_report": asdict(pis_report) if pis_report else None,
            "replan_result": replan_result.to_dict() if replan_result else None
        }
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"Experiment logged to {self.log_file}")


# =====================================================================
# QUICK TEST / USAGE EXAMPLE
# =====================================================================
if __name__ == "__main__":
    planner = AtomicPlannerRetriever()
    
    # 1. Add corpus & index
    planner.add_corpus([
        (101, "MI300X accelerates tensor operations via XDNA architecture."),
        (102, "Symbolic logic requires exact entity matching over graph nodes."),
        (103, "Neuro-symbolic integration bridges gradient descent with rule-based systems.")
    ])
    planner.encode_chunks()
    
    # 2. Generate initial candidates
    question = "How does MI300X execute neuro-symbolic logic?"
    candidates = planner.generate_atomic_plans(question, n_samples=3)
    
    # 3. Giả lập Component 3 (PIS) phát hiện lỗi ở Candidate 0, Step 2
    best_candidate = candidates[0]
    report = PISReport(
        failed_step_id="step_2", 
        error_type="Logic_Contradiction", 
        constraint="Must fetch architecture details of XDNA before inferring logic mapping."
    )
    
    # 4. Replan từ bước lỗi
    fixed_plan = planner.replan_from_pis(best_candidate, report)
    
    # 5. Log data chuẩn bị cho paper
    planner.log_experiment(
        question=question,
        candidates=candidates,
        selected_candidate_id=best_candidate.candidate_id,
        pis_report=report,
        replan_result=fixed_plan
    )
    print("Test finished. Check pis_experiments.jsonl")
@dataclass
class ActiveReasonerConfig:
    hidden_size:      int   = 4096
    num_layers:       int   = 32
    num_heads:        int   = 32
    intermediate_size: int  = 11008
    max_seq_len:      int   = 4096
    vocab_size:       int   = 32000
    dropout:          float = 0.1
    device:           str   = "cuda" if torch.cuda.is_available() else "cpu"