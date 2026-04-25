import torch
import torch.nn as nn
import torch.nn.functional as F
import uuid
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict

logger = logging.getLogger("mc_nsr.diff_symbolic_pis")

# ===========================================================================
# 1. DATA STRUCTURES (PIS-ALIGNED)
# ===========================================================================
def _validate_llm_program(raw_program):
    """
    Normalize raw logic program into Component-2 executable format.
    Expected output: List[Tuple[str, Dict, List[int]]]
    """
    validated = []

    for item in raw_program:
        if isinstance(item, tuple):
            if len(item) == 2:
                rule_name, args = item
                evidence_ids = []
            elif len(item) == 3:
                rule_name, args, evidence_ids = item
            else:
                raise ValueError(f"Invalid program tuple: {item}")

            validated.append({
                "action": rule_name,
                "entity_args": args or {},
                "source_refs": evidence_ids or [],
            })

        elif isinstance(item, dict):
            validated.append({
                "action": item.get("action", item.get("rule_name", "UNKNOWN")),
                "entity_args": item.get("entity_args", item.get("args", {})),
                "source_refs": item.get("source_refs", item.get("linked_evidence_ids", [])),
            })

        else:
            raise TypeError(f"Unsupported program step type: {type(item)}")

    return validated

@dataclass
class ConfidenceCI:
    lower: float
    upper: float

    def to_dict(self):
        return {"lower": round(self.lower, 4), "upper": round(self.upper, 4)}

@dataclass
class SymbolicProofStep:
    step_id: str
    step_index: int
    rule_name: str
    premises: List[str]              # Danh sách step_id của các premises
    conclusion: str                  # Tóm tắt state/tensor hash
    linked_evidence_ids: List[int]   # ID map trực tiếp từ Component 1
    confidence: ConfidenceCI
    status: str                      # "proven", "uncertain", "failed"
    failure_type: Optional[str] = None # INCONSISTENCY, MISSING_PREMISE, INVALID_INFERENCE

    def to_dict(self):
        d = asdict(self)
        d['confidence'] = self.confidence.to_dict()
        return d

@dataclass
class ProofTrace:
    steps: List[SymbolicProofStep]
    overall_confidence: ConfidenceCI
    
    def to_trace(self) -> Dict[str, Any]:
        """Expose API to_trace() trả JSON serializable cho C3 và Logging"""
        edges = []
        for step in self.steps:
            for premise_id in step.premises:
                edges.append({"source": premise_id, "target": step.step_id, "type": "entails"})
        
        return {
            "steps": [s.to_dict() for s in self.steps],
            "edges": edges,
            "overall_confidence": self.overall_confidence.to_dict()
        }

# ===========================================================================
# 2. DIFFERENTIABLE CONTEXT & RULES (Cập nhật để bắt lỗi)
# ===========================================================================

class DifferentiableContext:
    def __init__(self, num_entities: int, hidden_dim: int, device: str = "cuda"):
        self.device = device
        self.num_entities = num_entities
        self.hidden_dim = hidden_dim
        self.entities = torch.zeros(num_entities, hidden_dim, device=device, dtype=torch.bfloat16)
        self.fluents = []

    def clone(self) -> "DifferentiableContext":
        new_ctx = DifferentiableContext.__new__(DifferentiableContext)
        new_ctx.device = self.device
        new_ctx.num_entities = self.num_entities
        new_ctx.hidden_dim = self.hidden_dim
        new_ctx.entities = self.entities.clone()
        new_ctx.fluents = [f.clone() for f in self.fluents]
        return new_ctx

class DifferentiableRule(nn.Module):
    def forward(self, ctx: DifferentiableContext, *args, **kwargs) -> DifferentiableContext:
        raise NotImplementedError

class EventCalculusRule(DifferentiableRule):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W_persist = nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=torch.bfloat16)
        self.W_event = nn.Linear(hidden_dim, hidden_dim, dtype=torch.bfloat16)

    def forward(self, ctx: DifferentiableContext, current_time_step: int = 0, event_tensor: Optional[torch.Tensor] = None):
        if event_tensor is None:
            raise ValueError("MISSING_PREMISE: Event tensor cannot be None for EventCalculus")
        
        if not ctx.fluents:
            ctx.fluents.append(ctx.entities.clone()) 
            
        S_t = ctx.fluents[-1]
        try:
            S_next = F.silu(self.W_persist(S_t) + self.W_event(event_tensor.to(S_t.dtype)))
        except Exception as e:
            raise RuntimeError(f"INVALID_INFERENCE: Tensor dimension mismatch - {str(e)}")

        ctx.fluents.append(S_next)
        ctx.entities = S_next # Update current belief state
        return ctx

# ===========================================================================
# 3. CORE ENGINE: UNCERTAINTY-AWARE BEAM SEARCH
# ===========================================================================

@dataclass
class BeamNode:
    ctx: DifferentiableContext
    program_remaining: List[Tuple[str, Dict, List[int]]] # rule_name, args, linked_evidence_ids
    proof_steps: List[SymbolicProofStep]
    credal_width: float  
    failed_steps_count: int = 0

class NeuroSymbolicCompiler(nn.Module):
    def __init__(self, hidden_dim: int, pis_evaluator: Any):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pis = pis_evaluator # PIS Module (Component 3)
        self.core_rules = nn.ModuleDict({"EVENT_CALC": EventCalculusRule(hidden_dim)})
        
    def _parse_llm_program(self, raw_program: List[Dict]) -> List[Tuple[str, Dict, List[int]]]:
        """Parse đầu vào từ Component 1 (đảm bảo giữ lại evidence_ids)"""
        parsed = []
        for step in raw_program:
            rule_name = step.get("action", "UNKNOWN")
            args = step.get("entity_args", {})
            evidence_ids = step.get("source_refs", [])
            parsed.append((rule_name, args, evidence_ids))
        return parsed

    def _derive_ci(self, credal_width: float, is_failed: bool = False) -> ConfidenceCI:
        """Biến đổi Tensor Credal Width thành Confidence Interval"""
        if is_failed:
            return ConfidenceCI(0.0, 1.0) # Uninformative CI
        
        base_conf = max(0.0, 1.0 - credal_width)
        margin = credal_width / 2.0
        return ConfidenceCI(lower=max(0.0, base_conf - margin), upper=min(1.0, base_conf + margin))

    def execute_with_beam_search(self, initial_ctx_tensor: torch.Tensor, raw_program: List[Dict], beam_width=3) -> ProofTrace:
        ctx = DifferentiableContext(num_entities=1, hidden_dim=self.hidden_dim, device=str(initial_ctx_tensor.device))
        ctx.entities[0, :min(self.hidden_dim, initial_ctx_tensor.shape[0])] = initial_ctx_tensor[:self.hidden_dim]

        program = self._parse_llm_program(raw_program)
        
        beam = [BeamNode(
            ctx=ctx, 
            program_remaining=program, 
            proof_steps=[], 
            credal_width=0.0,
            failed_steps_count=0
        )]

        step_counter = 0

        while beam and any(node.program_remaining for node in beam):
            next_beam = []
            step_counter += 1

            for node in beam:
                if not node.program_remaining:
                    next_beam.append(node)
                    continue

                rule_name, args, ev_ids = node.program_remaining[0]
                step_id = f"exec_step_{step_counter}_{uuid.uuid4().hex[:4]}"
                parent_ids = [s.step_id for s in node.proof_steps[-1:]] # Lấy step trước làm premise
                
                rule_module = self.core_rules[rule_name] if rule_name in self.core_rules else None
                
                new_ctx = node.ctx.clone()
                new_credal_width = node.credal_width
                status = "proven"
                failure_type = None
                
                # Thực thi Rule với Partial Execution & Exception Handling
                try:
                    if rule_module is None:
                        raise ValueError(f"MISSING_PREMISE: Rule '{rule_name}' not in registry.")
                    
                    new_ctx = rule_module(new_ctx, **args)
                    
                    # (8) Bắt buộc PIS: tính Tensor Credal Width
                    new_credal_width = self.pis.compute_tensor_credal_width(new_ctx.entities)
                    
                    if new_credal_width > 0.4:  # Threshold tạm định
                        status = "uncertain"
                        failure_type = "INCONSISTENCY"
                        
                except Exception as e:
                    status = "failed"
                    # Classify lỗi theo prefix
                    err_msg = str(e)
                    if "MISSING_PREMISE" in err_msg: failure_type = "MISSING_PREMISE"
                    elif "INVALID_INFERENCE" in err_msg: failure_type = "INVALID_INFERENCE"
                    else: failure_type = "INVALID_INFERENCE"
                    
                    logger.warning(f"Step {step_id} failed: {failure_type}. Raw error: {str(e)}. Continuing execution.")

                # Đóng gói Step
                step_ci = self._derive_ci(new_credal_width, is_failed=(status == "failed"))
                proof_step = SymbolicProofStep(
                    step_id=step_id,
                    step_index=step_counter,
                    rule_name=rule_name,
                    premises=parent_ids,
                    conclusion=f"TensorHash_{hash(new_ctx.entities.sum().item())}",
                    linked_evidence_ids=ev_ids,
                    confidence=step_ci,
                    status=status,
                    failure_type=failure_type
                )
                
                new_proof_steps = list(node.proof_steps) + [proof_step]
                failed_count = node.failed_steps_count + (1 if status == "failed" else 0)

                next_beam.append(BeamNode(
                    ctx=new_ctx,
                    program_remaining=node.program_remaining[1:],
                    proof_steps=new_proof_steps,
                    credal_width=new_credal_width,
                    failed_steps_count=failed_count
                ))

            # (7) Sort Beam theo: Ưu tiên ít step hỏng nhất -> Credal width thấp nhất
            next_beam.sort(key=lambda x: (x.failed_steps_count, x.credal_width))
            beam = next_beam[:beam_width]

        # Trả về Trace tốt nhất
        best_node = beam[0]
        overall_ci = self._derive_ci(best_node.credal_width, is_failed=(best_node.failed_steps_count > 0))
        
        return ProofTrace(
            steps=best_node.proof_steps,
            overall_confidence=overall_ci
        )

# ===========================================================================
# MOCK PIS (Dành cho test)
# ===========================================================================
class MockPIS:
    def compute_tensor_credal_width(self, tensor: torch.Tensor) -> float:
        # Giả lập Credal width dựa trên độ phân tán của tensor
        variance = torch.var(tensor).item()
        return min(0.99, variance * 2.0)

if __name__ == "__main__":
    # Khởi tạo Mock PIS và Compiler
    pis = MockPIS()
    compiler = NeuroSymbolicCompiler(hidden_dim=128, pis_evaluator=pis)
    
    # Giả lập Input từ Component 1 (Atomic Planner)
    initial_tensor = torch.randn(128, dtype=torch.bfloat16, device="cuda" if torch.cuda.is_available() else "cpu")
    
    # 1 Step chuẩn, 1 Step cố tình lỗi (để trigger partial execution)
    raw_plan_from_c1 = [
        {"action": "EVENT_CALC", "entity_args": {"event_tensor": torch.randn(128)}, "source_refs": [101, 102]},
        {"action": "UNKNOWN_RULE", "entity_args": {}, "source_refs": [103]} # Gây lỗi Missing Premise
    ]
    
    # Thực thi
    trace = compiler.execute_with_beam_search(initial_tensor, raw_plan_from_c1)
    
    # Export Trace JSON
    print(json.dumps(trace.to_trace(), indent=2))