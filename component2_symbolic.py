"""
Component 2: Differentiable Symbolic Execution Engine
MC-NSR Framework — Meta-Cognitive Neuro-Symbolic Reasoning

Paradigm Shift:
  1. Differentiable Logic: Rules are Sparse Matrix Multiplications.
  2. High-Order Calculi: Event Calculus (Fluents) & Do-Calculus (Graph Surgery).
  3. Program Synthesis: LLM generates Proof Graphs, Engine compiles & executes.
  4. ILP (Dynamic Rule Discovery): Learns new logic weight matrices dynamically.
  5. Uncertainty-Aware Beam Search: Path finding guided by Credal Width.

Hardware Target: AMD MI300X, Sparse Tensor Core, FP8/BF16.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging

# Thiết lập Logger
logger = logging.getLogger("mc_nsr.diff_symbolic")

# ===========================================================================
# 1. Type Adapter (Chuẩn hóa đầu vào từ Master Brain)
# ===========================================================================

def _coerce_to_context(data, hidden_dim=768): 
    """
    Chuẩn hóa dữ liệu đầu vào thành bfloat16 Tensor trên đúng Device.
    Đổi 'initial_ctx' thành 'data' cho linh hoạt.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Nếu data là object (ev_obj) có thuộc tính embedding
    if hasattr(data, 'embedding') and data.embedding is not None:
        return data.embedding.to(device).to(torch.bfloat16)
        
    # 2. Nếu data đã là Tensor sẵn rồi
    if isinstance(data, torch.Tensor):
        return data.to(device).to(torch.bfloat16)
        
    # 3. Fallback: Tạo zero context nếu data không hợp lệ hoặc là string
    return torch.zeros((1, hidden_dim)).to(device).to(torch.bfloat16)


# ===========================================================================
# 2. Differentiable Execution Context (Không gian Vector)
# ===========================================================================

class DifferentiableContext:
    """
    Thay vì lưu chuỗi (strings), Context giờ lưu trạng thái dưới dạng Tensor.
    Mỗi Entity/Fluent là một vector. Quan hệ là ma trận (Adjacency Matrix).
    """
    def __init__(self, num_entities: int, hidden_dim: int, device: str = "cuda"):
        self.device = device
        self.num_entities = num_entities
        self.hidden_dim = hidden_dim

        # Trạng thái niềm tin của các thực thể (Belief State) [N, D]
        self.entities = torch.zeros(num_entities, hidden_dim, device=device, dtype=torch.bfloat16)

        # Causal Graph: Ma trận kề đại diện cho quan hệ nhân quả [N, N]
        self.causal_graph = torch.sparse_coo_tensor(
            indices=torch.empty((2, 0), dtype=torch.long),
            values=torch.empty(0, dtype=torch.bfloat16),
            size=(num_entities, num_entities),
            device=device
        )

        # Event Calculus: Fluents (Trạng thái theo thời gian) [Time_steps, N]
        self.fluents = []

        # Memory & Trace
        self.trace_log = []
        self.semantic_errors = []

    def clone(self) -> "DifferentiableContext":
        """Deep copy context — dùng khi Beam Search rẽ nhánh độc lập."""
        new_ctx = DifferentiableContext.__new__(DifferentiableContext)
        new_ctx.device = self.device
        new_ctx.num_entities = self.num_entities
        new_ctx.hidden_dim = self.hidden_dim
        new_ctx.entities = self.entities.clone()
        new_ctx.causal_graph = self.causal_graph.clone()
        new_ctx.fluents = [f.clone() for f in self.fluents]
        new_ctx.trace_log = list(self.trace_log)
        new_ctx.semantic_errors = list(self.semantic_errors)
        return new_ctx


# ===========================================================================
# 3. High-Order Temporal & Causal Rules (Differentiable Operators)
# ===========================================================================

class DifferentiableRule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ctx: DifferentiableContext, *args, **kwargs) -> DifferentiableContext:
        raise NotImplementedError


class EventCalculusRule(DifferentiableRule):
    """Xử lý trạng thái tồn tại qua thời gian bằng ma trận khả vi."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W_persist = nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=torch.bfloat16)
        self.W_event = nn.Linear(hidden_dim, hidden_dim, dtype=torch.bfloat16)

    def forward(self, ctx: DifferentiableContext, current_time_step: int, event_tensor: Optional[torch.Tensor] = None):
        if not ctx.fluents:
            ctx.fluents.append(ctx.entities.clone()) 

        _device = self.W_persist.weight.device
        _dtype  = self.W_persist.weight.dtype 

        S_t = ctx.fluents[-1].to(device=_device, dtype=_dtype)

        if event_tensor is None or not isinstance(event_tensor, torch.Tensor):
            event_tensor = torch.zeros_like(S_t)
        else:
            event_tensor = event_tensor.to(device=_device, dtype=_dtype)

        S_next = F.silu(self.W_persist(S_t) + self.W_event(event_tensor))
        ctx.fluents.append(S_next)
        ctx.trace_log.append(f"EventCalculus: Transitioned to t={current_time_step + 1}")
        return ctx


class DoCalculusRule(DifferentiableRule):
    """Can thiệp vào đồ thị nhân quả (Graph Surgery)."""
    def forward(self, ctx: DifferentiableContext, intervention_node_idx: int):
        dense_graph = ctx.causal_graph.to_dense()
        dense_graph[:, intervention_node_idx] = 0.0 
        ctx.causal_graph = dense_graph.to_sparse()
        ctx.trace_log.append(f"DoCalculus: Intervention applied do(X={intervention_node_idx})")
        return ctx


# ===========================================================================
# 4. Dynamic Rule Discovery (Inductive Logic Programming)
# ===========================================================================

class DynamicRuleRegistry(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.core_rules = nn.ModuleDict({
            "EVENT_CALC": EventCalculusRule(hidden_dim),
            "DO_CALC":    DoCalculusRule(),
        })
        self.discovered_rules = nn.ModuleDict()

    @staticmethod
    def _safe_get(module_dict: nn.ModuleDict, key: str) -> Optional[nn.Module]:
        return module_dict[key] if key in module_dict else None

    def get_rule(self, rule_name: str) -> Optional[nn.Module]:
        rule = self._safe_get(self.core_rules, rule_name)
        if rule is None:
            rule = self._safe_get(self.discovered_rules, rule_name)
        return rule

    def discover_new_rule(self, rule_name: str):
        logger.info(f"ILP Triggered: Discovering new logical manifold for '{rule_name}'")
        new_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2, dtype=torch.bfloat16),
            nn.LayerNorm(self.hidden_dim * 2, dtype=torch.bfloat16),
            nn.GELU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim, dtype=torch.bfloat16)
        )
        self.discovered_rules[rule_name] = new_layer


# ===========================================================================
# 5. Utilities: Validation & Execution Node
# ===========================================================================

@dataclass
class BeamNode:
    """Đại diện cho một trạng thái trong cây tìm kiếm Beam Search."""
    ctx: DifferentiableContext
    program_remaining: List[Tuple[str, Dict]]
    credal_width: float  
    path_trace: str
    
    parent: Optional["BeamNode"] = field(default=None)
    rule_name: str = field(default="UNKNOWN")
    
    # BẮT BUỘC cho Component 3: The Calculus Fix (Tránh lỗi Type Float)
    confidence: float = field(default=1.0) 

    def __post_init__(self):
        # Chuyển đổi Credal Width thành Confidence
        self.confidence = float(1.0 - min(max(self.credal_width, 0.0), 1.0))


def _validate_llm_program(llm_program: Any) -> List[Tuple[str, Dict]]:
    """Validate và chuẩn hóa LLM program thô trước khi đưa vào Beam Search."""
    default_plan = [("EVENT_CALC", {"current_time_step": 0, "event_tensor": None})]

    if llm_program is None:
        return default_plan

    if isinstance(llm_program, dict):
        inner = llm_program.get("program", None)
        return _validate_llm_program(inner) if inner else default_plan

    if isinstance(llm_program, str):
        return default_plan

    if isinstance(llm_program, list):
        if len(llm_program) == 0:
            return default_plan
        
        validated = []
        for item in llm_program:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                rule_name, args = item[0], item[1]
                if not isinstance(args, dict):
                    args = {} 
                validated.append((str(rule_name), args))
        
        return validated if validated else default_plan

    return default_plan


# ===========================================================================
# 6. Engine Core: Uncertainty-Aware Beam Search Compiler
# ===========================================================================

class NeuroSymbolicCompiler(nn.Module): # <-- Kế thừa nn.Module theo chuẩn
    def __init__(self, hidden_dim: int, pis_evaluator: Any):
        super().__init__() # <-- Khởi tạo module
        self.hidden_dim = hidden_dim
        self.registry = DynamicRuleRegistry(hidden_dim)
        self.pis = pis_evaluator  # Component 3 (Probabilistic Inconsistency Signal)

    def compile(self, initial_ctx, logic_plan, **kwargs):
        """
        Hàm Wrapper để Master Brain gọi. Đảm bảo tham số đầu tiên là initial_ctx.
        """
        # Nếu Brain truyền 'question' thay vì 'logic_plan'
        if logic_plan is None and 'question' in kwargs:
            logic_plan = kwargs['question']
            
        return self.execute_with_beam_search(
            initial_ctx=initial_ctx, 
            llm_program=logic_plan, 
            **kwargs
        )

    def execute_with_beam_search(self, initial_ctx, llm_program, beam_width=3, **kwargs):
        """
        Sử dụng **kwargs để 'hứng' tất cả các tham số dư thừa, 
        tránh lỗi duplicate hoặc missing argument.
        """
        # Nếu trong kwargs có llm_program thì ưu tiên nó (đề phòng gọi nhầm)
        # Cú pháp này đảm bảo 'program_raw' luôn lấy được giá trị đúng nhất
        program_raw = kwargs.get('llm_program', llm_program)
        
        # Fallback thêm một lớp: nếu Brain dùng chữ 'question' thay cho logic plan
        if program_raw is None and 'question' in kwargs:
            program_raw = kwargs['question']

        # 1. Gọi hàm adapter chính xác theo chuẩn để lấy Tensor
        tensor_state = _coerce_to_context(initial_ctx, hidden_dim=self.hidden_dim)

        # 2. Bọc Tensor vào DifferentiableContext để khởi động Engine
        num_ents = tensor_state.shape[0] if tensor_state.dim() == 2 else 1
        ctx = DifferentiableContext(
            num_entities=num_ents, 
            hidden_dim=self.hidden_dim, 
            device=str(tensor_state.device)
        )
        
        # Gán giá trị cẩn thận để tránh lỗi shape mismatch
        if tensor_state.dim() == 1:
            length = min(self.hidden_dim, tensor_state.shape[0])
            ctx.entities[0, :length] = tensor_state[:length]
        else:
            ctx.entities = tensor_state
            
        ctx.trace_log.append(f"Adapter: Injected raw tensor (shape={tensor_state.shape}) into Context")

        # 3. Chuẩn hóa kế hoạch logic với biến 'program_raw' đã hứng được ở trên
        program = _validate_llm_program(program_raw)

        # 4. Khởi tạo Node gốc
        beam = [BeamNode(
            ctx=ctx, 
            program_remaining=program, 
            credal_width=0.0, 
            path_trace="Start",
            parent=None,
            rule_name="START"
        )]

        # 5. Vòng lặp Beam Search
        while beam and any(node.program_remaining for node in beam):
            next_beam = []

            for node in beam:
                if not node.program_remaining:
                    next_beam.append(node)
                    continue

                step = node.program_remaining[0]
                rule_name, args = step[0], step[1]

                rule_module = self.registry.get_rule(rule_name)

                if rule_module is None:
                    logger.warning(f"Rule '{rule_name}' not found. Triggering ILP discovery.")
                    self.registry.discover_new_rule(rule_name)
                    rule_module = self.registry.get_rule(rule_name)

                if rule_module is None:
                    node.ctx.semantic_errors.append(f"SemanticError: Operator '{rule_name}' undefined.")
                    continue

                try:
                    # Chạy logic khả vi
                    new_ctx = node.ctx.clone()
                    new_ctx = rule_module(new_ctx, **args)

                    # Đánh giá Độ bất định
                    if hasattr(self.pis, "compute_tensor_credal_width"):
                        new_credal_width = self.pis.compute_tensor_credal_width(new_ctx.entities)
                    else:
                        new_credal_width = node.credal_width + 0.05 

                    # Mở rộng cây (Lưu lại dấu vết parent & rule_name)
                    next_beam.append(BeamNode(
                        ctx=new_ctx,
                        program_remaining=node.program_remaining[1:],
                        credal_width=new_credal_width,
                        path_trace=node.path_trace + f" -> {rule_name}",
                        parent=node,           
                        rule_name=rule_name    
                    ))

                except Exception as e:
                    node.ctx.semantic_errors.append(f"RuntimeError in {rule_name}: {str(e)}")
                    logger.error(f"[Beam Search Error] {rule_name} raised: {e}")

            # Cắt tỉa nhánh
            next_beam.sort(key=lambda x: x.credal_width)
            beam = next_beam[:beam_width]

        # 6. Fallback an toàn nếu toàn bộ nhánh sập
        if not beam:
            logger.error("[execute_with_beam_search] All beam paths failed. Returning FAILED node.")
            empty_tensor = _coerce_to_context("FALLBACK", hidden_dim=self.hidden_dim)
            empty_ctx = DifferentiableContext(num_entities=1, hidden_dim=self.hidden_dim, device=str(empty_tensor.device))
            return BeamNode(
                ctx=empty_ctx, 
                program_remaining=[], 
                credal_width=1.0, 
                path_trace="FAILED",
                parent=None,
                rule_name="FAILED"
            )

        return beam[0]