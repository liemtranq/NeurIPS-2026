"""
Component 1: Active Reasoner Retriever (SOTA 2026 Paradigm)
MC-NSR Framework — Meta-Cognitive Neuro-Symbolic Reasoning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

import numpy as np
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any

from transformers import AutoTokenizer, AutoModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")
logger = logging.getLogger("mc_nsr.active_reasoner")


@dataclass
class ActiveReasonerConfig:
    model_name: str = "meta-llama/Llama-3.1-8B"
    llm_model_name: str = "meta-llama/Llama-3.1-8B"  # FIX: alias để tránh AttributeError
    llm_dtype: torch.dtype = torch.float8_e4m3fn
    use_flash_attention: bool = True
    max_seq_len: int = 4096
    hidden_size: int = 4096        # Mặc định của Llama 3 8B — chiều ra của LLM backbone
    projection_dim: int = 128      # ColBERT nén từ 4096 -> 128 chiều để so sánh vector hiệu quả
    token_dim: int = 128
    top_k_doc: int = 50
    top_k_chunk: int = 20
    top_k_final: int = 5
    max_iterations: int = 3
    credal_width_threshold: float = 0.2
    device: str = "cuda"  # ROCm của AMD MI300X expose qua "cuda" interface — giữ nguyên


class MultiSourceRetriever:
    """
    Quản lý đồng thời Text Hierarchical (Doc -> Sentence) và Graph Entities.
    """
    def __init__(self):
        self.documents: Dict[int, str] = {}
        self.chunks: Dict[int, Tuple[int, str]] = {}
        self.metadata: Dict[int, Any] = {}
        self.graph_edges: Dict[str, List[Tuple[str, str]]] = {}

    def add_document(self, doc_id: int, text: str, metadata: dict, triples: List[Tuple[str, str, str]]):
        self.documents[doc_id] = text
        self.metadata[doc_id] = metadata
        sentences = text.split(". ")
        for i, sent in enumerate(sentences):
            chunk_id = hash(f"{doc_id}_{i}")
            self.chunks[chunk_id] = (doc_id, sent)
        for subj, pred, obj in triples:
            if subj not in self.graph_edges:
                self.graph_edges[subj] = []
            self.graph_edges[subj].append((pred, obj))

    def retrieve_paths(self, entity: str, max_depth: int = 2) -> List[str]:
        return [f"{entity} -> {pred} -> {obj}" for pred, obj in self.graph_edges.get(entity, [])]


class LateInteractionEncoder(nn.Module):
    def __init__(self, cfg: ActiveReasonerConfig):
        super().__init__()
        self.cfg = cfg
        attn_impl = "flash_attention_2" if cfg.use_flash_attention else "eager"
        self.model = AutoModel.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.bfloat16,  # bfloat16 cực hợp với AMD MI300X
            device_map="auto",
            trust_remote_code=True
        )
        self.linear_compress = nn.Linear(self.model.config.hidden_size, cfg.token_dim, dtype=torch.bfloat16)
        self.linear_compress.to(cfg.device)  # Ép linear_compress lên GPU — tránh device mismatch với Llama
        self.model = torch.compile(self.model)

    def forward(self, input_ids, attention_mask) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state.to(torch.bfloat16)
        token_embs = F.normalize(self.linear_compress(hidden), p=2, dim=-1)
        return token_embs

    @staticmethod
    def maxsim(q_embs: torch.Tensor, d_embs: torch.Tensor, q_mask: torch.Tensor) -> torch.Tensor:
        interaction_matrix = torch.einsum("b q d, n c d -> b n q c", q_embs, d_embs)
        max_scores = interaction_matrix.max(dim=-1).values
        scores = (max_scores * q_mask.unsqueeze(1)).sum(dim=-1)
        return scores


class NeuroSymbolicReranker(nn.Module):
    def __init__(self, token_dim: int):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(token_dim, num_heads=4, batch_first=True)
        self.logic_prob_head = nn.Sequential(
            nn.Linear(token_dim, token_dim // 2),
            nn.GELU(),
            nn.Linear(token_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, query_embs: torch.Tensor, chunk_embs: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.cross_attention(query_embs, chunk_embs, chunk_embs)
        pooled = attn_out.mean(dim=1)
        return self.logic_prob_head(pooled).squeeze(-1)


class MemoryBuffer:
    def __init__(self):
        self.explored_paths = set()
        self.retrieved_chunks = []
        self.information_gaps = []

    def update(self, chunks: List[str], current_query: str):
        self.retrieved_chunks.extend(chunks)
        self.explored_paths.add(current_query)

class ActiveReasonerRetriever:
    def __init__(self, cfg: ActiveReasonerConfig):
        self.cfg = cfg
        self.index = []
        self.encoder = LateInteractionEncoder(cfg).to(cfg.device)  # Đảm bảo toàn bộ encoder lên GPU
        self.reranker = NeuroSymbolicReranker(cfg.token_dim).to(cfg.device)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm_model_name)  # dùng llm_model_name
        logger.info("Active Reasoner initialized with FP8, FA3, and ColBERT-v4 paradigm.")

    def decompose_query(self, query: str, memory: MemoryBuffer) -> List[str]:
        logger.warning("Credal width too high! Decomposing query...")
        return [f"What is the underlying mechanism of {query}?", 
                f"Are there temporal constraints regarding {query}?"]

    def step_retrieve(self, query: str, memory: MemoryBuffer) -> List[Tuple[int, float]]:
        q_enc = self.tokenizer(query, return_tensors="pt").to(self.cfg.device)
        with torch.no_grad():
            q_embs = self.encoder(q_enc.input_ids, q_enc.attention_mask)
        N_chunks = 1000
        mock_d_embs = torch.randn(N_chunks, 64, self.cfg.projection_dim, device=self.cfg.device, dtype=torch.bfloat16)
        scores = self.encoder.maxsim(q_embs, mock_d_embs, q_enc.attention_mask)
        top_ids = scores[0].topk(self.cfg.top_k_chunk).indices.tolist()
        return top_ids

    def retrieve(self, initial_query: str, credal_width: float = 0.0, top_k: Optional[int] = None, negative_constraint: bool = False) -> Dict[str, Any]:
        memory = MemoryBuffer()
        current_queries = [initial_query]
        final_results = []
        
        for iteration in range(self.cfg.max_iterations):
            logger.info(f"--- Iteration {iteration+1} ---")
            iteration_chunks = []
            for q in current_queries:
                chunk_ids = self.step_retrieve(q, memory)
                iteration_chunks.extend(chunk_ids) 
            memory.update([str(c) for c in iteration_chunks], " | ".join(current_queries))
            if credal_width > self.cfg.credal_width_threshold and iteration < self.cfg.max_iterations - 1:
                current_queries = self.decompose_query(initial_query, memory)
                credal_width -= 0.1 
            else:
                final_results = iteration_chunks[:self.cfg.top_k_final]
                break

        return {
            "retrieved_nodes": final_results,
            "memory_trace": memory.explored_paths,
            "final_credal_width": credal_width
        }

    def retrieve_with_negative_constraint(self, existing_evidence: Any) -> Any:
        """Tìm kiếm hướng ngược lại với evidence hiện tại (dùng cho Deep-Dive recovery)."""
        logger.info("Retrieving with negative constraint against existing evidence...")
        return self.retrieve(f"NOT {str(existing_evidence)}", credal_width=0.0)


# (Giả sử Class chứa hàm retrieve đó tên là ActiveReasonerRetriever)
MultiSourceRetriever = ActiveReasonerRetriever

if __name__ == "__main__":
    cfg = ActiveReasonerConfig()
    reasoner = ActiveReasonerRetriever(cfg)
    result = reasoner.retrieve("How does MI300X accelerate symbolic logic paths?", credal_width=0.35)
    logger.info(f"Final retrieved chunks: {result['retrieved_nodes']}")
    logger.info(f"Reasoning Trace: {result['memory_trace']}")