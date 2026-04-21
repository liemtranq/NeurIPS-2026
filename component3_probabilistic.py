"""
Component 3 — Probabilistic Inconsistency Signal (PIS)
=====================================================
NeurIPS 2026 Submission

Novelty: Replace scalar inconsistency I(R,P) with credal interval [L, U]
inspired by Logical Credal Networks (Marinescu et al., NeurIPS 2022),
and add I_meta computed from LLM hidden states following the neurofeedback
paradigm of Ji-An et al. (NeurIPS 2025).

This is the first integration of credal-interval inconsistency + LLM
metacognitive signals in Neuro-Symbolic QA.

Hardware target: AMD MI300X 192GB — Llama-3.1-70B full precision fits in 1x GPU.

Fixes applied (vs. original draft):
  FIX-1  _aggregate_independent: lower/upper bounds were swapped.
          Correct derivation:
            prod_complement_upper = ∏(1 - l_i)   [pessimistic complement]
            prod_complement_lower = ∏(1 - u_i)   [optimistic complement]
            L_agg = 1 - prod_complement_upper
            U_agg = 1 - prod_complement_lower
  FIX-2  LRAxisTrainer.collect_activations: hidden states not moved to CPU
          before appending, causing VRAM accumulation with large datasets.
  FIX-3  compute_batch: was sequential; now uses ThreadPoolExecutor to
          parallelise credal computation across samples (CPU-bound).
  FIX-4  _classify_type: extended to cover all 40 rule names from Component 2
          (BRIDGE, PATH, SEQUENCE, SPAN, DIFF, RATIO, etc.).
  FIX-5  ProbabilisticInconsistencySignal.compute: mutation_threshold exposed
          via config so it can be tuned on dev set without editing source.
  FIX-6  MetaCognitiveExtractor.train_lr_axes: moved model to eval() outside
          the loop to avoid repeated mode switches; added weight_decay to Adam.
"""

from __future__ import annotations

import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============================================================================
# Section 1: Data Structures
# ============================================================================

class InconsistencyType(Enum):
    """Categories of inconsistency detected between retrieval and proof."""
    NONE        = "none"
    FACTUAL     = "factual"       # Retrieved fact contradicts symbolic proof
    TEMPORAL    = "temporal"      # Temporal ordering violation
    CAUSAL      = "causal"        # Causal chain break
    COMPARATIVE = "comparative"   # Comparative relation mismatch
    LOGICAL     = "logical"       # Pure logical contradiction


@dataclass
class CredalInterval:
    """
    A credal interval [lower, upper] representing imprecise probability.

    Inspired by LCN (Marinescu et al., 2022): instead of a single P(q),
    we maintain bounds  l_q ≤ P(q) ≤ u_q.

    Invariant (enforced in __post_init__):
        0 ≤ lower ≤ upper ≤ 1
    """
    lower: float
    upper: float

    def __post_init__(self) -> None:
        lo = float(np.clip(self.lower, 0.0, 1.0))
        hi = float(np.clip(self.upper, 0.0, 1.0))
        # Defensive swap so callers never get an inverted interval
        self.lower = min(lo, hi)
        self.upper = max(lo, hi)

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def midpoint(self) -> float:
        return (self.lower + self.upper) / 2.0

    @property
    def width(self) -> float:
        """Epistemic uncertainty — wider means less certain."""
        return self.upper - self.lower

    @property
    def is_precise(self) -> bool:
        return self.width < 1e-6

    # ------------------------------------------------------------------
    # Interval algebra
    # ------------------------------------------------------------------

    def intersect(self, other: "CredalInterval") -> Optional["CredalInterval"]:
        """
        Intersection of two credal intervals.
        Returns None if the intervals are disjoint (empty intersection),
        which is a hard inconsistency signal.
        """
        lo = max(self.lower, other.lower)
        hi = min(self.upper, other.upper)
        if lo > hi + 1e-9:
            return None            # Disjoint → definite inconsistency
        return CredalInterval(lo, hi)

    def contains(self, value: float) -> bool:
        return self.lower - 1e-9 <= value <= self.upper + 1e-9

    def __repr__(self) -> str:
        return f"[{self.lower:.4f}, {self.upper:.4f}]"


@dataclass
class RetrievedEvidence:
    """A single piece of retrieved evidence with confidence bounds."""
    text:             str
    source_id:        str
    confidence:       CredalInterval   # Retrieval confidence as credal interval
    relevance_score:  float            # Dense retrieval score
    sparse_score:     float            # BM25 score
    hop_depth:        int = 1          # Which hop this came from


@dataclass
class SymbolicProofStep:
    """A single step in the symbolic proof chain (from Component 2)."""
    rule_name:  str
    premises:   List[str]
    conclusion: str
    confidence: CredalInterval
    step_index: int = 0


@dataclass
class InconsistencySignal:
    """
    The full inconsistency signal combining credal interval + metacognitive
    signal.  This is what gets passed to Component 4 (Bandit Mutation Engine).
    """
    # Credal inconsistency interval — replaces scalar I(R,P)
    credal_inconsistency: CredalInterval

    # Metacognitive signal from LLM hidden states (Ji-An et al.)
    i_meta: float

    # Combined signal
    combined_score: float

    # Breakdown by inconsistency type
    type_scores: Dict[InconsistencyType, CredalInterval] = field(
        default_factory=dict
    )

    # Whether to trigger additional mutation round (Component 4)
    trigger_mutation: bool = False

    # Raw diagnostics for logging / ablation
    diagnostics: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Section 2: Credal Inconsistency Engine (LCN-inspired)
# ============================================================================

class CredalInconsistencyEngine:
    """
    Computes credal interval [L, U] for inconsistency between retrieved
    evidence E and symbolic proof chain P.

    Inspired by Logical Credal Networks (Marinescu et al., NeurIPS 2022):
      - Each evidence piece and proof step carries probability bounds.
      - Inconsistency = degree to which the credal sets fail to intersect.
      - Three aggregation modes: "markov" | "independent" | "worst_case".

    Key difference from full LCN: we do not solve the NLP over all 2^n
    interpretations (intractable at scale).  Instead we decompose into
    pairwise consistency checks and aggregate with a generalized Markov
    condition over the proof DAG.
    """

    def __init__(
        self,
        aggregation: str = "markov",   # "markov" | "independent" | "worst_case"
        epsilon: float = 1e-6,
    ) -> None:
        assert aggregation in ("markov", "independent", "worst_case"), \
            f"Unknown aggregation mode: {aggregation!r}"
        self.aggregation = aggregation
        self.epsilon = epsilon

    # ------------------------------------------------------------------
    # Pairwise inconsistency
    # ------------------------------------------------------------------

    def compute_pairwise_inconsistency(
        self,
        evidence:   RetrievedEvidence,
        proof_step: SymbolicProofStep,
    ) -> CredalInterval:
        """
        Compute credal inconsistency between one evidence piece and one
        proof step.

        Following LCN semantics:
          - Disjoint credal sets → definite inconsistency.
            L_inc = gap between intervals (minimum possible inconsistency).
            U_inc = full span of union  (maximum possible inconsistency).
          - Overlapping credal sets → partial inconsistency.
            L_inc = 0   (consistent assignment exists).
            U_inc = 1 - overlap_width / max_width.
        """
        e = evidence.confidence
        p = proof_step.confidence
        intersection = e.intersect(p)

        if intersection is None:
            # Disjoint intervals → hard inconsistency
            gap  = max(0.0, e.lower - p.upper, p.lower - e.upper)
            span = max(e.upper, p.upper) - min(e.lower, p.lower)
            return CredalInterval(
                lower=min(1.0, gap),
                upper=min(1.0, max(gap, span)),   # U ≥ L always
            )
        else:
            # Partial inconsistency from non-overlapping mass
            total_width   = max(e.width, p.width, self.epsilon)
            overlap_width = intersection.width
            u_inc = float(np.clip(1.0 - overlap_width / total_width, 0.0, 1.0))
            return CredalInterval(lower=0.0, upper=u_inc)

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    def _aggregate_independent(
        self, pairwise: List[CredalInterval]
    ) -> CredalInterval:
        """
        Aggregate assuming independence between evidence–proof pairs.

        P(any inconsistency) = 1 - ∏ P(no inconsistency_i)

        With credal intervals the bound propagation is:

            U_agg = 1 - ∏(1 - l_i)   ← pessimistic: each l_i is smallest
                                         possible individual P(consistent_i)
            L_agg = 1 - ∏(1 - u_i)   ← optimistic: each u_i is largest
                                         possible individual P(consistent_i)

        FIX-1: original code had prod_lower/prod_upper swapped.
        """
        if not pairwise:
            return CredalInterval(0.0, 0.0)

        # Product of (1 - u_i): optimistic complement → gives L_agg
        prod_opt = 1.0
        # Product of (1 - l_i): pessimistic complement → gives U_agg
        prod_pes = 1.0
        for ci in pairwise:
            prod_opt *= (1.0 - ci.upper)   # optimistic  (smallest inconsistency)
            prod_pes *= (1.0 - ci.lower)   # pessimistic (largest  inconsistency)

        l_agg = float(np.clip(1.0 - prod_pes, 0.0, 1.0))
        u_agg = float(np.clip(1.0 - prod_opt, 0.0, 1.0))
        return CredalInterval(lower=l_agg, upper=u_agg)

    def _aggregate_worst_case(
        self, pairwise: List[CredalInterval]
    ) -> CredalInterval:
        """
        Worst-case (no independence assumption):
            L = min(l_i),  U = max(u_i).
        Conservative but never wrong.
        """
        if not pairwise:
            return CredalInterval(0.0, 0.0)
        return CredalInterval(
            lower=min(ci.lower for ci in pairwise),
            upper=max(ci.upper for ci in pairwise),
        )

    def _aggregate_markov(
        self,
        pairwise:    List[CredalInterval],
        proof_steps: List[SymbolicProofStep],
    ) -> CredalInterval:
        """
        Aggregate using a generalized Markov condition over the proof DAG.

        Key idea from LCN (Def. 5): each proof step is conditionally
        independent of non-descendant, non-parent steps given its parents.
        We proxy DAG depth with step_index.

        Within each depth layer  → worst_case  (steps are correlated).
        Across depth layers      → independent (steps are causally separated).
        """
        if not pairwise:
            return CredalInterval(0.0, 0.0)

        # Group by step_index (proxy for DAG depth layer)
        depth_groups: Dict[int, List[CredalInterval]] = {}
        for ci, step in zip(pairwise, proof_steps):
            depth_groups.setdefault(step.step_index, []).append(ci)

        # Within each depth: worst-case (dependent steps)
        layer_intervals = [
            self._aggregate_worst_case(depth_groups[d])
            for d in sorted(depth_groups)
        ]

        # Across depths: independent aggregation
        return self._aggregate_independent(layer_intervals)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def compute(
        self,
        evidence_set: List[RetrievedEvidence],
        proof_chain:  List[SymbolicProofStep],
    ) -> Tuple[CredalInterval, Dict[InconsistencyType, CredalInterval]]:
        """
        Compute overall credal inconsistency interval + per-type breakdown.

        Args:
            evidence_set: Retrieved evidence from Component 1.
            proof_chain:  Symbolic proof from Component 2.

        Returns:
            (overall_interval, per_type_breakdown)
        """
        if not evidence_set or not proof_chain:
            return CredalInterval(0.0, 0.0), {}

        # Pairwise inconsistencies + type classification
        pairwise_all: List[CredalInterval] = []
        type_buckets: Dict[InconsistencyType, List[CredalInterval]] = {
            t: [] for t in InconsistencyType if t != InconsistencyType.NONE
        }

        for evidence in evidence_set:
            for step in proof_chain:
                ci       = self.compute_pairwise_inconsistency(evidence, step)
                inc_type = self._classify_type(step.rule_name)
                pairwise_all.append(ci)
                if inc_type != InconsistencyType.NONE:
                    type_buckets[inc_type].append(ci)

        # Overall aggregation
        if self.aggregation == "markov":
            # Expand pairwise list to match proof steps (|evidence| copies each)
            expanded_steps = proof_chain * len(evidence_set)
            overall = self._aggregate_markov(pairwise_all, expanded_steps)
        elif self.aggregation == "independent":
            overall = self._aggregate_independent(pairwise_all)
        else:
            overall = self._aggregate_worst_case(pairwise_all)

        # Per-type breakdown (independent aggregation within each type)
        type_scores: Dict[InconsistencyType, CredalInterval] = {
            t: self._aggregate_independent(cis)
            for t, cis in type_buckets.items()
            if cis
        }

        return overall, type_scores

    # ------------------------------------------------------------------
    # Rule → inconsistency type mapping
    # FIX-4: extended to cover all 40 ops from Component 2
    # ------------------------------------------------------------------

    _TEMPORAL_KEYWORDS = frozenset([
        "temporal", "before", "after", "during", "since", "until",
        "order_time", "span", "sequence",
    ])
    _CAUSAL_KEYWORDS = frozenset([
        "caus", "implies", "because", "enable", "prevent", "require",
        "effect", "mediat", "confounder", "instrument",
    ])
    _COMPARATIVE_KEYWORDS = frozenset([
        "compar", "greater", "less", "above", "below", "max", "min",
        "rank", "diff", "ratio", "threshold", "flip", "equaliz",
        "normaliz",
    ])
    _LOGICAL_KEYWORDS = frozenset([
        "contradict", "negat", "negate", "contrapos", "tollens",
        "double_neg", "disjunct", "conjunct", "universal", "existential",
        "logical", "implication",
    ])
    _FACTUAL_KEYWORDS = frozenset([
        "fact", "entity", "find", "filter", "bridge", "path",
        "link", "join", "relate", "verify", "exists", "select",
        "project", "resolve", "clarify", "explain",
    ])

    @classmethod
    def _classify_type(cls, rule_name: str) -> InconsistencyType:
        """Map a symbolic rule name to an InconsistencyType."""
        rn = rule_name.lower()
        # Check in priority order (most specific first)
        for kw in cls._TEMPORAL_KEYWORDS:
            if kw in rn:
                return InconsistencyType.TEMPORAL
        for kw in cls._CAUSAL_KEYWORDS:
            if kw in rn:
                return InconsistencyType.CAUSAL
        for kw in cls._COMPARATIVE_KEYWORDS:
            if kw in rn:
                return InconsistencyType.COMPARATIVE
        for kw in cls._LOGICAL_KEYWORDS:
            if kw in rn:
                return InconsistencyType.LOGICAL
        for kw in cls._FACTUAL_KEYWORDS:
            if kw in rn:
                return InconsistencyType.FACTUAL
        return InconsistencyType.NONE


# ============================================================================
# Section 3: Metacognitive Signal Extractor  (Ji-An et al. inspired)
# ============================================================================

class MetaCognitiveExtractor(nn.Module):
    """
    Extracts I_meta from LLM hidden states following the neurofeedback
    paradigm of Ji-An et al. (NeurIPS 2025).

    Key ideas implemented:
      1. Extract residual stream activations from layers 24-32 of
         Llama-3.1-70B (middle-to-late layers show strongest control
         effects per Fig. 5a of Ji-An et al.).
      2. Project onto LR axis (logistic regression direction) trained to
         predict consistency/inconsistency from internal activations.
      3. Soft binarisation → scalar I_meta ∈ [0, 1].

    The LR axis is trained offline on (question, evidence, proof) tuples
    where ground-truth consistency labels are available.

    This implements the "second-order metacognitive process" (Ji-An et al.
    Sec. A.4): the LLM's first-order process handles QA; the second-order
    process monitors whether internal activations signal inconsistency.
    """

    def __init__(
        self,
        hidden_dim:           int   = 8192,   # Llama-3.1-70B hidden dim
        n_layers_to_extract:  int   = 9,      # Layers 24–32 (0-indexed)
        start_layer:          int   = 24,
        n_bins:               int   = 2,      # Binary labels (extendable to 8)
        device:               str   = "cuda",
    ) -> None:
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.n_layers    = n_layers_to_extract
        self.start_layer = start_layer
        self.n_bins      = n_bins
        self.device      = device

        # LR axis: one direction vector per target layer
        self.lr_axes = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_dim) * 0.01)
            for _ in range(n_layers_to_extract)
        ])

        # Binarisation threshold θ^l per layer (initialised to 0)
        self.thresholds = nn.Parameter(torch.zeros(n_layers_to_extract))

        # Cross-layer importance weights: deeper → more weight (Fig. 5a)
        self.layer_weights = nn.Parameter(
            torch.linspace(0.5, 1.5, n_layers_to_extract)
        )

        # Aggregator MLP: (n_layers,) → I_meta scalar
        self.aggregator = nn.Sequential(
            nn.Linear(n_layers_to_extract, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),             # I_meta ∈ [0, 1]
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def extract_residual_streams(
        self, model_outputs
    ) -> List[torch.Tensor]:
        """
        Extract mean-pooled residual stream h̄^l from target layers.

        Per Ji-An et al. Sec. 2.3:
            h̄^l = mean_{t} h^l_{i,t}      (average over token positions)
        """
        hidden_states = model_outputs.hidden_states
        extracted = []
        for offset in range(self.n_layers):
            layer_idx = self.start_layer + offset
            if layer_idx >= len(hidden_states):
                layer_idx = len(hidden_states) - 1
            h_bar = hidden_states[layer_idx].mean(dim=1)   # (batch, hidden_dim)
            extracted.append(h_bar)
        return extracted

    def project_onto_lr_axes(
        self, residual_streams: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Project each layer's h̄^l onto its LR axis w^l.

        Per Ji-An et al. Sec. 2.3:
            a^l_i = (w^l)^T h̄^l_i

        Returns: (batch, n_layers)
        """
        projections = []
        for offset, h_bar in enumerate(residual_streams):
            w      = self.lr_axes[offset]
            w_norm = w / (w.norm() + 1e-8)
            a_l    = torch.matmul(h_bar, w_norm)   # (batch,)
            projections.append(a_l)
        return torch.stack(projections, dim=-1)    # (batch, n_layers)

    def binarize(self, projections: torch.Tensor) -> torch.Tensor:
        """
        Soft binarisation:  y^l = H(a^l - θ^l)   (Ji-An et al. Sec. 2.3)

        We use sigmoid with temperature=5 for differentiability during
        LR-axis training.
        """
        temperature = 5.0
        return torch.sigmoid(
            (projections - self.thresholds.unsqueeze(0)) * temperature
        )   # (batch, n_layers)

    def forward(
        self, model_outputs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute I_meta ∈ [0, 1] from LLM hidden states.

        Returns:
            i_meta:      (batch,)
            diagnostics: per-layer projections / labels / weights
        """
        streams    = self.extract_residual_streams(model_outputs)
        proj       = self.project_onto_lr_axes(streams)          # (B, n_layers)
        labels     = self.binarize(proj)                          # (B, n_layers)
        weights    = F.softmax(self.layer_weights, dim=0)
        weighted   = labels * weights.unsqueeze(0)               # (B, n_layers)
        i_meta     = self.aggregator(weighted).squeeze(-1)        # (B,)

        diagnostics = {
            "per_layer_projections": proj.detach(),
            "per_layer_labels":      labels.detach(),
            "layer_weights":         weights.detach(),
        }
        return i_meta, diagnostics

    # ------------------------------------------------------------------
    # Offline LR-axis training
    # FIX-6: eval() called once outside loop; weight_decay added to Adam
    # ------------------------------------------------------------------

    def train_lr_axes(
        self,
        dataset:  List[Tuple[List[torch.Tensor], int]],
        n_epochs: int   = 50,
        lr:       float = 1e-3,
    ) -> None:
        """
        Train LR axes + aggregator on labelled
        (per_layer_activations, consistency_label) pairs.

        Labels:  0 = consistent,  1 = inconsistent.

        Per Ji-An et al. Sec. 2.5: fit a logistic regression at each
        layer to predict dataset labels from that layer's activations.
        """
        optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=1e-5   # FIX-6
        )
        criterion = nn.BCELoss()

        self.train()                         # FIX-6: set once before loop
        for epoch in range(n_epochs):
            total_loss = 0.0
            for hidden_states_tuple, label in dataset:
                projections_list = []
                for offset in range(self.n_layers):
                    w      = self.lr_axes[offset]
                    w_norm = w / (w.norm() + 1e-8)
                    h      = hidden_states_tuple[offset]   # (1, hidden_dim)
                    a      = torch.matmul(h, w_norm)       # (1,)
                    projections_list.append(a)

                proj        = torch.stack(projections_list, dim=-1)   # (1, n_layers)
                labels_soft = self.binarize(proj)
                weights     = F.softmax(self.layer_weights, dim=0)
                weighted    = labels_soft * weights.unsqueeze(0)
                i_meta_out  = self.aggregator(weighted).squeeze(-1)   # (1,)
                target      = torch.tensor(
                    [float(label)], device=i_meta_out.device
                )
                loss = criterion(i_meta_out, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                logger.info(
                    "LR axis training epoch %d/%d  loss=%.4f",
                    epoch + 1, n_epochs,
                    total_loss / max(len(dataset), 1),
                )

        self.eval()


# ============================================================================
# Section 4: LR Axis Trainer  (offline, before inference)
# ============================================================================

class LRAxisTrainer:
    """
    Offline trainer for MetaCognitiveExtractor LR axes.

    Per Ji-An et al. Sec. 2.5:
        "We fit LR at each layer to predict original dataset labels,
         using that layer's activations across dataset sentences."

    For our NeSy-QA setting:
        - Dataset = (question + evidence + proof) tuples.
        - Labels  = {0: consistent, 1: inconsistent}.

    After training, the LR weight vectors become w^l in MetaCognitiveExtractor.
    """

    def __init__(
        self,
        meta_extractor: MetaCognitiveExtractor,
        device:         str = "cuda",
    ) -> None:
        self.meta_extractor = meta_extractor
        self.device         = device

    def collect_activations(
        self,
        model,
        tokenizer,
        texts:      List[str],
        labels:     List[int],
        batch_size: int = 8,
    ) -> List[Tuple[List[torch.Tensor], int]]:
        """
        Run texts through the LLM, collect hidden states per target layer.

        FIX-2: tensors are moved to CPU immediately after extraction to
        avoid accumulating gigabytes of VRAM across large datasets.

        Returns:
            List of (per_layer_activations_on_CPU, label) pairs.
        """
        model.eval()
        dataset: List[Tuple[List[torch.Tensor], int]] = []

        for i in range(0, len(texts), batch_size):
            batch_texts  = texts[i : i + batch_size]
            batch_labels = labels[i : i + batch_size]

            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            hidden_states = outputs.hidden_states   # tuple of (B, L, D)

            for j in range(len(batch_texts)):
                per_layer: List[torch.Tensor] = []
                for offset in range(self.meta_extractor.n_layers):
                    layer_idx = self.meta_extractor.start_layer + offset
                    if layer_idx >= len(hidden_states):
                        layer_idx = len(hidden_states) - 1
                    # Mean over token positions, then move to CPU (FIX-2)
                    h = (
                        hidden_states[layer_idx][j]
                        .mean(dim=0, keepdim=True)
                        .cpu()               # ← FIX-2
                    )
                    per_layer.append(h)
                dataset.append((per_layer, batch_labels[j]))

            logger.debug("Collected activations %d/%d", i + len(batch_texts), len(texts))

        return dataset

    def train(
        self,
        model,
        tokenizer,
        texts:      List[str],
        labels:     List[int],
        n_epochs:   int   = 50,
        lr:         float = 1e-3,
        batch_size: int   = 8,
    ) -> None:
        """
        Full training pipeline:
          1. Collect activations from LLM (GPU, batched).
          2. Train LR axes + aggregator (CPU-resident tensors).
        """
        logger.info("Collecting activations from LLM (%d samples)…", len(texts))
        dataset = self.collect_activations(
            model, tokenizer, texts, labels, batch_size
        )
        logger.info("Training LR axes on %d samples for %d epochs…", len(dataset), n_epochs)
        self.meta_extractor.train_lr_axes(dataset, n_epochs=n_epochs, lr=lr)
        logger.info("LR axis training complete.")


# ============================================================================
# Section 5: Combined Inconsistency Signal
# ============================================================================

class ProbabilisticInconsistencySignal:
    """
    Component 3 main class — ties CredalInconsistencyEngine + MetaCognitiveExtractor
    together into a single InconsistencySignal consumed by Component 4.

    Combination formula
    -------------------
        combined = α * credal.midpoint + (1 - α) * I_meta

    Adaptive α:
        When credal width is large (high epistemic uncertainty in symbolic
        reasoning) we rely more on I_meta and reduce α.
        When credal width is small (confident symbolic reasoning) we trust
        the credal midpoint and increase α.
        α is clamped to [alpha_min, alpha_max] so neither signal dominates.

    Mutation trigger:
        Trigger Component 4 mutation if:
          combined > mutation_threshold   OR
          |credal.midpoint - I_meta| > disagreement_threshold
        The disagreement trigger catches cases where symbolic and neural
        pathways see different things — exactly where NeSy adds value.

    FIX-5: mutation_threshold and disagreement_threshold are constructor
    parameters (not hardcoded) so they can be tuned on a dev set.
    """

    def __init__(
        self,
        alpha_base:              float = 0.6,
        alpha_min:               float = 0.2,
        alpha_max:               float = 0.8,
        mutation_threshold:      float = 0.5,    # FIX-5
        disagreement_threshold:  float = 0.4,    # FIX-5
        meta_extractor:          Optional[MetaCognitiveExtractor] = None,
        aggregation:             str   = "markov",
        device:                  str   = "cuda",
    ) -> None:
        self.alpha_base             = alpha_base
        self.alpha_min              = alpha_min
        self.alpha_max              = alpha_max
        self.mutation_threshold     = mutation_threshold
        self.disagreement_threshold = disagreement_threshold
        self.device                 = device

        self.credal_engine  = CredalInconsistencyEngine(aggregation=aggregation)
        self.meta_extractor = meta_extractor or MetaCognitiveExtractor(
            device=device
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _adaptive_alpha(self, credal_ci: CredalInterval) -> float:
        """
        α = α_base × (1 - width),  clamped to [alpha_min, alpha_max].

        Wider credal interval → more uncertainty → lower α → trust I_meta more.
        """
        alpha = self.alpha_base * (1.0 - credal_ci.width)
        return float(np.clip(alpha, self.alpha_min, self.alpha_max))

    # ------------------------------------------------------------------
    # Single sample
    # ------------------------------------------------------------------

    def compute(
        self,
        evidence_set: List[RetrievedEvidence],
        proof_chain:  List[SymbolicProofStep],
        llm_outputs=None,   # HuggingFace output with output_hidden_states=True
    ) -> InconsistencySignal:
        """
        Compute the full inconsistency signal for one (evidence, proof) pair.

        Args:
            evidence_set: from Component 1 (Hybrid Adaptive Retriever).
            proof_chain:  from Component 2 (Symbolic Engine).
            llm_outputs:  Llama-3.1-70B forward pass output
                          (with output_hidden_states=True).
                          Pass None to skip I_meta (credal-only mode).

        Returns:
            InconsistencySignal
        """
        # --- Part 1: Credal Inconsistency ---
        credal_ci, type_scores = self.credal_engine.compute(
            evidence_set, proof_chain
        )

        # --- Part 2: Metacognitive Signal ---
        i_meta_value    = 0.0
        meta_diagnostics: Dict[str, Any] = {}

        if llm_outputs is not None and hasattr(llm_outputs, "hidden_states"):
            with torch.no_grad():
                self.meta_extractor.eval()
                i_meta_tensor, meta_diagnostics = self.meta_extractor(llm_outputs)
                i_meta_value = float(i_meta_tensor.mean().item())

        # --- Part 3: Adaptive combination ---
        alpha    = self._adaptive_alpha(credal_ci)
        combined = alpha * credal_ci.midpoint + (1.0 - alpha) * i_meta_value

        # --- Part 4: Mutation trigger ---
        disagreement = abs(credal_ci.midpoint - i_meta_value)
        trigger = (
            combined > self.mutation_threshold
            or disagreement > self.disagreement_threshold
        )

        # Build diagnostics dict
        diagnostics: Dict[str, Any] = {
            "alpha":           alpha,
            "credal_midpoint": credal_ci.midpoint,
            "credal_width":    credal_ci.width,
            "i_meta":          i_meta_value,
            "combined":        combined,
            "disagreement":    disagreement,
        }
        for key, val in meta_diagnostics.items():
            if isinstance(val, torch.Tensor):
                diagnostics[f"meta_{key}"] = val.cpu().numpy().tolist()

        return InconsistencySignal(
            credal_inconsistency=credal_ci,
            i_meta=i_meta_value,
            combined_score=float(combined),
            type_scores=type_scores,
            trigger_mutation=trigger,
            diagnostics=diagnostics,
        )

    # ------------------------------------------------------------------
    # Batch computation
    # FIX-3: credal computation parallelised with ThreadPoolExecutor
    # ------------------------------------------------------------------

    def compute_batch(
        self,
        batch_evidence: List[List[RetrievedEvidence]],
        batch_proofs:   List[List[SymbolicProofStep]],
        llm_outputs=None,
        max_workers:    int = 4,
    ) -> List[InconsistencySignal]:
        """
        Batch computation for MI300X parallelism.

        FIX-3: credal computation is CPU-bound; we run it in parallel
        with ThreadPoolExecutor.  I_meta is GPU-bound and processed once
        via a single batched forward pass (if llm_outputs provided).

        Args:
            batch_evidence: list of evidence lists, one per question.
            batch_proofs:   list of proof chains, one per question.
            llm_outputs:    single batched HF output (batch_size = len batch).
            max_workers:    number of CPU threads for credal computation.
        """
        n = len(batch_evidence)
        assert len(batch_proofs) == n, "batch_evidence and batch_proofs must match"

        # --- GPU path: extract I_meta for entire batch at once ---
        i_meta_values: List[float] = [0.0] * n
        if llm_outputs is not None and hasattr(llm_outputs, "hidden_states"):
            with torch.no_grad():
                self.meta_extractor.eval()
                i_meta_tensor, _ = self.meta_extractor(llm_outputs)  # (B,)
                for idx in range(min(n, i_meta_tensor.shape[0])):
                    i_meta_values[idx] = float(i_meta_tensor[idx].item())

        # --- CPU path: credal computation in parallel ---
        credal_results: List[Tuple[CredalInterval, Dict]] = [None] * n  # type: ignore

        def _compute_credal(idx: int):
            return idx, self.credal_engine.compute(
                batch_evidence[idx], batch_proofs[idx]
            )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_compute_credal, i): i for i in range(n)}
            for future in as_completed(futures):
                idx, result = future.result()
                credal_results[idx] = result

        # --- Combine ---
        signals: List[InconsistencySignal] = []
        for idx in range(n):
            credal_ci, type_scores = credal_results[idx]
            i_meta_value = i_meta_values[idx]

            alpha        = self._adaptive_alpha(credal_ci)
            combined     = alpha * credal_ci.midpoint + (1.0 - alpha) * i_meta_value
            disagreement = abs(credal_ci.midpoint - i_meta_value)
            trigger      = (
                combined > self.mutation_threshold
                or disagreement > self.disagreement_threshold
            )

            signals.append(InconsistencySignal(
                credal_inconsistency=credal_ci,
                i_meta=i_meta_value,
                combined_score=float(combined),
                type_scores=type_scores,
                trigger_mutation=trigger,
                diagnostics={
                    "alpha":           alpha,
                    "credal_midpoint": credal_ci.midpoint,
                    "credal_width":    credal_ci.width,
                    "i_meta":          i_meta_value,
                    "combined":        combined,
                    "disagreement":    disagreement,
                },
            ))

        return signals# Cần import thêm các thư viện sau ở đầu file
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet

# ============================================================================
# Section 3: Attention-Steered Evidential Extractor (Thay thế MetaCognitiveExtractor)
# ============================================================================

class AttentionSteeredEvidentialExtractor(nn.Module):
    """
    Nâng cấp SOTA:
    1. Attention-based Probe: Focus vào logical tokens thay vì mean pooling.
    2. Cross-Layer Dynamics: So sánh layer semantic (vd: 24) và reasoning (vd: 32).
    3. Evidential Deep Learning (EDL): Trả về tham số Dirichlet alpha thay vì I_meta vô hướng.
    """
    def __init__(
        self,
        hidden_dim: int = 8192,
        start_layer: int = 24, # Semantic
        end_layer: int = 32,   # Reasoning
        context_dim: int = 1024,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.device = device
        
        # Cross-layer Attention (Học cách nhìn sự mâu thuẫn)
        self.logic_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, batch_first=True
        )
        
        # Evidential Head (Tạo tham số Dirichlet alpha)
        # Thay vì sigmoid ra 1 số, ta xuất ra 2 số (evidence_consistent, evidence_inconsistent)
        self.evidential_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2),
            nn.Softplus() # Evidence e_k >= 0
        )
        
        # Differentiable Thresholding (Học ngưỡng dựa trên độ khó context)
        self.diff_thresh = nn.Sequential(
            nn.Linear(context_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def extract_and_attend(self, hidden_states: Tuple[torch.Tensor], logic_mask: Optional[torch.Tensor] = None):
        """Cross-Layer Probing thay vì Mean Pooling."""
        # hidden_states: (B, Seq, Dim)
        semantic_state = hidden_states[self.start_layer]
        reasoning_state = hidden_states[self.end_layer]
        
        # Nếu có mask của các token "if", "not", "because", dùng nó để guide attention
        attn_mask = ~logic_mask if logic_mask is not None else None
        
        # Reasoning attend vào Semantic để tìm mâu thuẫn logic
        context_aware_state, _ = self.logic_attention(
            query=reasoning_state,
            key=semantic_state,
            value=semantic_state,
            key_padding_mask=attn_mask
        )
        
        # Pool lấy token cuối cùng mang tính chất tóm lược (hoặc mean pooling có trọng số)
        pooled_reasoning = reasoning_state[:, -1, :]
        pooled_context = context_aware_state[:, -1, :]
        
        # Nối lại tạo vector mâu thuẫn cross-layer
        return torch.cat([pooled_reasoning, pooled_context], dim=-1)

    def forward(self, model_outputs, context_embed, logic_mask=None):
        hidden_states = model_outputs.hidden_states
        features = self.extract_and_attend(hidden_states, logic_mask)
        
        # Tính Dirichlet Alphas: alpha_k = e_k + 1
        evidence = self.evidential_head(features)
        alphas = evidence + 1.0
        
        # Tính Epistemic Uncertainty & Aleatoric
        S = torch.sum(alphas, dim=-1, keepdim=True)
        epistemic_uncertainty = 2.0 / S # Bất định bậc 2
        p_inconsistent = alphas[..., 1:2] / S # Xác suất lỗi
        
        dyn_threshold = self.diff_thresh(context_embed)
        
        return p_inconsistent.squeeze(-1), epistemic_uncertainty.squeeze(-1), features, dyn_threshold

# ============================================================================
# Section 4: Contrastive Evidential Trainer (Thay thế LRAxisTrainer)
# ============================================================================

class ContrastiveEvidentialTrainer:
    """
    Sử dụng Contrastive Inconsistency Learning (CIL) với Triplet Loss.
    Ép mô hình hiểu được lỗi logic tinh vi (Subtle Fallacies).
    """
    def __init__(self, meta_extractor: AttentionSteeredEvidentialExtractor, device: str = "cuda"):
        self.extractor = meta_extractor
        self.device = device
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    def compute_edl_loss(self, alphas: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Loss function cho Evidential Deep Learning (Type II Maximum Likelihood)"""
        S = torch.sum(alphas, dim=-1, keepdim=True)
        # NCE Loss
        loss_err = torch.sum(labels * (torch.digamma(S) - torch.digamma(alphas)), dim=-1)
        # KL Divergence term (phạt mô hình nếu tự tin mù quáng vào data ngoài phân phối)
        # ... (Rút gọn KL Divergence implementation)
        return loss_err.mean()

    def train_step(self, anchor_hs, pos_hs, neg_hs, anchor_ctx, labels):
        self.extractor.train()
        
        # 1. Forward 3 nhánh để lấy features
        _, _, feat_anchor, _ = self.extractor(anchor_hs, anchor_ctx)
        _, _, feat_pos, _ = self.extractor(pos_hs, anchor_ctx) # Pos: Lỗi logic tinh vi
        _, _, feat_neg, _ = self.extractor(neg_hs, anchor_ctx) # Neg: Proof hoàn toàn đúng
        
        # 2. Triplet Loss để tách feature space
        loss_contrastive = self.triplet_loss(feat_anchor, feat_pos, feat_neg)
        
        # 3. EDL Loss để học alphas
        alphas = self.extractor.evidential_head(feat_anchor) + 1.0
        loss_edl = self.compute_edl_loss(alphas, labels)
        
        return loss_contrastive + 0.5 * loss_edl

# ============================================================================
# Section 5: SOTA Probabilistic Inconsistency Signal
# ============================================================================

def sinkhorn_wasserstein(M: torch.Tensor, r: torch.Tensor, c: torch.Tensor, reg: float = 0.1, iters: int = 20):
    """Tính Optimal Transport (S2A Alignment Score)"""
    K = torch.exp(-M / reg)
    u = torch.ones_like(r)
    v = torch.ones_like(c)
    for _ in range(iters):
        u = r / (torch.matmul(K, v) + 1e-8)
        v = c / (torch.matmul(K.transpose(-2, -1), u) + 1e-8)
    T = u.unsqueeze(-1) * K * v.unsqueeze(-2)
    return torch.sum(T * M, dim=(-2, -1))

class ProbabilisticInconsistencySignal:
    def __init__(self, alpha_base=0.6, aggregation="markov", device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.alpha_base = alpha_base
        self.device = device
        self.credal_engine = CredalInconsistencyEngine(aggregation=aggregation)
        self.meta_extractor = AttentionSteeredEvidentialExtractor(device=device).to(device)
        self.stability_counter = 0 # Quản lý Full Reset

    def _explain_inconsistency(self, p_inc: float, epistemic: float, s2a_gap: float, credal_mid: float) -> str:
        """Module giải thích Explainable Inconsistency"""
        explanations = []
        if epistemic > 0.5:
            explanations.append("High Epistemic Uncertainty: LLM chưa từng gặp cấu trúc logic này.")
        if s2a_gap > 0.4:
            explanations.append("S2A Alignment Fail: Luồng chú ý của LLM lệch khỏi Symbolic Graph.")
        if credal_mid > 0.7:
            explanations.append("Credal Collision: Bằng chứng truy xuất được hoàn toàn trái ngược với các tiên đề logic.")
        return " | ".join(explanations) if explanations else "Reasoning stream stable."

    def compute(
        self,
        evidence_set: List[RetrievedEvidence],
        proof_chain: List[SymbolicProofStep],
        llm_outputs, 
        context_embed,
        symbolic_adj_matrix, # Cần thêm vào input
        llm_attention_matrix, # Cần thêm vào input
        ensemble_preds: Optional[torch.Tensor] = None # Lấy từ MI300X Multi-GPU Dropout
    ):
        # 1. Credal Signal (Bất định bậc một từ data)
        credal_ci, type_scores = self.credal_engine.compute(evidence_set, proof_chain)
        
        # 2. Metacognitive Signal (Bất định bậc hai từ model)
        p_inc = 0.0
        epistemic = 0.0
        dyn_thresh = 0.5
        
        if llm_outputs is not None:
            with torch.no_grad():
                self.meta_extractor.eval()
                p_inc_tensor, epis_tensor, _, thresh_tensor = self.meta_extractor(llm_outputs, context_embed)
                p_inc = p_inc_tensor.item()
                epistemic = epis_tensor.item()
                dyn_thresh = thresh_tensor.item()

        # 3. S2A (Optimal Transport)
        s2a_gap = 0.0
        if symbolic_adj_matrix is not None and llm_attention_matrix is not None:
            M = torch.cdist(symbolic_adj_matrix, llm_attention_matrix)
            r = torch.ones(M.size(-2), device=self.device) / M.size(-2)
            c = torch.ones(M.size(-1), device=self.device) / M.size(-1)
            s2a_gap = sinkhorn_wasserstein(M, r, c).item()

        # 4. Hardware-Aware MI (MI300X Deep Ensembles)
        hw_entropy = 0.0
        if ensemble_preds is not None:
            mean_pred = torch.mean(ensemble_preds, dim=0)
            entropy_of_mean = - (mean_pred * torch.log(mean_pred + 1e-8) + (1-mean_pred) * torch.log(1-mean_pred + 1e-8))
            mean_of_entropies = torch.mean(- (ensemble_preds * torch.log(ensemble_preds + 1e-8) + (1-ensemble_preds) * torch.log(1-ensemble_preds + 1e-8)))
            hw_entropy = (entropy_of_mean - mean_of_entropies).item() # Mutual Information

        # 5. Adaptive Combination & Trigger Logic
        # Sử dụng Dynamic Threshold đã học được
        alpha = self.alpha_base * (1.0 - credal_ci.width)
        combined = (alpha * credal_ci.midpoint) + ((1 - alpha) * p_inc) + (0.1 * s2a_gap) + (0.1 * hw_entropy)
        
        disagreement = abs(credal_ci.midpoint - p_inc)
        trigger = (combined > dyn_thresh) or (disagreement > 0.4)

        # 6. Stability Metric & Full Reset
        if trigger:
            self.stability_counter += 1
        else:
            self.stability_counter = 0
            
        force_reset = self.stability_counter >= 3

        explanation = self._explain_inconsistency(p_inc, epistemic, s2a_gap, credal_ci.midpoint)

        return InconsistencySignal(
            credal_inconsistency=credal_ci,
            i_meta=p_inc, # i_meta giờ là expectation probability
            combined_score=float(combined),
            type_scores=type_scores,
            trigger_mutation=trigger,
            diagnostics={
                "epistemic_uncertainty": epistemic,
                "s2a_alignment_gap": s2a_gap,
                "hardware_mi": hw_entropy,
                "dynamic_threshold": dyn_thresh,
                "explanation": explanation,
                "trigger_full_reset": force_reset
            }
        )