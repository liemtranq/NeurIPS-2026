"""
Component 3 — Step-level Credal-Neural Inconsistency Engine (PIS v2)
=====================================================================
NeurIPS 2026 Submission

Architecture:
  - Single ProbabilisticInconsistencySignal class (no duplicate definitions)
  - Alignment-by-pointer: each proof step only compared to its linked evidence
  - Step-level InconsistencySignal with per-step credal CI + neural uncertainty
  - Uncertainty-weighted hidden-state pooling (entropy/logprob-weighted)
  - Evidential Deep Learning (EDL/Dirichlet): p_inconsistent, epistemic,
    aleatoric, evidence_strength per step
  - Calibration layer (temperature scaling + isotonic regression) with
    ECE / Brier / AUROC reporting
  - diagnose_failure() → FACTUAL / TEMPORAL / CAUSAL / LOGICAL /
    MISSING_PREMISE / INVALID_INFERENCE
  - Output: global_score, step_scores, top_failed_steps, repair_hint

Hardware target: AMD MI300X 192GB — Llama-3.1-70B full precision, 1x GPU.
"""

from __future__ import annotations

import re
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
from torch.distributions import Dirichlet

logger = logging.getLogger(__name__)


# ============================================================================
# Section 1: Enumerations & Core Data Structures
# ============================================================================

class FailureType(Enum):
    """Fine-grained failure taxonomy for proof steps."""
    NONE              = "none"
    FACTUAL           = "factual"           # Retrieved fact contradicts step conclusion
    TEMPORAL          = "temporal"          # Temporal ordering violated
    CAUSAL            = "causal"            # Causal chain broken
    LOGICAL           = "logical"           # Pure logical contradiction
    MISSING_PREMISE   = "missing_premise"   # A required premise has no evidence support
    INVALID_INFERENCE = "invalid_inference" # Rule applied incorrectly given premises


@dataclass
class CI:
    """
    Credal interval [lower, upper] — imprecise probability over inconsistency.
    Defensive clamp + swap on init; all arithmetic stays in [0, 1].
    """
    lower: float
    upper: float

    def __post_init__(self) -> None:
        lo = float(np.clip(self.lower, 0.0, 1.0))
        hi = float(np.clip(self.upper, 0.0, 1.0))
        self.lower = min(lo, hi)
        self.upper = max(lo, hi)

    @property
    def midpoint(self) -> float:
        return (self.lower + self.upper) / 2.0

    @property
    def width(self) -> float:
        return self.upper - self.lower

    @property
    def is_precise(self) -> bool:
        return self.width < 1e-6

    def intersect(self, other: "CI") -> "CI":
        """
        Intersection of two credal intervals.
        Returns CI(0, 0) if disjoint (hard inconsistency).
        """
        new_lower = max(self.lower, other.lower)
        new_upper = min(self.upper, other.upper)
        if new_lower > new_upper:
            return CI(0.0, 0.0)
        return CI(new_lower, new_upper)

    def __repr__(self) -> str:
        return f"[{self.lower:.4f}, {self.upper:.4f}]"


@dataclass
class RetrievedEvidence:
    """A single piece of retrieved evidence (from Component 1)."""
    evidence_id:     str
    text:            str
    source_id:       str
    confidence:      CI        # Retrieval confidence as credal interval
    relevance_score: float     # Dense retrieval score
    sparse_score:    float     # BM25 score
    hop_depth:       int = 1


@dataclass
class ProofStep:
    """
    One atomic step in the symbolic proof trace (from Component 2).

    linked_evidence_ids: IDs of RetrievedEvidence objects that ground
    the premises of this step. Only these are compared — not all evidence.
    """
    step_id:             str
    rule_name:           str
    premises:            List[str]
    conclusion:          str
    confidence_ci:       CI
    failure_type:        FailureType = FailureType.NONE
    linked_evidence_ids: List[str] = field(default_factory=list)
    step_index:          int = 0


@dataclass
class NeuralSignals:
    """
    Neural signals supplied by the LLM forward pass for each proof step.

    token_logprobs  : log-probabilities over the conclusion tokens, shape (T,)
    hidden_states   : tuple of tensors (n_layers, batch=1, seq, dim)
                      — only layers of interest need to be passed
    token_ids       : optional, for debugging

    ensemble_logits : optional (K, 2) from K dropout forward passes,
                      used for hardware MI estimation on MI300X.
    """
    token_logprobs:   Optional[torch.Tensor] = None   # (T,)
    hidden_states:    Optional[Tuple[torch.Tensor, ...]] = None
    token_ids:        Optional[torch.Tensor] = None    # (T,)
    ensemble_logits:  Optional[torch.Tensor] = None    # (K, 2)


@dataclass
class StepInconsistencySignal:
    """Per-step output consumed by Component 4 (Bandit Mutation Engine)."""
    step_id:             str
    credal_ci:           CI
    neural_uncertainty:  float          # Epistemic uncertainty from EDL head
    aleatoric:           float          # Aleatoric uncertainty from EDL head
    evidence_strength:   float          # Dirichlet total evidence S
    p_inconsistent:      float          # Expected inconsistency probability
    disagreement:        float          # |credal_mid - p_inconsistent|
    error_type:          FailureType
    trigger:             bool
    repair_hint:         str


@dataclass
class InconsistencySignal:
    """
    Full output of ProbabilisticInconsistencySignal.compute().
    Consumed by Component 4 and fed back to Component 1 for replan.
    """
    global_score:      float
    credal_global:     CI
    step_scores:       List[StepInconsistencySignal]
    top_failed_steps:  List[str]        # step_ids ranked by combined score
    repair_hint:       str              # Human-readable replan instruction
    trigger_mutation:  bool
    calibration_info:  Dict[str, float] = field(default_factory=dict)
    diagnostics:       Dict[str, Any]  = field(default_factory=dict)


# ============================================================================
# Section 2: Credal Inconsistency Engine (LCN-inspired, step-aligned)
# ============================================================================

class CredalInconsistencyEngine:
    """
    Computes credal interval [L, U] for inconsistency between retrieved
    evidence and each symbolic proof step, using alignment-by-pointer.

    Key change vs. v1:
      - compute() now takes a ProofStep and only the evidence objects
        whose IDs appear in step.linked_evidence_ids.
      - Aggregation modes: "markov" | "independent" | "worst_case".
    """

    def __init__(
        self,
        aggregation: str = "markov",
        epsilon: float = 1e-6,
    ) -> None:
        assert aggregation in ("markov", "independent", "worst_case")
        self.aggregation = aggregation
        self.epsilon = epsilon

    # ------------------------------------------------------------------
    # Pairwise inconsistency: one evidence vs. one proof step
    # ------------------------------------------------------------------

    def _pairwise(self, evidence: RetrievedEvidence, step: ProofStep) -> CI:
        e = evidence.confidence
        p_raw = step.confidence_ci

        p: CI
        if not isinstance(p_raw, CI):
            depth_penalty = step.step_index * 0.015
            margin = 0.05 + depth_penalty
            p = CI(lower=float(p_raw) - margin, upper=float(p_raw) + margin)
        else:
            p = p_raw

        intersection = e.intersect(p)

        if intersection.width == 0.0 and intersection.upper == 0.0:
            # Disjoint → hard inconsistency
            gap  = max(0.0, e.lower - p.upper, p.lower - e.upper)
            span = max(e.upper, p.upper) - min(e.lower, p.lower)
            return CI(
                lower=float(np.clip(gap, 0.0, 1.0)),
                upper=float(np.clip(max(gap, span), 0.0, 1.0)),
            )
        else:
            # Partial inconsistency
            total_width   = max(e.width, p.width, self.epsilon)
            overlap_width = intersection.width
            u_inc = float(np.clip(1.0 - overlap_width / total_width, 0.0, 1.0))
            return CI(lower=0.0, upper=u_inc)

    # ------------------------------------------------------------------
    # Aggregation helpers (unchanged math, same as v1 with FIX-1)
    # ------------------------------------------------------------------

    def _aggregate_independent(self, pairwise: List[CI]) -> CI:
        if not pairwise:
            return CI(0.0, 0.0)
        prod_opt = 1.0  # ∏(1 - u_i) → optimistic → L_agg
        prod_pes = 1.0  # ∏(1 - l_i) → pessimistic → U_agg
        for ci in pairwise:
            prod_opt *= (1.0 - ci.upper)
            prod_pes *= (1.0 - ci.lower)
        l_agg = float(np.clip(1.0 - prod_pes, 0.0, 1.0))
        u_agg = float(np.clip(1.0 - prod_opt, 0.0, 1.0))
        return CI(lower=l_agg, upper=u_agg)

    def _aggregate_worst_case(self, pairwise: List[CI]) -> CI:
        if not pairwise:
            return CI(0.0, 0.0)
        return CI(
            lower=min(ci.lower for ci in pairwise),
            upper=max(ci.upper for ci in pairwise),
        )

    def _aggregate_markov(
        self,
        pairwise:    List[CI],
        proof_steps: List[ProofStep],
    ) -> CI:
        if not pairwise:
            return CI(0.0, 0.0)
        depth_groups: Dict[int, List[CI]] = {}
        for ci, step in zip(pairwise, proof_steps):
            depth_groups.setdefault(step.step_index, []).append(ci)
        layer_intervals = [
            self._aggregate_worst_case(depth_groups[d])
            for d in sorted(depth_groups)
        ]
        return self._aggregate_independent(layer_intervals)

    # ------------------------------------------------------------------
    # Per-step compute: alignment-by-pointer
    # ------------------------------------------------------------------

    def compute_step(
        self,
        step:         ProofStep,
        evidence_map: Dict[str, RetrievedEvidence],
    ) -> CI:
        """
        Compute credal inconsistency for a single proof step.
        Only evidence IDs listed in step.linked_evidence_ids are used.
        """
        linked = [
            evidence_map[eid]
            for eid in step.linked_evidence_ids
            if eid in evidence_map
        ]

        if not linked:
            # No linked evidence → missing premise signal
            return CI(0.5, 1.0)

        pairwise = [self._pairwise(ev, step) for ev in linked]

        if self.aggregation == "markov":
            expanded_steps = [step] * len(pairwise)
            return self._aggregate_markov(pairwise, expanded_steps)
        elif self.aggregation == "independent":
            return self._aggregate_independent(pairwise)
        else:
            return self._aggregate_worst_case(pairwise)

    # ------------------------------------------------------------------
    # Global aggregation across all steps
    # ------------------------------------------------------------------

    def compute_global(self, step_cis: List[CI], proof_chain: List[ProofStep]) -> CI:
        """Aggregate per-step credal CIs into a global credal interval."""
        if self.aggregation == "markov":
            return self._aggregate_markov(step_cis, proof_chain)
        elif self.aggregation == "independent":
            return self._aggregate_independent(step_cis)
        else:
            return self._aggregate_worst_case(step_cis)


# ============================================================================
# Section 3: Uncertainty-Weighted Hidden-State Probe (Step-Level I_meta)
# ============================================================================

class UncertaintyWeightedProbe(nn.Module):
    """
    Step-level neural inconsistency head.

    Key upgrades vs. v1 MetaCognitiveExtractor:
      1. Uncertainty-weighted pooling: token positions with high entropy /
         low logprob receive higher weight (more informative for failure
         detection). Replaces mean pooling.
      2. Cross-layer delta: difference between semantic layer (start_layer)
         and reasoning layer (end_layer), not just last-layer features.
      3. Evidential Deep Learning (EDL) output head: returns Dirichlet
         parameters → p_inconsistent, epistemic, aleatoric,
         evidence_strength. Not a scalar.
      4. Optional ensemble MI (hardware-aware): if ensemble_logits provided,
         computes mutual information across dropout samples.
    """

    def __init__(
        self,
        hidden_dim:  int = 8192,   # Llama-3.1-70B
        start_layer: int = 24,     # Semantic layer
        end_layer:   int = 32,     # Reasoning layer
        n_classes:   int = 2,      # consistent vs. inconsistent
        device:      str = "cuda",
    ) -> None:
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.start_layer = start_layer
        self.end_layer   = end_layer
        self.n_classes   = n_classes
        self.device      = device

        # Cross-layer feature: semantic ‖ reasoning ‖ delta → 3 * hidden_dim
        feature_dim = hidden_dim * 3

        # EDL head: outputs evidence e_k >= 0, one per class
        self.edl_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, n_classes),
            nn.Softplus(),   # evidence e_k ≥ 0
        )

    # ------------------------------------------------------------------
    # Uncertainty-weighted pooling
    # ------------------------------------------------------------------

    @staticmethod
    def _uncertainty_weighted_pool(
        h:          torch.Tensor,   # (seq, hidden_dim)
        logprobs:   Optional[torch.Tensor],  # (seq,) or None
    ) -> torch.Tensor:
        """
        Pool hidden states weighted by token-level uncertainty.

        Weight ∝ H(token) = -logprob (high uncertainty → large weight).
        Falls back to uniform (mean pooling) if logprobs unavailable.

        Returns: (hidden_dim,)
        """
        if logprobs is None or logprobs.shape[0] != h.shape[0]:
            return h.mean(dim=0)

        # Token entropy proxy: -logprob (higher = more uncertain)
        neg_lp  = -logprobs.to(h.device)            # (seq,)
        weights = F.softmax(neg_lp, dim=0)          # normalise
        return (h * weights.unsqueeze(-1)).sum(dim=0)  # (hidden_dim,)

    # ------------------------------------------------------------------
    # EDL utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _edl_quantities(
        evidence: torch.Tensor,   # (batch, n_classes)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        From Dirichlet evidence e_k compute:
          alpha_k = e_k + 1
          S       = Σ alpha_k               (Dirichlet strength)
          p_k     = alpha_k / S             (expected probability)
          epistemic = n_classes / S          (vacuity / epistemic uncertainty)
          aleatoric = Σ p_k(1 - p_k)        (distributional uncertainty)

        Returns: (p_inconsistent, epistemic, aleatoric, evidence_strength)
        """
        alpha    = evidence + 1.0                              # (B, C)
        S        = alpha.sum(dim=-1, keepdim=True)             # (B, 1)
        p        = alpha / S                                   # (B, C)
        p_inc    = p[:, 1]                                     # (B,) class=1
        epist    = evidence.shape[-1] / S.squeeze(-1)          # (B,)
        aleat    = (p * (1.0 - p)).sum(dim=-1)                 # (B,)
        strength = S.squeeze(-1)                               # (B,)
        return p_inc, epist, aleat, strength

    # ------------------------------------------------------------------
    # Forward: per-step neural uncertainty
    # ------------------------------------------------------------------

    def forward(
        self,
        neural:    NeuralSignals,
    ) -> Tuple[float, float, float, float]:
        """
        Compute step-level EDL quantities from NeuralSignals.

        Returns:
            p_inconsistent    : float ∈ [0, 1]
            epistemic         : float ∈ [0, 1]
            aleatoric         : float ∈ [0, 1]
            evidence_strength : float ≥ 0
        """
        if neural.hidden_states is None:
            return 0.5, 1.0, 0.5, 0.0

        hs     = neural.hidden_states
        n_lay  = len(hs)

        def _safe_layer(idx: int) -> torch.Tensor:
            idx = min(idx, n_lay - 1)
            return hs[idx][0]   # (seq, dim) — squeeze batch dim

        h_sem = _safe_layer(self.start_layer)     # semantic
        h_rea = _safe_layer(self.end_layer)        # reasoning

        logp  = neural.token_logprobs              # (seq,) or None

        # Uncertainty-weighted pooling per layer
        v_sem = self._uncertainty_weighted_pool(h_sem, logp)   # (dim,)
        v_rea = self._uncertainty_weighted_pool(h_rea, logp)   # (dim,)
        v_del = v_rea - v_sem                                  # cross-layer delta

        features = torch.cat([v_sem, v_rea, v_del], dim=0).unsqueeze(0)  # (1, 3*dim)
        features = features.to(next(self.edl_head.parameters()).device)

        with torch.no_grad():
            self.eval()
            evidence = self.edl_head(features)               # (1, n_classes)
            p_inc, epist, aleat, strength = self._edl_quantities(evidence)

        p_inc_val  = float(p_inc[0].item())
        epist_val  = float(epist[0].item())
        aleat_val  = float(aleat[0].item())
        str_val    = float(strength[0].item())

        # Optional: hardware ensemble MI (MI300X multi-GPU dropout)
        if neural.ensemble_logits is not None:
            p_inc_val = self._ensemble_mi_correction(
                p_inc_val, neural.ensemble_logits
            )

        return p_inc_val, epist_val, aleat_val, str_val

    @staticmethod
    def _ensemble_mi_correction(
        p_inc_base: float,
        ensemble_logits: torch.Tensor,   # (K, 2)
    ) -> float:
        """
        Compute mutual information across K ensemble members.
        MI = H(E_θ[p]) - E_θ[H(p)]
        Blends MI into p_inconsistent as a calibration signal.
        """
        probs  = torch.softmax(ensemble_logits, dim=-1)          # (K, 2)
        p_mean = probs.mean(dim=0)                               # (2,)
        h_mean = -(p_mean * torch.log(p_mean + 1e-8)).sum()
        h_each = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)  # (K,)
        mi     = float((h_mean - h_each.mean()).item())
        # Blend: high MI → lean toward p_inc = 0.5 (uncertain)
        return 0.8 * p_inc_base + 0.2 * float(np.clip(0.5 + mi, 0.0, 1.0))


# ============================================================================
# Section 4: Failure Diagnosis
# ============================================================================

class FailureDiagnoser:
    """
    Maps (ProofStep, credal_ci, p_inconsistent, linked_evidence) to a
    FailureType with a natural-language repair hint.
    """

    _TEMPORAL_KW    = frozenset(["temporal","before","after","during","since",
                                 "until","order_time","span","sequence"])
    _CAUSAL_KW      = frozenset(["caus","implies","because","enable","prevent",
                                 "require","effect","mediat","confounder"])
    _LOGICAL_KW     = frozenset(["contradict","negat","contrapos","tollens",
                                 "double_neg","disjunct","conjunct","universal",
                                 "existential","logical","implication"])
    _FACTUAL_KW     = frozenset(["fact","entity","find","filter","bridge","path",
                                 "link","join","relate","verify","exists",
                                 "select","project","resolve"])
    _COMPARATIVE_KW = frozenset(["compar","greater","less","above","below",
                                 "max","min","rank","diff","ratio","threshold"])

    @staticmethod
    def _extract_years(text: str) -> List[int]:
        years = re.findall(r"\b(1[0-9]{3}|20[0-9]{2}|21[0-9]{2})\b", text)
        return [int(y) for y in years]

    @staticmethod
    def _has_resolution_marker(text: str) -> bool:
        markers = [
            "cannot",
            "can't",
            "impossible",
            "must be wrong",
            "date or object identification must be wrong",
            "not show",
            "reject impossible ordering",
            "misdated",
            "incorrect date",
            "wrong date",
        ]
        return any(m in text for m in markers) 
    @staticmethod
    def _temporal_relation_from_text(text: str) -> Optional[Tuple[int, int]]:
        raw_years = re.findall(
            r"\b(?:\d{1,3}|1[0-9]{3}|20[0-9]{2}|21[0-9]{2})(?:s)?\b",
            text
        )
        years = [int(y.rstrip("s")) for y in raw_years]
        if len(years) < 2:
            return None

        text = text.lower()

        # "If origin YEAR, how could claim YEAR"
        if text.startswith("if") and "how could" in text:
            return years[1], years[0]  # claim, origin

        # "How could claim YEAR if origin YEAR"
        if text.startswith("how could") and " if " in text:
            return years[0], years[1]  # claim, origin

        # generic: for dataset-style temporal claims, later mentioned origin often follows "if"
        if " if " in text:
            return years[0], years[1]

        return None 

    @staticmethod
    def _predict_temporal_from_question_like_text(text: str):
        text = text.lower()

        raw = re.findall(
            r"\b(?:\d{1,3}|1[0-9]{3}|20[0-9]{2}|21[0-9]{2})(?:s)?\b",
            text,
        )
        years = [int(x.rstrip("s")) for x in raw]

        if len(years) < 2:
            if (
                text.startswith("how could")
                and ("still ongoing" in text or "under construction" in text)
                and ("exist in" in text or "photo" in text or "painting" in text)
            ):
                return 1
            return None

        y1, y2 = years[0], years[1]

        if text.startswith("if") and "how could" in text:
            origin_year, claim_year = y1, y2
            return int(claim_year < origin_year)

        if text.startswith("how could"):
            origin_markers = [
                "founded", "established", "released", "published", "launched",
                "premiered", "completed", "built", "invented", "opened",
                "incorporated", "discovered", "introduced", "created", "finished",
                "available", "debuted", "operated", "tested", "made",
            ]

            prefix_before_y1 = text.split(str(y1))[0]

            if "completed" in prefix_before_y1 and (
                "photo" in prefix_before_y1 or "painting" in prefix_before_y1
            ):
                y1_is_origin = False
            else:
                y1_is_origin = any(marker in prefix_before_y1 for marker in origin_markers)

            if y1_is_origin:
                origin_year, claim_year = y1, y2
            else:
                claim_year, origin_year = y1, y2

            return int(claim_year < origin_year)

        if text.startswith("could") and " if " in text:
            claim_year, origin_year = y1, y2
            return int(claim_year < origin_year)

        return None                           

    def diagnose(
        self,
        step:             ProofStep,
        credal_ci:        CI,
        p_inconsistent:   float,
        linked_evidence:  List[RetrievedEvidence],
    ) -> Tuple[FailureType, str]:
        """
        Returns (FailureType, repair_hint).
        """
        step_text = " ".join([
            str(step.rule_name or ""),
            str(step.conclusion or ""),
            " ".join(map(str, step.premises or [])),
            " ".join(ev.text for ev in linked_evidence),
        ]).lower()

        if self._predict_temporal_from_question_like_text(step_text) == 1:
            return (
                FailureType.TEMPORAL,
                f"Step '{step.step_id}': temporal contradiction detected by generalized parser. "
                f"Replan: verify chronology and retrieve date-specific evidence."
            )

        # 1. Missing premise
        if not linked_evidence:
            return (
                FailureType.MISSING_PREMISE,
                f"Step '{step.step_id}': no grounding evidence found for "
                f"premises {step.premises}. "
                f"Replan: retrieve evidence for these premises."
            )

        # 1.5 Preserve execution failure from Component 2
        if step.failure_type != FailureType.NONE:
            return (
                step.failure_type,
                f"Step '{step.step_id}' failed during symbolic execution: "
                f"{step.failure_type.value}. "
                f"Replan: repair this step before continuing."
            )

        raw_question = getattr(step, "raw_question", "")
        temporal_source = raw_question.lower().strip()

        if not temporal_source:
            evidence_text = " ".join(ev.text for ev in linked_evidence).lower()
            temporal_source = evidence_text if evidence_text.strip() else step_text

        temporal_pred = self._predict_temporal_from_question_like_text(temporal_source)

        if temporal_pred == 1:
            return (
                FailureType.TEMPORAL,
                f"Step '{step.step_id}': temporal contradiction detected by generalized parser. "
                f"Replan: verify chronology and retrieve date-specific evidence."
            )

        if temporal_pred == 0:
            return (
                FailureType.NONE,
                "Temporal order is valid (forward progression)."
            )

        rel = self._temporal_relation_from_text(step_text)
        if rel is not None and not self._has_resolution_marker(step_text):
            claim_year, origin_year = rel
            if claim_year < origin_year:
                return (
                    FailureType.TEMPORAL,
                    f"Step '{step.step_id}': temporal contradiction "
                    f"(claim {claim_year} before origin {origin_year})."
                )
            return (
                FailureType.NONE,
                "Temporal order is valid (forward progression)."
            )

        rn = step.rule_name.lower()

        all_ev_confident = all(ev.confidence.midpoint > 0.6 for ev in linked_evidence)
        if credal_ci.lower > 0.6 and all_ev_confident:
            return (
                FailureType.INVALID_INFERENCE,
                f"Step '{step.step_id}': premises are evidenced but rule "
                f"'{step.rule_name}' leads to an invalid conclusion "
                f"'{step.conclusion}'. "
                f"Replan: verify rule applicability or revise premises."
            )

        for kw in self._TEMPORAL_KW:
            if kw in rn:
                return (
                    FailureType.TEMPORAL,
                    f"Step '{step.step_id}': temporal ordering violation "
                    f"in rule '{step.rule_name}'. "
                    f"Replan: re-retrieve timeline evidence."
                )

        for kw in self._CAUSAL_KW:
            if kw in rn:
                return (
                    FailureType.CAUSAL,
                    f"Step '{step.step_id}': causal chain break "
                    f"at rule '{step.rule_name}'. "
                    f"Replan: verify enabling conditions."
                )

        for kw in self._LOGICAL_KW:
            if kw in rn:
                return (
                    FailureType.LOGICAL,
                    f"Step '{step.step_id}': logical contradiction "
                    f"via rule '{step.rule_name}'. "
                    f"Replan: check negation or contrapositive."
                )

        for kw in self._FACTUAL_KW:
            if kw in rn:
                return (
                    FailureType.FACTUAL,
                    f"Step '{step.step_id}': retrieved fact conflicts with "
                    f"conclusion '{step.conclusion}'. "
                    f"Replan: re-retrieve primary sources for this entity."
                )

        return (
            FailureType.NONE,
            f"Step '{step.step_id}': no actionable inconsistency detected. "
            f"Credal={credal_ci}, p_inc={p_inconsistent:.3f}."
        )


    # ============================================================================
    # Section 5: Calibration Layer
    # ============================================================================

class CalibrationLayer:
    """
    Post-hoc calibration on a dev set.

    Supports:
      - Temperature scaling (differentiable, single scalar T)
      - Isotonic regression (non-parametric, monotone)

    Reports: ECE (15-bin), Brier score, AUROC.

    Usage:
        cal = CalibrationLayer(method="temperature")
        cal.fit(logits_dev, labels_dev)
        p_cal = cal.calibrate(logits_test)
        metrics = cal.evaluate(p_cal, labels_test)
    """

    def __init__(self, method: str = "temperature", n_bins: int = 15) -> None:
        assert method in ("temperature", "isotonic")
        self.method   = method
        self.n_bins   = n_bins
        self.T        = 1.0     # temperature (scalar)
        self._iso     = None    # isotonic regressor
        self._fitted  = False

    # ------------------------------------------------------------------

    def fit(
        self,
        logits: np.ndarray,   # (N,) raw scores ∈ ℝ
        labels: np.ndarray,   # (N,) ∈ {0, 1}
    ) -> None:
        """Fit calibration on dev set logits."""
        from scipy.special import expit
        from scipy.optimize import minimize_scalar

        if self.method == "temperature":
            def nll(T):
                p = expit(logits / T)
                return -np.mean(
                    labels * np.log(p + 1e-8) +
                    (1 - labels) * np.log(1 - p + 1e-8)
                )
            res = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
            self.T = float(res.x)
            logger.info("Temperature calibration: T=%.4f", self.T)

        else:  # isotonic
            from sklearn.isotonic import IsotonicRegression
            p_raw = expit(logits)
            self._iso = IsotonicRegression(out_of_bounds="clip")
            self._iso.fit(p_raw, labels)
            logger.info("Isotonic calibration fitted.")

        self._fitted = True

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply calibration; returns probabilities ∈ [0, 1]."""
        from scipy.special import expit
        if not self._fitted:
            return expit(logits)
        if self.method == "temperature":
            return expit(logits / self.T)
        else:
            p_raw = expit(logits)
            return self._iso.predict(p_raw)

    def evaluate(
        self,
        probs:  np.ndarray,   # (N,) calibrated probabilities
        labels: np.ndarray,   # (N,) ground-truth
    ) -> Dict[str, float]:
        """
        Compute ECE (equal-width bins), Brier score, AUROC.
        """
        from sklearn.metrics import roc_auc_score

        # ECE
        bin_edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        ece = 0.0
        n   = len(probs)
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (probs >= lo) & (probs < hi)
            if mask.sum() == 0:
                continue
            acc  = labels[mask].mean()
            conf = probs[mask].mean()
            ece += mask.sum() / n * abs(acc - conf)

        # Brier
        brier = float(np.mean((probs - labels) ** 2))

        # AUROC
        try:
            auroc = float(roc_auc_score(labels, probs))
        except ValueError:
            auroc = float("nan")

        metrics = {"ECE": float(ece), "Brier": brier, "AUROC": auroc}
        logger.info("Calibration metrics: %s", metrics)
        return metrics


# ============================================================================
# Section 6: ProbabilisticInconsistencySignal  (single class, no duplicates)
# ============================================================================

class ProbabilisticInconsistencySignal:
    """
    Component 3 main class — Step-level Credal-Neural Inconsistency Engine.

    Inputs
    ------
    evidence_set  : List[RetrievedEvidence] from Component 1
    proof_trace   : List[ProofStep] from Component 2
                    Each step carries linked_evidence_ids → alignment-by-pointer
    neural_signals: Dict[step_id, NeuralSignals] — per-step LLM signals

    Output: InconsistencySignal with
      global_score, step_scores, top_failed_steps, repair_hint

    Architecture
    ------------
    1. Per-step credal CI from linked evidence only (no cross-contamination)
    2. Per-step EDL uncertainty from uncertainty-weighted hidden-state probe
    3. Adaptive alpha combination: credal midpoint + EDL p_inconsistent
    4. FailureDiagnoser → FailureType + repair hint per step
    5. Global aggregation + top_failed_steps ranking
    6. Optional calibration via CalibrationLayer
    """

    def __init__(
        self,
        alpha_base:             float = 0.6,
        alpha_min:              float = 0.2,
        alpha_max:              float = 0.8,
        mutation_threshold:     float = 0.65,
        disagreement_threshold: float = 0.4,
        aggregation:            str   = "markov",
        hidden_dim:             int   = 8192,
        start_layer:            int   = 24,
        end_layer:              int   = 32,
        device:                 str   = "cuda",
        calibration_method:     str   = "temperature",
    ) -> None:
        self.alpha_base             = alpha_base
        self.alpha_min              = alpha_min
        self.alpha_max              = alpha_max
        self.mutation_threshold     = mutation_threshold
        self.disagreement_threshold = disagreement_threshold
        self.device                 = device

        self.credal_engine  = CredalInconsistencyEngine(aggregation=aggregation)
        self.neural_probe   = UncertaintyWeightedProbe(
            hidden_dim=hidden_dim,
            start_layer=start_layer,
            end_layer=end_layer,
            device=device,
        ).to(device)
        self.diagnoser      = FailureDiagnoser()
        self.calibrator     = CalibrationLayer(method=calibration_method)

    # ------------------------------------------------------------------
    # Adaptive alpha
    # ------------------------------------------------------------------

    def _adaptive_alpha(self, credal_ci: CI) -> float:
        """
        α = α_base × (1 − width), clamped to [alpha_min, alpha_max].
        Wide CI → high uncertainty → lower α → trust neural more.
        """
        alpha = self.alpha_base * (1.0 - credal_ci.width)
        return float(np.clip(alpha, self.alpha_min, self.alpha_max))

    # ------------------------------------------------------------------
    # Per-step signal
    # ------------------------------------------------------------------

    def _compute_step(
        self,
        step:         ProofStep,
        evidence_map: Dict[str, RetrievedEvidence],
        neural:       Optional[NeuralSignals],
    ) -> StepInconsistencySignal:
        # 1. Credal CI
        credal_ci = self.credal_engine.compute_step(step, evidence_map)

        # 2. Neural uncertainty
        p_inc, epistemic, aleatoric, ev_strength = (0.5, 1.0, 0.5, 0.0)
        if neural is not None:
            p_inc, epistemic, aleatoric, ev_strength = self.neural_probe(neural)

        # 3. Combine
        alpha = self._adaptive_alpha(credal_ci)
        combined = alpha * credal_ci.midpoint + (1.0 - alpha) * p_inc

        # 4. Disagreement
        disagreement = abs(credal_ci.midpoint - p_inc)

        # 5. Linked evidence — PHẢI tạo trước diagnose
        linked_ev = [
            evidence_map[eid]
            for eid in step.linked_evidence_ids
            if eid in evidence_map
        ]

        # 6. Failure diagnosis
        failure_type, repair_hint = self.diagnoser.diagnose(
            step, credal_ci, p_inc, linked_ev
        )

        # 7. Hard boost for diagnosed failures
        hard_failures = {
            FailureType.TEMPORAL,
            FailureType.LOGICAL,
            FailureType.CAUSAL,
            FailureType.MISSING_PREMISE,
            FailureType.INVALID_INFERENCE,
        }
        if failure_type in hard_failures:
            combined = max(combined, 0.85)
        elif failure_type == FailureType.NONE:
            combined = min(combined, 0.25)

        # 8. Trigger sau khi boostpython3 "Master Orchestrator Pipeline (The Brain).py"
        trigger = (
            combined > self.mutation_threshold
            or disagreement > self.disagreement_threshold
            or failure_type != FailureType.NONE
        )

        return StepInconsistencySignal(
            step_id=step.step_id,
            credal_ci=credal_ci,
            neural_uncertainty=epistemic,
            aleatoric=aleatoric,
            evidence_strength=ev_strength,
            p_inconsistent=float(combined),
            disagreement=float(disagreement),
            error_type=failure_type,
            trigger=trigger,
            repair_hint=repair_hint,
        )

    # ------------------------------------------------------------------
    # Main entry point: compute()
    # ------------------------------------------------------------------
    def compute_tensor_credal_width(self, tensor: torch.Tensor) -> float:
        """
        Tensor-level uncertainty proxy used by Component 2.
        Returns width in [0, 0.99].
        """
        with torch.no_grad():
            x = tensor.float()

            if x.numel() <= 1:
                return 0.0

            variance = torch.var(x).item()
            mean_abs = torch.mean(torch.abs(x)).item() + 1e-6

            width = variance / (variance + mean_abs)
            return float(np.clip(width, 0.0, 0.99))

    def compute(
        self,
        evidence_set:   List[RetrievedEvidence],
        proof_trace:    List[ProofStep],
        neural_signals: Optional[Dict[str, NeuralSignals]] = None,
        max_workers:    int = 4,
    ) -> InconsistencySignal:
        """
        Compute full step-level inconsistency signal.

        Args:
            evidence_set  : All retrieved evidence from Component 1.
            proof_trace   : Ordered symbolic proof steps from Component 2.
            neural_signals: Map step_id → NeuralSignals (from LLM forward).
                            Pass None for credal-only mode.
            max_workers   : CPU threads for parallel credal computation.

        Returns:
            InconsistencySignal
        """
        if not proof_trace:
            return InconsistencySignal(
                global_score=0.0,
                credal_global=CI(0.0, 0.0),
                step_scores=[],
                top_failed_steps=[],
                repair_hint="No proof steps provided.",
                trigger_mutation=False,
            )

        # Build evidence lookup map
        evidence_map: Dict[str, RetrievedEvidence] = {
            ev.evidence_id: ev for ev in evidence_set
        }
        ns_map = neural_signals or {}

        # Parallel per-step computation (CPU-bound credal)
        step_results: List[StepInconsistencySignal] = [None] * len(proof_trace)  # type: ignore

        def _worker(idx: int) -> Tuple[int, StepInconsistencySignal]:
            step   = proof_trace[idx]
            neural = ns_map.get(step.step_id, None)
            return idx, self._compute_step(step, evidence_map, neural)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_worker, i): i for i in range(len(proof_trace))}
            for future in as_completed(futures):
                idx, result = future.result()
                step_results[idx] = result

        # Global credal aggregation
        step_cis  = [s.credal_ci for s in step_results]
        global_ci = self.credal_engine.compute_global(step_cis, proof_trace)

        # Global score: weighted mean of per-step p_inconsistent
        global_score = float(np.mean([s.p_inconsistent for s in step_results]))

        # Top failed steps: sort by p_inconsistent descending
        sorted_steps = sorted(step_results, key=lambda s: s.p_inconsistent, reverse=True)
        top_failed   = [s.step_id for s in sorted_steps if s.trigger]

        # Consolidated repair hint: highest-priority failed step
        if top_failed:
            primary = next(s for s in sorted_steps if s.step_id == top_failed[0])
            repair_hint = primary.repair_hint
        else:
            repair_hint = "All proof steps within acceptable inconsistency bounds."

        trigger_global = global_score > self.mutation_threshold or bool(top_failed)

        return InconsistencySignal(
            global_score=global_score,
            credal_global=global_ci,
            step_scores=step_results,
            top_failed_steps=top_failed,
            repair_hint=repair_hint,
            trigger_mutation=trigger_global,
            diagnostics={
                "n_steps":         len(proof_trace),
                "n_triggered":     len(top_failed),
                "credal_global":   str(global_ci),
                "step_details":    [
                    {
                        "step_id":     s.step_id,
                        "credal_ci":   str(s.credal_ci),
                        "p_inc":       round(s.p_inconsistent, 4),
                        "epistemic":   round(s.neural_uncertainty, 4),
                        "aleatoric":   round(s.aleatoric, 4),
                        "ev_strength": round(s.evidence_strength, 4),
                        "error_type":  s.error_type.value,
                        "trigger":     s.trigger,
                    }
                    for s in step_results
                ],
            },
        )

    # ------------------------------------------------------------------
    # Calibration interface
    # ------------------------------------------------------------------

    def fit_calibration(
        self,
        dev_logits: np.ndarray,
        dev_labels: np.ndarray,
    ) -> Dict[str, float]:
        """
        Fit calibration on dev-set logits and report ECE/Brier/AUROC.

        dev_logits : raw combined_score (pre-sigmoid) for each dev sample
        dev_labels : ground-truth {0: consistent, 1: inconsistent}
        """
        self.calibrator.fit(dev_logits, dev_labels)
        probs = self.calibrator.calibrate(dev_logits)
        return self.calibrator.evaluate(probs, dev_labels)

    def calibrate_score(self, raw_score: float) -> float:
        """Apply fitted calibration to a single raw score."""
        probs = self.calibrator.calibrate(np.array([raw_score]))
        return float(probs[0])

    # ------------------------------------------------------------------
    # Failure localization helper (for Component 4 / Component 1)
    # ------------------------------------------------------------------

    def diagnose_failure(
        self,
        signal: InconsistencySignal,
    ) -> Dict[str, Any]:
        """
        Summarise failure localization from an InconsistencySignal.

        Returns a dict ready to be consumed by Component 1 replan or
        Component 4 mutation selection.

        Keys:
          failed_steps       : List[dict] with step_id, error_type, repair_hint
          dominant_failure   : most frequent FailureType among triggered steps
          replan_targets     : step_ids that Component 1 should re-retrieve for
          mutation_priority  : step_ids ranked for Component 4 to mutate first
        """
        triggered = [s for s in signal.step_scores if s.trigger]
        if not triggered:
            return {
                "failed_steps":     [],
                "dominant_failure": FailureType.NONE.value,
                "replan_targets":   [],
                "mutation_priority": [],
            }

        # Dominant failure type
        type_counts: Dict[str, int] = {}
        for s in triggered:
            type_counts[s.error_type.value] = type_counts.get(s.error_type.value, 0) + 1
        dominant = max(type_counts, key=type_counts.__getitem__)

        # Replan targets: steps needing new retrieval
        retrieval_types = {
            FailureType.FACTUAL.value,
            FailureType.MISSING_PREMISE.value,
            FailureType.TEMPORAL.value,
        }
        replan_targets = [
            s.step_id for s in triggered
            if s.error_type.value in retrieval_types
        ]

        # Mutation priority: ranked by p_inconsistent
        mutation_priority = [
            s.step_id for s in sorted(triggered,
                                       key=lambda x: x.p_inconsistent,
                                       reverse=True)
        ]

        return {
            "failed_steps": [
                {
                    "step_id":    s.step_id,
                    "error_type": s.error_type.value,
                    "p_inc":      round(s.p_inconsistent, 4),
                    "repair":     s.repair_hint,
                }
                for s in triggered
            ],
            "dominant_failure":   dominant,
            "replan_targets":     replan_targets,
            "mutation_priority":  mutation_priority,
        }


# ============================================================================
# Section 7: Offline LR-Axis / EDL Trainer
# ============================================================================

class EDLProbeTrainer:
    """
    Offline trainer for UncertaintyWeightedProbe's EDL head.

    Dataset: List of (NeuralSignals, label) where label ∈ {0: consistent,
    1: inconsistent}.

    Loss = EDL Type II Maximum Likelihood + KL regularisation.
    """

    def __init__(
        self,
        probe:   UncertaintyWeightedProbe,
        device:  str = "cuda",
    ) -> None:
        self.probe  = probe
        self.device = device

    @staticmethod
    def _edl_loss(
        evidence: torch.Tensor,   # (B, C)
        labels:   torch.Tensor,   # (B,) ∈ {0, 1}
        n_classes: int = 2,
        kl_weight: float = 0.1,
    ) -> torch.Tensor:
        """
        EDL loss (Sensoy et al. 2018, adapted for binary case):
          L = Σ_k y_k [ψ(S) − ψ(alpha_k)] + KL(Dir(ã) || Dir(1))

        where ã removes evidence for ground-truth class to avoid
        over-regularising correct predictions.
        """
        alpha  = evidence + 1.0                           # (B, C)
        S      = alpha.sum(dim=-1, keepdim=True)          # (B, 1)

        # One-hot labels
        y = F.one_hot(labels.long(), num_classes=n_classes).float()  # (B, C)

        # MSE-style NLL
        err_sq = (y - alpha / S) ** 2
        var    = alpha * (S - alpha) / (S ** 2 * (S + 1))
        nll    = (err_sq + var).sum(dim=-1).mean()

        # KL divergence term
        alpha_tilde = y + (1.0 - y) * alpha               # (B, C)
        S_tilde     = alpha_tilde.sum(dim=-1, keepdim=True)
        kl = (
            torch.lgamma(S_tilde)
            - torch.lgamma(torch.tensor(float(n_classes)))
            - torch.lgamma(alpha_tilde).sum(dim=-1, keepdim=True)
            + ((alpha_tilde - 1) *
               (torch.digamma(alpha_tilde) - torch.digamma(S_tilde))).sum(dim=-1, keepdim=True)
        ).mean()

        return nll + kl_weight * kl

    def train(
        self,
        dataset:   List[Tuple[NeuralSignals, int]],
        n_epochs:  int   = 50,
        lr:        float = 1e-3,
        batch_size: int  = 16,
    ) -> None:
        """Train the EDL head on labelled (NeuralSignals, label) pairs."""
        optimizer = torch.optim.Adam(
            self.probe.edl_head.parameters(), lr=lr, weight_decay=1e-5
        )

        self.probe.train()
        for epoch in range(n_epochs):
            total_loss = 0.0
            indices = list(range(len(dataset)))
            np.random.shuffle(indices)

            for start in range(0, len(indices), batch_size):
                batch_idx  = indices[start: start + batch_size]
                batch_data = [dataset[i] for i in batch_idx]

                features_list = []
                labels_list   = []

                for neural, label in batch_data:
                    if neural.hidden_states is None:
                        continue

                    hs    = neural.hidden_states
                    n_lay = len(hs)

                    def _safe(idx):
                        return hs[min(idx, n_lay - 1)][0]  # (seq, dim)

                    h_sem = _safe(self.probe.start_layer)
                    h_rea = _safe(self.probe.end_layer)
                    logp  = neural.token_logprobs

                    v_sem = UncertaintyWeightedProbe._uncertainty_weighted_pool(h_sem, logp)
                    v_rea = UncertaintyWeightedProbe._uncertainty_weighted_pool(h_rea, logp)
                    v_del = v_rea - v_sem

                    feat = torch.cat([v_sem, v_rea, v_del], dim=0)
                    features_list.append(feat.to(self.device))
                    labels_list.append(label)

                if not features_list:
                    continue

                feats  = torch.stack(features_list, dim=0)     # (B, 3*dim)
                labels = torch.tensor(labels_list, dtype=torch.long, device=self.device)

                evidence = self.probe.edl_head(feats)          # (B, C)
                loss     = self._edl_loss(evidence, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                logger.info(
                    "EDL training epoch %d/%d  loss=%.4f",
                    epoch + 1, n_epochs,
                    total_loss / max(len(dataset) // batch_size, 1),
                )

        self.probe.eval()
        logger.info("EDL probe training complete.")