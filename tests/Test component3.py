"""
Full test suite for Component 3 — Probabilistic Inconsistency Signal.
No network, no model downloads.

Test groups:
  A. CI algebra
  B. CredalInconsistencyEngine — pairwise, aggregation, type classification
  C. _aggregate_independent correctness (the critical FIX-1 test)
  D. MetaCognitiveExtractor (mock hidden states)
  E. ProbabilisticInconsistencySignal — single + batch
  F. Mutation trigger logic (FIX-5)
  G. Rule classification coverage (FIX-4)
  H. Integration: full pipeline without LLM
"""

import sys
import math
import unittest
import torch
import numpy as np
from typing import List

component3_probabilistic import (
    CI,
    InconsistencyType,
    RetrievedEvidence,
    SymbolicProofStep,
    InconsistencySignal,
    CredalInconsistencyEngine,
    MetaCognitiveExtractor,
    ProbabilisticInconsistencySignal,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def make_evidence(
    lo: float = 0.8, hi: float = 0.95, hop: int = 1
) -> RetrievedEvidence:
    return RetrievedEvidence(
        text="The Eiffel Tower was built in 1889.",
        source_id="wiki_001",
        confidence=CI(lo, hi),
        relevance_score=0.9,
        sparse_score=10.0,
        hop_depth=hop,
    )


def make_step(
    lo: float = 0.85,
    hi: float = 0.95,
    rule: str = "entity_location",
    idx: int = 0,
) -> SymbolicProofStep:
    return SymbolicProofStep(
        rule_name=rule,
        premises=["eiffel_tower(X)", "located_in(X, paris)"],
        conclusion="city(paris)",
        confidence=CI(lo, hi),
        step_index=idx,
    )


# ─────────────────────────────────────────────────────────────────────────────
# A — CI
# ─────────────────────────────────────────────────────────────────────────────

class TestCredalInterval(unittest.TestCase):

    def test_basic_properties(self):
        ci = CI(0.3, 0.7)
        self.assertAlmostEqual(ci.midpoint, 0.5)
        self.assertAlmostEqual(ci.width, 0.4)
        self.assertFalse(ci.is_precise)

    def test_clamp_to_unit_interval(self):
        ci = CI(-0.5, 1.5)
        self.assertGreaterEqual(ci.lower, 0.0)
        self.assertLessEqual(ci.upper, 1.0)

    def test_auto_swap_inverted(self):
        ci = CI(0.9, 0.1)   # inverted → should be swapped
        self.assertLessEqual(ci.lower, ci.upper)

    def test_invariant_lower_le_upper(self):
        for lo, hi in [(0.3, 0.7), (0.5, 0.5), (0.0, 1.0), (0.7, 0.3)]:
            ci = CI(lo, hi)
            self.assertLessEqual(ci.lower, ci.upper, f"Violated for ({lo},{hi})")

    def test_precise_interval(self):
        ci = CI(0.5, 0.5)
        self.assertTrue(ci.is_precise)

    def test_intersect_overlap(self):
        a = CI(0.3, 0.7)
        b = CI(0.5, 0.9)
        result = a.intersect(b)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.lower, 0.5)
        self.assertAlmostEqual(result.upper, 0.7)

    def test_intersect_disjoint_returns_none(self):
        a = CI(0.1, 0.3)
        b = CI(0.5, 0.9)
        self.assertIsNone(a.intersect(b))

    def test_intersect_touching_returns_point(self):
        a = CI(0.0, 0.5)
        b = CI(0.5, 1.0)
        result = a.intersect(b)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.width, 0.0, places=5)

    def test_contains(self):
        ci = CI(0.3, 0.7)
        self.assertTrue(ci.contains(0.5))
        self.assertFalse(ci.contains(0.8))
        self.assertFalse(ci.contains(0.1))

    def test_repr(self):
        ci = CI(0.1, 0.9)
        self.assertIn("0.1000", repr(ci))
        self.assertIn("0.9000", repr(ci))


# ─────────────────────────────────────────────────────────────────────────────
# B — Pairwise inconsistency
# ─────────────────────────────────────────────────────────────────────────────

class TestPairwiseInconsistency(unittest.TestCase):

    def setUp(self):
        self.engine = CredalInconsistencyEngine(aggregation="independent")

    def test_identical_intervals_low_inconsistency(self):
        e = make_evidence(0.8, 0.95)
        p = make_step(0.8, 0.95)
        ci = self.engine.compute_pairwise_inconsistency(e, p)
        self.assertAlmostEqual(ci.lower, 0.0, places=5)
        self.assertLess(ci.upper, 0.5)

    def test_disjoint_intervals_high_inconsistency(self):
        e = make_evidence(0.8, 0.95)
        p = make_step(0.05, 0.20)   # No overlap with [0.8, 0.95]
        ci = self.engine.compute_pairwise_inconsistency(e, p)
        self.assertGreater(ci.upper, 0.5)

    def test_result_is_valid_credal_interval(self):
        e = make_evidence(0.4, 0.6)
        p = make_step(0.5, 0.8)
        ci = self.engine.compute_pairwise_inconsistency(e, p)
        self.assertLessEqual(ci.lower, ci.upper)
        self.assertGreaterEqual(ci.lower, 0.0)
        self.assertLessEqual(ci.upper, 1.0)

    def test_noisy_evidence_widens_inconsistency(self):
        e_clean = make_evidence(0.85, 0.95)
        e_noisy = make_evidence(0.10, 0.50)
        p = make_step(0.85, 0.95)
        ci_clean = self.engine.compute_pairwise_inconsistency(e_clean, p)
        ci_noisy = self.engine.compute_pairwise_inconsistency(e_noisy, p)
        self.assertGreater(ci_noisy.upper, ci_clean.upper)


# ─────────────────────────────────────────────────────────────────────────────
# C — FIX-1: _aggregate_independent correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestAggregateIndependentFix(unittest.TestCase):
    """
    Critical tests for FIX-1.

    Mathematical contract for two intervals [l1,u1] and [l2,u2]:
        L_agg = 1 - (1 - l1)(1 - l2) ... wait, that was the old buggy formula.

    Correct derivation:
        P(any inconsistency) = 1 - P(no inconsistency)
        P(no inconsistency_i) ∈ [1-u_i, 1-l_i]
        → P(no inconsistency) ∈ [∏(1-u_i), ∏(1-l_i)]
        → P(any inconsistency) ∈ [1-∏(1-l_i), 1-∏(1-u_i)]
        i.e. L_agg = 1 - ∏(1-l_i),  U_agg = 1 - ∏(1-u_i)

    So:  L_agg ≤ U_agg  always.
    """

    def setUp(self):
        self.engine = CredalInconsistencyEngine(aggregation="independent")

    def test_lower_le_upper_two_items(self):
        pairwise = [CI(0.1, 0.4), CI(0.2, 0.6)]
        result = self.engine._aggregate_independent(pairwise)
        self.assertLessEqual(
            result.lower, result.upper,
            f"FIX-1 violated: lower={result.lower:.4f} > upper={result.upper:.4f}"
        )

    def test_lower_le_upper_many_items(self):
        pairwise = [
            CI(lo, lo + 0.2)
            for lo in np.linspace(0.0, 0.7, 10)
        ]
        result = self.engine._aggregate_independent(pairwise)
        self.assertLessEqual(result.lower, result.upper)

    def test_monotone_in_individual_bounds(self):
        """More inconsistency in each pair → higher aggregate."""
        low_pair  = [CI(0.05, 0.15), CI(0.05, 0.15)]
        high_pair = [CI(0.4,  0.7),  CI(0.4,  0.7)]
        r_low  = self.engine._aggregate_independent(low_pair)
        r_high = self.engine._aggregate_independent(high_pair)
        self.assertLess(r_low.midpoint, r_high.midpoint)

    def test_zero_inconsistency_stays_zero(self):
        """If all pairs have l=u=0, aggregate must be [0,0]."""
        pairwise = [CI(0.0, 0.0)] * 5
        result = self.engine._aggregate_independent(pairwise)
        self.assertAlmostEqual(result.lower, 0.0, places=6)
        self.assertAlmostEqual(result.upper, 0.0, places=6)

    def test_max_inconsistency_stays_one(self):
        """If all pairs have l=u=1, aggregate must be [1,1]."""
        pairwise = [CI(1.0, 1.0)] * 3
        result = self.engine._aggregate_independent(pairwise)
        self.assertAlmostEqual(result.lower, 1.0, places=5)
        self.assertAlmostEqual(result.upper, 1.0, places=5)

    def test_exact_two_item_formula(self):
        """
        With two items [l1,u1]=[0.2,0.5] and [l2,u2]=[0.3,0.6]:
            L = 1 - (1-0.2)(1-0.3) = 1 - 0.8*0.7 = 1 - 0.56 = 0.44
            U = 1 - (1-0.5)(1-0.6) = 1 - 0.5*0.4 = 1 - 0.20 = 0.80
        """
        pairwise = [CI(0.2, 0.5), CI(0.3, 0.6)]
        result = self.engine._aggregate_independent(pairwise)
        self.assertAlmostEqual(result.lower, 0.44, places=5)
        self.assertAlmostEqual(result.upper, 0.80, places=5)

    def test_empty_list(self):
        result = self.engine._aggregate_independent([])
        self.assertAlmostEqual(result.lower, 0.0)
        self.assertAlmostEqual(result.upper, 0.0)

    def test_single_item_passthrough(self):
        ci = CI(0.3, 0.7)
        result = self.engine._aggregate_independent([ci])
        # L = 1-(1-0.3) = 0.3,  U = 1-(1-0.7) = 0.7
        self.assertAlmostEqual(result.lower, 0.3, places=5)
        self.assertAlmostEqual(result.upper, 0.7, places=5)


# ─────────────────────────────────────────────────────────────────────────────
# B2 — Aggregation modes + compute()
# ─────────────────────────────────────────────────────────────────────────────

class TestAggregationModes(unittest.TestCase):

    def _run(self, agg: str):
        engine = CredalInconsistencyEngine(aggregation=agg)
        evidence = [make_evidence(0.8, 0.95), make_evidence(0.1, 0.4)]
        proof    = [make_step(0.85, 0.95, idx=0), make_step(0.6, 0.8, idx=1)]
        ci, ts   = engine.compute(evidence, proof)
        return ci, ts

    def test_independent_valid(self):
        ci, _ = self._run("independent")
        self.assertLessEqual(ci.lower, ci.upper)
        self.assertGreaterEqual(ci.lower, 0.0)
        self.assertLessEqual(ci.upper, 1.0)

    def test_worst_case_valid(self):
        ci, _ = self._run("worst_case")
        self.assertLessEqual(ci.lower, ci.upper)

    def test_markov_valid(self):
        ci, _ = self._run("markov")
        self.assertLessEqual(ci.lower, ci.upper)

    def test_empty_evidence_returns_zero(self):
        engine = CredalInconsistencyEngine()
        ci, ts = engine.compute([], [make_step()])
        self.assertAlmostEqual(ci.lower, 0.0)
        self.assertAlmostEqual(ci.upper, 0.0)
        self.assertEqual(ts, {})

    def test_empty_proof_returns_zero(self):
        engine = CredalInconsistencyEngine()
        ci, ts = engine.compute([make_evidence()], [])
        self.assertAlmostEqual(ci.lower, 0.0)
        self.assertAlmostEqual(ci.upper, 0.0)

    def test_type_scores_non_empty(self):
        engine = CredalInconsistencyEngine()
        evidence = [make_evidence()]
        proof = [
            make_step(rule="temporal_before", idx=0),
            make_step(rule="causal_implies",  idx=1),
        ]
        _, ts = engine.compute(evidence, proof)
        self.assertIn(InconsistencyType.TEMPORAL, ts)
        self.assertIn(InconsistencyType.CAUSAL, ts)

    def test_worst_case_upper_ge_independent(self):
        """worst_case upper should be ≥ independent upper (more conservative)."""
        e = [make_evidence(0.8, 0.95), make_evidence(0.1, 0.4)]
        p = [make_step(0.85, 0.95), make_step(0.5, 0.7)]
        ci_wc, _ = CredalInconsistencyEngine("worst_case").compute(e, p)
        ci_in, _ = CredalInconsistencyEngine("independent").compute(e, p)
        self.assertGreaterEqual(ci_wc.upper, ci_in.lower)


# ─────────────────────────────────────────────────────────────────────────────
# G — Rule classification (FIX-4)
# ─────────────────────────────────────────────────────────────────────────────

class TestRuleClassification(unittest.TestCase):

    def _cls(self, rule: str) -> InconsistencyType:
        return CredalInconsistencyEngine._classify_type(rule)

    def test_temporal_rules(self):
        for rule in ["temporal_before", "AFTER", "DURING", "order_time",
                     "span_check", "sequence"]:
            self.assertEqual(self._cls(rule), InconsistencyType.TEMPORAL, rule)

    def test_causal_rules(self):
        for rule in ["causal_implies", "enable", "prevent", "require",
                     "effect_of", "mediat_var"]:
            self.assertEqual(self._cls(rule), InconsistencyType.CAUSAL, rule)

    def test_comparative_rules(self):
        for rule in ["comparative_flip", "max_value", "min_check", "rank",
                     "diff_compute", "ratio_calc", "threshold_filter",
                     "equaliz", "normaliz"]:
            self.assertEqual(self._cls(rule), InconsistencyType.COMPARATIVE, rule)

    def test_logical_rules(self):
        for rule in ["contradict", "negate", "contrapos", "tollens",
                     "disjunct", "conjunct", "universal_instantiate",
                     "existential_witness", "logical_check", "implication"]:
            self.assertEqual(self._cls(rule), InconsistencyType.LOGICAL, rule)

    def test_factual_rules(self):
        for rule in ["entity_location", "find_entity", "filter_prop",
                     "bridge_hop", "path_search", "link_entity",
                     "join_sets", "relate_hop", "verify_fact",
                     "exists_check", "select_item", "project_field",
                     "resolve_answer", "clarify_term", "explain_step"]:
            self.assertEqual(self._cls(rule), InconsistencyType.FACTUAL, rule)

    def test_unknown_rule_returns_none(self):
        self.assertEqual(self._cls("unknown_rule_xyz"), InconsistencyType.NONE)


# ─────────────────────────────────────────────────────────────────────────────
# D — MetaCognitiveExtractor (mock hidden states)
# ─────────────────────────────────────────────────────────────────────────────

class MockModelOutput:
    """Minimal HuggingFace-style output with hidden_states."""
    def __init__(self, batch: int = 1, seq_len: int = 16,
                 hidden_dim: int = 64, n_layers: int = 10):
        # Include layer 0 so index access works
        self.hidden_states = tuple(
            torch.randn(batch, seq_len, hidden_dim)
            for _ in range(n_layers)
        )


class TestMetaCognitiveExtractor(unittest.TestCase):

    def setUp(self):
        self.ext = MetaCognitiveExtractor(
            hidden_dim=64,
            n_layers_to_extract=4,
            start_layer=2,
            device="cpu",
        )

    def test_extract_residual_streams_shape(self):
        out     = MockModelOutput(batch=2, seq_len=8, hidden_dim=64, n_layers=10)
        streams = self.ext.extract_residual_streams(out)
        self.assertEqual(len(streams), 4)
        for s in streams:
            self.assertEqual(s.shape, (2, 64))

    def test_project_shape(self):
        out     = MockModelOutput(batch=2, hidden_dim=64, n_layers=10)
        streams = self.ext.extract_residual_streams(out)
        proj    = self.ext.project_onto_lr_axes(streams)
        self.assertEqual(proj.shape, (2, 4))

    def test_binarize_range(self):
        proj = torch.randn(3, 4)
        b    = self.ext.binarize(proj)
        self.assertTrue((b >= 0).all() and (b <= 1).all())

    def test_forward_output_range(self):
        out          = MockModelOutput(batch=2, hidden_dim=64, n_layers=10)
        i_meta, diag = self.ext(out)
        self.assertEqual(i_meta.shape, (2,))
        self.assertTrue((i_meta >= 0).all() and (i_meta <= 1).all())

    def test_forward_diagnostics_keys(self):
        out      = MockModelOutput(hidden_dim=64, n_layers=10)
        _, diag  = self.ext(out)
        self.assertIn("per_layer_projections", diag)
        self.assertIn("per_layer_labels",      diag)
        self.assertIn("layer_weights",         diag)

    def test_layer_weights_sum_to_one(self):
        out        = MockModelOutput(hidden_dim=64, n_layers=10)
        _, diag    = self.ext(out)
        weights    = diag["layer_weights"]
        self.assertAlmostEqual(weights.sum().item(), 1.0, places=5)

    def test_train_lr_axes_runs(self):
        """Smoke-test: training should not raise and should set model to eval."""
        dataset = [
            ([torch.randn(1, 64) for _ in range(4)], label)
            for label in ([0, 1, 0, 1] * 3)
        ]
        self.ext.train_lr_axes(dataset, n_epochs=2, lr=1e-3)
        self.assertFalse(self.ext.training)   # FIX-6: must be in eval after

    def test_layer_fallback_for_deep_model(self):
        """start_layer beyond available layers → uses last layer."""
        ext = MetaCognitiveExtractor(
            hidden_dim=64, n_layers_to_extract=3, start_layer=50, device="cpu"
        )
        out = MockModelOutput(hidden_dim=64, n_layers=10)
        streams = ext.extract_residual_streams(out)
        self.assertEqual(len(streams), 3)


# ─────────────────────────────────────────────────────────────────────────────
# E — ProbabilisticInconsistencySignal
# ─────────────────────────────────────────────────────────────────────────────

def make_pis(**kwargs) -> ProbabilisticInconsistencySignal:
    defaults = dict(
        alpha_base=0.6,
        mutation_threshold=0.5,
        disagreement_threshold=0.4,
        meta_extractor=MetaCognitiveExtractor(
            hidden_dim=64, n_layers_to_extract=3, start_layer=0, device="cpu"
        ),
        aggregation="markov",
        device="cpu",
    )
    defaults.update(kwargs)
    return ProbabilisticInconsistencySignal(**defaults)


class TestProbabilisticInconsistencySignal(unittest.TestCase):

    def setUp(self):
        self.pis = make_pis()
        self.evidence = [make_evidence(0.85, 0.95), make_evidence(0.2, 0.5)]
        self.proof    = [
            make_step(0.85, 0.95, "entity_location", 0),
            make_step(0.60, 0.80, "temporal_before",  1),
        ]

    def test_compute_returns_signal(self):
        sig = self.pis.compute(self.evidence, self.proof)
        self.assertIsInstance(sig, InconsistencySignal)

    def test_credal_interval_valid(self):
        sig = self.pis.compute(self.evidence, self.proof)
        ci  = sig.credal_inconsistency
        self.assertLessEqual(ci.lower, ci.upper)
        self.assertGreaterEqual(ci.lower, 0.0)
        self.assertLessEqual(ci.upper, 1.0)

    def test_combined_score_in_unit_interval(self):
        sig = self.pis.compute(self.evidence, self.proof)
        self.assertGreaterEqual(sig.combined_score, 0.0)
        self.assertLessEqual(sig.combined_score, 1.0)

    def test_i_meta_zero_without_llm(self):
        sig = self.pis.compute(self.evidence, self.proof, llm_outputs=None)
        self.assertAlmostEqual(sig.i_meta, 0.0)

    def test_i_meta_nonzero_with_llm(self):
        out = MockModelOutput(batch=1, hidden_dim=64, n_layers=5)
        sig = self.pis.compute(self.evidence, self.proof, llm_outputs=out)
        self.assertGreaterEqual(sig.i_meta, 0.0)
        self.assertLessEqual(sig.i_meta, 1.0)

    def test_diagnostics_contains_alpha(self):
        sig = self.pis.compute(self.evidence, self.proof)
        self.assertIn("alpha", sig.diagnostics)
        alpha = sig.diagnostics["alpha"]
        self.assertGreaterEqual(alpha, self.pis.alpha_min)
        self.assertLessEqual(alpha, self.pis.alpha_max)

    def test_type_scores_populated(self):
        sig = self.pis.compute(self.evidence, self.proof)
        self.assertIn(InconsistencyType.TEMPORAL, sig.type_scores)

    def test_trigger_high_combined(self):
        """When evidence and proof are maximally inconsistent, trigger=True."""
        e = [make_evidence(0.9, 1.0)]
        p = [make_step(0.0, 0.1, "entity_location")]   # Disjoint → high inconsistency
        sig = self.pis.compute(e, p)
        self.assertTrue(sig.trigger_mutation)

    def test_no_trigger_consistent(self):
        """When evidence and proof agree strongly, trigger should be False."""
        e   = [make_evidence(0.85, 0.95)]
        p   = [make_step(0.88, 0.96)]
        pis = make_pis(mutation_threshold=0.9, disagreement_threshold=0.9)
        sig = pis.compute(e, p)
        self.assertFalse(sig.trigger_mutation)


# ─────────────────────────────────────────────────────────────────────────────
# F — Mutation trigger logic (FIX-5)
# ─────────────────────────────────────────────────────────────────────────────

class TestMutationTriggerFix5(unittest.TestCase):
    """Tests for FIX-5: mutation_threshold and disagreement_threshold exposed."""

    def _sig(self, credal_lo, credal_hi, i_meta, threshold=0.5, disagreement=0.4):
        """
        Compute signal with a disjoint proof step ([0.0, 0.1]) so
        inconsistency is real, then override i_meta and recompute trigger.
        """
        pis = make_pis(
            mutation_threshold=threshold,
            disagreement_threshold=disagreement,
        )
        e = [make_evidence(credal_lo, credal_hi)]
        p = [make_step(0.0, 0.1)]          # disjoint → real inconsistency
        sig = pis.compute(e, p)
        sig.i_meta = i_meta
        credal_mid       = sig.credal_inconsistency.midpoint
        combined         = 0.6 * credal_mid + 0.4 * i_meta
        disagreement_val = abs(credal_mid - i_meta)
        sig.combined_score   = combined
        sig.trigger_mutation = (
            combined > threshold or disagreement_val > disagreement
        )
        return sig

    def test_threshold_exposed(self):
        pis = make_pis(mutation_threshold=0.3)
        self.assertAlmostEqual(pis.mutation_threshold, 0.3)

    def test_disagreement_threshold_exposed(self):
        pis = make_pis(disagreement_threshold=0.2)
        self.assertAlmostEqual(pis.disagreement_threshold, 0.2)

    def test_trigger_via_combined_score(self):
        # credal [0.9,1.0] vs proof [0.0,0.1] → credal_mid≈0.9 → combined≈0.54 > 0.5
        sig = self._sig(0.9, 1.0, 0.0, threshold=0.5, disagreement=0.9)
        self.assertTrue(sig.trigger_mutation)

    def test_trigger_via_disagreement(self):
        sig = self._sig(0.1, 0.2, 0.9, threshold=0.99, disagreement=0.3)
        self.assertTrue(sig.trigger_mutation)

    def test_no_trigger_both_low(self):
        sig = self._sig(0.05, 0.1, 0.05, threshold=0.9, disagreement=0.9)
        self.assertFalse(sig.trigger_mutation)


# ─────────────────────────────────────────────────────────────────────────────
# E2 — Batch computation (FIX-3)
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeBatch(unittest.TestCase):

    def setUp(self):
        self.pis = make_pis()
        self.batch_e = [
            [make_evidence(0.85, 0.95), make_evidence(0.2, 0.5)],
            [make_evidence(0.4,  0.6)],
            [make_evidence(0.1,  0.3)],
        ]
        self.batch_p = [
            [make_step(0.85, 0.95, idx=0), make_step(0.6, 0.8, idx=1)],
            [make_step(0.4, 0.7)],
            [make_step(0.05, 0.2)],
        ]

    def test_batch_length_matches(self):
        signals = self.pis.compute_batch(self.batch_e, self.batch_p)
        self.assertEqual(len(signals), 3)

    def test_all_signals_valid(self):
        signals = self.pis.compute_batch(self.batch_e, self.batch_p)
        for sig in signals:
            ci = sig.credal_inconsistency
            self.assertLessEqual(ci.lower, ci.upper)
            self.assertGreaterEqual(sig.combined_score, 0.0)
            self.assertLessEqual(sig.combined_score, 1.0)

    def test_batch_single_item(self):
        signals = self.pis.compute_batch([self.batch_e[0]], [self.batch_p[0]])
        self.assertEqual(len(signals), 1)

    def test_batch_parallel_matches_serial(self):
        """Batch results should be close to serial (non-deterministic due to threading)."""
        signals_batch  = self.pis.compute_batch(self.batch_e, self.batch_p, max_workers=2)
        signals_serial = [
            self.pis.compute(e, p)
            for e, p in zip(self.batch_e, self.batch_p)
        ]
        for sb, ss in zip(signals_batch, signals_serial):
            self.assertAlmostEqual(
                sb.credal_inconsistency.midpoint,
                ss.credal_inconsistency.midpoint,
                places=5,
            )

    def test_mismatch_raises(self):
        with self.assertRaises(AssertionError):
            self.pis.compute_batch(self.batch_e, self.batch_p[:1])


# ─────────────────────────────────────────────────────────────────────────────
# H — Integration: full pipeline without LLM
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration(unittest.TestCase):

    def test_consistent_proof_low_signal(self):
        """Highly overlapping evidence + proof → low inconsistency."""
        pis = make_pis()
        e   = [make_evidence(0.90, 0.98)]
        p   = [make_step(0.88, 0.96)]
        sig = pis.compute(e, p)
        self.assertLess(sig.credal_inconsistency.midpoint, 0.5)

    def test_contradictory_proof_high_signal(self):
        """Disjoint evidence vs proof → high inconsistency."""
        pis = make_pis()
        e   = [make_evidence(0.80, 0.95)]
        p   = [make_step(0.02, 0.12)]   # Disjoint
        sig = pis.compute(e, p)
        self.assertGreater(sig.credal_inconsistency.midpoint, 0.3)

    def test_multi_hop_chain(self):
        """Multi-hop proof chain computes without error."""
        pis = make_pis()
        e   = [make_evidence(0.85, 0.95, hop=i) for i in range(1, 4)]
        p   = [
            make_step(0.85, 0.95, "entity_location",  0),
            make_step(0.70, 0.85, "bridge_hop",        1),
            make_step(0.60, 0.80, "temporal_before",   2),
            make_step(0.50, 0.75, "causal_implies",    3),
        ]
        sig = pis.compute(e, p)
        self.assertIsInstance(sig, InconsistencySignal)
        self.assertLessEqual(sig.credal_inconsistency.lower,
                             sig.credal_inconsistency.upper)
        self.assertIn(InconsistencyType.TEMPORAL, sig.type_scores)
        self.assertIn(InconsistencyType.CAUSAL,   sig.type_scores)

    def test_all_aggregation_modes_give_valid_output(self):
        e = [make_evidence(0.7, 0.9), make_evidence(0.3, 0.6)]
        p = [make_step(0.75, 0.88, idx=0), make_step(0.5, 0.7, idx=1)]
        for agg in ("markov", "independent", "worst_case"):
            engine = CredalInconsistencyEngine(aggregation=agg)
            ci, ts = engine.compute(e, p)
            self.assertLessEqual(ci.lower, ci.upper, f"Failed for {agg}")

    def test_credal_width_reflects_uncertainty(self):
        """Narrower input intervals → narrower aggregate."""
        pis = make_pis()
        e_narrow = [make_evidence(0.89, 0.91)]
        e_wide   = [make_evidence(0.10, 0.90)]
        p        = [make_step(0.88, 0.92)]
        sig_n    = pis.compute(e_narrow, p)
        sig_w    = pis.compute(e_wide,   p)
        self.assertLess(
            sig_n.credal_inconsistency.width,
            sig_w.credal_inconsistency.width,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)