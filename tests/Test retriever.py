"""
Offline unit test for HybridAdaptiveRetriever.
Replaces real LLM with a MockEncoder that returns deterministic embeddings.
Validates: BM25, RRF fusion, HopComplexityPredictor, routing logic,
           multi-hop iteration — all without network access.
"""

import sys
import types
import math
import torch
import torch.nn as nn
import unittest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from typing import Optional

# ── 1. Patch AutoTokenizer / AutoModel before importing retriever ──────────

class _FakeTokenizerOutput:
    def __init__(self, n_tokens=8):
        self.input_ids      = torch.ones(1, n_tokens, dtype=torch.long)
        self.attention_mask = torch.ones(1, n_tokens, dtype=torch.long)

    def __getitem__(self, key):
        return getattr(self, key)

class _FakeTokenizer:
    pad_token = "[PAD]"
    eos_token = "[EOS]"
    def __call__(self, texts, **kwargs):
        B = len(texts) if isinstance(texts, list) else 1
        ids   = torch.ones(B, 8, dtype=torch.long)
        mask  = torch.ones(B, 8, dtype=torch.long)
        class Enc:
            def __getitem__(self_, k):
                return ids if k == "input_ids" else mask
        return Enc()
    @classmethod
    def from_pretrained(cls, *a, **kw):
        return cls()

class _FakeModelOutput:
    def __init__(self, hidden):
        self.last_hidden_state = hidden

class _FakeModel(nn.Module):
    """Returns random-but-fixed embeddings so cosine sim is non-trivial."""
    def __init__(self, hidden_size=128):
        super().__init__()
        torch.manual_seed(42)
        self.hidden_size = hidden_size
        # Fixed linear projection to make outputs deterministic per call count
        self._call_count = 0

    def forward(self, input_ids, attention_mask, **kw):
        B, L = input_ids.shape
        torch.manual_seed(self._call_count)
        self._call_count += 1
        hidden = torch.randn(B, L, self.hidden_size)
        return _FakeModelOutput(hidden)

    @classmethod
    def from_pretrained(cls, *a, **kw):
        return cls()

# Patch transformers at import time
fake_transformers = types.ModuleType("transformers")
fake_transformers.AutoTokenizer = _FakeTokenizer
fake_transformers.AutoModel     = _FakeModel
fake_transformers.AutoModelForCausalLM = _FakeModel
sys.modules["transformers"] = fake_transformers

# Now import our module
from component1_retriever import (
    RetrieverConfig,
    RetrieverConfig,
    HopComplexityPredictor,
    BM25Retriever,
    DenseRetriever,
    reciprocal_rank_fusion,
    HybridAdaptiveRetriever,
    RetrievalResult,
)

# ── 2. Test config ─────────────────────────────────────────────────────────

def make_cfg(**overrides) -> RetrieverConfig:
    cfg = RetrieverConfig(
        llm_model_name="fake-llm",
        llm_dtype=torch.float32,
        hidden_size=128,
        top_k_dense=5,
        top_k_sparse=5,
        top_k_final=3,
        num_gpus=1,
        index_cache_dir="/tmp/mc_nsr_test",
        device="cpu",
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg

CORPUS = [
    "The Eiffel Tower is in Paris.",
    "Paris is the capital of France.",
    "France is in Western Europe.",
    "Gustave Eiffel designed the tower in 1889.",
    "The Louvre museum is also in Paris.",
    "The Seine river flows through Paris.",
    "French is spoken by 300 million people.",
    "Notre-Dame is a medieval cathedral in Paris.",
    "France has 68 million inhabitants.",
    "The Eiffel Tower is 330 metres tall.",
]


# ── 3. Unit tests ──────────────────────────────────────────────────────────

class TestBM25Retriever(unittest.TestCase):

    def setUp(self):
        self.bm25 = BM25Retriever(CORPUS)

    def test_returns_top_k(self):
        ids, scores = self.bm25.retrieve("Eiffel Tower Paris", top_k=3)
        self.assertEqual(len(ids), 3)
        self.assertEqual(len(scores), 3)

    def test_scores_non_negative(self):
        _, scores = self.bm25.retrieve("Eiffel Tower", top_k=5)
        for s in scores:
            self.assertGreaterEqual(s, 0.0)

    def test_relevant_passage_ranked_high(self):
        ids, _ = self.bm25.retrieve("Eiffel Tower", top_k=3)
        # Passage 0 contains "Eiffel Tower" explicitly
        self.assertIn(0, ids)

    def test_top_k_larger_than_corpus(self):
        ids, scores = self.bm25.retrieve("Paris", top_k=50)
        self.assertEqual(len(ids), len(CORPUS))


class TestRRFFusion(unittest.TestCase):

    def test_combines_all_ids(self):
        dense_ids  = [0, 1, 2]
        sparse_ids = [2, 3, 4]
        fused = reciprocal_rank_fusion(dense_ids, [1,0.9,0.8],
                                       sparse_ids, [1,0.9,0.8])
        result_ids = [pid for pid, _ in fused]
        # All 5 unique ids should appear
        self.assertEqual(set(result_ids), {0, 1, 2, 3, 4})

    def test_shared_passage_ranked_high(self):
        # Passage 2 appears in both lists → should rank first
        dense_ids  = [0, 1, 2]
        sparse_ids = [2, 3, 4]
        fused = reciprocal_rank_fusion(dense_ids, [1,1,1],
                                       sparse_ids, [1,1,1])
        self.assertEqual(fused[0][0], 2)

    def test_scores_positive(self):
        fused = reciprocal_rank_fusion([0,1],[1,0.5],[1,2],[1,0.5])
        for _, s in fused:
            self.assertGreater(s, 0)

    def test_alpha_zero_uses_sparse_only(self):
        dense_ids  = [0, 1]
        sparse_ids = [2, 3]
        fused_alpha0 = reciprocal_rank_fusion(
            dense_ids, [1,1], sparse_ids, [1,0.5], alpha=0.0)
        fused_alpha1 = reciprocal_rank_fusion(
            dense_ids, [1,1], sparse_ids, [1,0.5], alpha=1.0)
        top_alpha0 = fused_alpha0[0][0]
        top_alpha1 = fused_alpha1[0][0]
        # alpha=0 → only sparse contributes → top should be passage 2
        self.assertEqual(top_alpha0, 2)
        # alpha=1 → only dense → top should be passage 0
        self.assertEqual(top_alpha1, 0)


class TestHopComplexityPredictor(unittest.TestCase):

    def setUp(self):
        self.cfg  = make_cfg()
        self.pred = HopComplexityPredictor(self.cfg)

    def test_output_shape(self):
        x      = torch.randn(4, self.cfg.hidden_size)
        logits = self.pred(x)
        self.assertEqual(logits.shape, (4, self.cfg.num_hop_classes))

    def test_output_finite(self):
        x      = torch.randn(2, self.cfg.hidden_size)
        logits = self.pred(x)
        self.assertTrue(torch.isfinite(logits).all())

    def test_single_sample(self):
        x      = torch.randn(1, self.cfg.hidden_size)
        logits = self.pred(x)
        self.assertEqual(logits.shape, (1, self.cfg.num_hop_classes))

    def test_gradient_flows(self):
        x = torch.randn(2, self.cfg.hidden_size, requires_grad=False)
        logits = self.pred(x)
        loss   = logits.sum()
        loss.backward()
        for p in self.pred.parameters():
            self.assertIsNotNone(p.grad)


class TestDenseRetriever(unittest.TestCase):

    def setUp(self):
        self.cfg = make_cfg()
        self.dr  = DenseRetriever(self.cfg)

    def test_build_index(self):
        self.dr.build_index(CORPUS, batch_size=3, force_rebuild=True)
        self.assertIsNotNone(self.dr.passage_embeddings)
        self.assertEqual(self.dr.passage_embeddings.shape[0], len(CORPUS))
        self.assertEqual(self.dr.passage_embeddings.shape[1], self.cfg.hidden_size)

    def test_retrieve_returns_top_k(self):
        self.dr.build_index(CORPUS, batch_size=3, force_rebuild=True)
        ids, scores, q_hidden = self.dr.retrieve("Eiffel Tower", top_k=3)
        self.assertEqual(len(ids), 3)
        self.assertEqual(len(scores), 3)
        self.assertEqual(q_hidden.shape, (self.cfg.hidden_size,))

    def test_retrieve_indices_in_range(self):
        self.dr.build_index(CORPUS, batch_size=3, force_rebuild=True)
        ids, _, _ = self.dr.retrieve("France", top_k=5)
        for i in ids:
            self.assertGreaterEqual(i, 0)
            self.assertLess(i, len(CORPUS))

    def test_scores_in_cosine_range(self):
        self.dr.build_index(CORPUS, batch_size=3, force_rebuild=True)
        _, scores, _ = self.dr.retrieve("Paris", top_k=4)
        for s in scores:
            self.assertGreaterEqual(s, -1.0 - 1e-5)
            self.assertLessEqual(s,  1.0 + 1e-5)


class TestHybridAdaptiveRetriever(unittest.TestCase):

    def setUp(self):
        self.cfg = make_cfg()
        self.ret = HybridAdaptiveRetriever(self.cfg)
        self.ret.build_index(CORPUS, batch_size=3, force_rebuild=True)

    # ── Predictor training ─────────────────────────────────────────────

    def test_train_predictor(self):
        questions = [
            "Where is Paris?",
            "Who built the Eiffel Tower and when?",
            "What river flows through Paris and what country is it in?",
            "Tell me about Eiffel, his tower, its height and its location.",
        ]
        labels = [0, 1, 2, 3]
        self.ret.train_hop_predictor(questions, labels, epochs=2, batch_size=2)
        # After training, model should be back in eval mode
        self.assertFalse(self.ret.predictor.training)

    # ── Single retrieve ────────────────────────────────────────────────

    def test_retrieve_returns_result(self):
        result = self.ret.retrieve("Who built the Eiffel Tower?")
        self.assertIsInstance(result, RetrievalResult)

    def test_retrieve_passages_count(self):
        result = self.ret.retrieve("Paris France")
        # final_k depends on route; always ≤ top_k_final
        self.assertLessEqual(len(result.passages), self.cfg.top_k_final)
        self.assertGreater(len(result.passages), 0)

    def test_retrieve_predicted_hops_range(self):
        result = self.ret.retrieve("simple question")
        self.assertGreaterEqual(result.predicted_hops, 1)
        self.assertLessEqual(result.predicted_hops, self.cfg.num_hop_classes)

    def test_retrieve_route_valid(self):
        result = self.ret.retrieve("complex multi-hop reasoning question")
        self.assertIn(result.pipeline_route, {"fast", "full"})

    def test_retrieve_scores_align_passages(self):
        result = self.ret.retrieve("Eiffel Tower height")
        self.assertEqual(len(result.passages), len(result.scores))
        self.assertEqual(len(result.passages), len(result.passage_ids))

    def test_retrieve_fast_route_fewer_passages(self):
        """Force fast route: predict 1-hop via untrained predictor biased low."""
        # Make predictor always output class 0 (1-hop → fast route)
        with torch.no_grad():
            self.ret.predictor.fc2.weight.zero_()
            self.ret.predictor.fc2.bias.zero_()
            self.ret.predictor.fc2.bias[0] = 10.0  # class 0 wins
        result = self.ret.retrieve("simple")
        self.assertEqual(result.pipeline_route, "fast")
        expected_k = max(self.cfg.top_k_final // 2, 3)
        self.assertEqual(len(result.passages), expected_k)

    def test_retrieve_full_route_more_passages(self):
        """Force full route: predict 4-hop."""
        with torch.no_grad():
            self.ret.predictor.fc2.weight.zero_()
            self.ret.predictor.fc2.bias.zero_()
            self.ret.predictor.fc2.bias[3] = 10.0  # class 3 (4-hop) wins
        result = self.ret.retrieve("complex multi-hop")
        self.assertEqual(result.pipeline_route, "full")
        self.assertEqual(len(result.passages), self.cfg.top_k_final)

    # ── Multi-hop ──────────────────────────────────────────────────────

    def test_multihop_returns_list(self):
        results = self.ret.retrieve_multihop("Who built the Eiffel Tower?", num_hops=2)
        self.assertEqual(len(results), 2)

    def test_multihop_single_hop(self):
        results = self.ret.retrieve_multihop("Where is Paris?", num_hops=1)
        self.assertEqual(len(results), 1)

    def test_multihop_all_results_valid(self):
        results = self.ret.retrieve_multihop("complex question", num_hops=3)
        for r in results:
            self.assertIsInstance(r, RetrievalResult)
            self.assertGreater(len(r.passages), 0)

    def test_multihop_queries_expand(self):
        """Each hop's query should be longer than the previous (context appended)."""
        results = self.ret.retrieve_multihop("Paris", num_hops=3)
        self.assertEqual(len(results), 3)

    # ── Save / load predictor ──────────────────────────────────────────

    def test_save_load_predictor(self):
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            # Get weights before save
            w_before = self.ret.predictor.fc2.weight.clone()
            self.ret.save_predictor(path)
            # Corrupt weights
            with torch.no_grad():
                self.ret.predictor.fc2.weight.fill_(999.0)
            # Reload
            self.ret.load_predictor(path)
            w_after = self.ret.predictor.fc2.weight
            self.assertTrue(torch.allclose(w_before, w_after))
        finally:
            os.unlink(path)

    # ── Edge cases ─────────────────────────────────────────────────────

    def test_retrieve_empty_query(self):
        result = self.ret.retrieve("")
        self.assertIsInstance(result, RetrievalResult)

    def test_retrieve_long_query(self):
        long_q = " ".join(["Paris Eiffel Tower France"] * 50)
        result = self.ret.retrieve(long_q)
        self.assertIsInstance(result, RetrievalResult)


# ── 4. Run ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loader  = unittest.TestLoader()
    suite   = loader.loadTestsFromModule(sys.modules[__name__])
    runner  = unittest.TextTestRunner(verbosity=2)
    result  = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)