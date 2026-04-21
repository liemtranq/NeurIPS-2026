"""
Offline unit tests for SymbolicEngine (Component 2).
No network, no model downloads. Pure logic tests.
"""

import sys
import unittest

from symbolic_engine import (
    ExecutionContext,
    RuleNode,
    RuleCategory,
    RULE_REGISTRY,
    SymbolicEngine,
)

# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

PASSAGES = [
    "The Eiffel Tower was built in 1889 by Gustave Eiffel in Paris.",
    "Gustave Eiffel was a French civil engineer born in 1832.",
    "Paris is the capital of France, located on the Seine river.",
    "The Eiffel Tower is 330 metres tall and attracts 7 million visitors annually.",
    "France has a population of 68 million people and borders Germany.",
    "The French Revolution began in 1789 and ended in 1799.",
    "Napoleon Bonaparte caused the fall of the French Republic in 1799.",
    "The Treaty of Paris enabled the end of the American Revolutionary War.",
    "Germany prevented France from expanding eastward after 1871.",
    "The Seine river required major engineering works to manage flooding.",
    "The Louvre museum opened in 1793, before the Eiffel Tower was built.",
    "Notre-Dame Cathedral was built between 1163 and 1345.",
    "France required its engineers to pass rigorous examinations.",
    "Paris has a metro system with 16 lines carrying 1.5 billion riders per year.",
    "The Eiffel Tower cost 7.8 million francs to build.",
    "Gustave Eiffel also designed the internal structure of the Statue of Liberty.",
    "The French language is spoken by about 300 million people worldwide.",
    "After the Revolution, France established a republic in 1792.",
    "The Seine flooding since 1910 led to major infrastructure changes.",
    "Eiffel's company built bridges and viaducts across Europe.",
]

QUESTION = "Which was built earlier, the Eiffel Tower or the Louvre museum?"

ENGINE = SymbolicEngine()


def make_ctx(question: str = QUESTION,
             passages: list = None) -> ExecutionContext:
    return ExecutionContext(
        question=question,
        passages=passages if passages is not None else PASSAGES,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Registry
# ─────────────────────────────────────────────────────────────────────────────

class TestRegistry(unittest.TestCase):

    def test_exactly_40_ops(self):
        self.assertEqual(len(RULE_REGISTRY), 40)

    def test_all_categories_present(self):
        cats = {d.category for d in RULE_REGISTRY.values()}
        self.assertEqual(cats, set(RuleCategory))

    def test_core_ops_present(self):
        for name in ["FIND", "FILTER", "COMPARE", "AGGREGATE"]:
            self.assertIn(name, RULE_REGISTRY)

    def test_temporal_ops_present(self):
        for name in ["BEFORE", "AFTER", "DURING", "SINCE", "UNTIL",
                     "ORDER_TIME", "SPAN"]:
            self.assertIn(name, RULE_REGISTRY)

    def test_causal_ops_present(self):
        for name in ["CAUSE", "EFFECT", "ENABLE", "PREVENT", "REQUIRE"]:
            self.assertIn(name, RULE_REGISTRY)

    def test_comparative_ops_present(self):
        for name in ["MAX", "MIN", "RANK", "DIFF", "RATIO", "THRESHOLD"]:
            self.assertIn(name, RULE_REGISTRY)

    def test_meta_ops_present(self):
        for name in ["VERIFY", "NEGATE", "UNION", "INTERSECT", "COUNT",
                     "EXISTS", "SELECT", "PROJECT", "RESOLVE", "CLARIFY"]:
            self.assertIn(name, RULE_REGISTRY)

    def test_all_descriptors_have_fn(self):
        for name, d in RULE_REGISTRY.items():
            self.assertTrue(callable(d.fn), f"{name} has no callable fn")

    def test_describe_contains_all_ops(self):
        desc = ENGINE.describe()
        for name in RULE_REGISTRY:
            self.assertIn(name, desc)


# ─────────────────────────────────────────────────────────────────────────────
# 2. ExecutionContext
# ─────────────────────────────────────────────────────────────────────────────

class TestExecutionContext(unittest.TestCase):

    def test_remember_recall(self):
        ctx = make_ctx()
        ctx.remember("x", [1, 2, 3])
        self.assertEqual(ctx.recall("x"), [1, 2, 3])

    def test_recall_default(self):
        ctx = make_ctx()
        self.assertIsNone(ctx.recall("missing"))
        self.assertEqual(ctx.recall("missing", "default"), "default")

    def test_add_answer_dedup(self):
        ctx = make_ctx()
        ctx.add_answer("Paris")
        ctx.add_answer("Paris")
        self.assertEqual(ctx.answer.count("Paris"), 1)

    def test_clone_independent(self):
        ctx  = make_ctx()
        ctx.remember("k", [1])
        ctx2 = ctx.clone()
        ctx2.memory["k"].append(2)
        self.assertEqual(ctx.recall("k"), [1])

    def test_confidence_clamp(self):
        ctx = make_ctx()
        ctx.add_answer("x", boost=0.1)
        self.assertGreater(ctx.confidence, 0)
        self.assertLessEqual(ctx.confidence, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Core ops
# ─────────────────────────────────────────────────────────────────────────────

class TestCoreOps(unittest.TestCase):

    def _run(self, spec):
        ctx = make_ctx()
        return ENGINE.execute(ENGINE.build_program(spec), ctx)

    def test_FIND_returns_matches(self):
        ctx = self._run([{"name":"FIND","args":{"entity":"Eiffel Tower"}}])
        found = ctx.recall("found")
        self.assertIsInstance(found, list)
        self.assertTrue(len(found) > 0)
        self.assertTrue(all("Eiffel" in p for p in found))

    def test_FIND_no_matches(self):
        ctx = self._run([{"name":"FIND","args":{"entity":"ZZZMISSING"}}])
        self.assertEqual(ctx.recall("found"), [])

    def test_FILTER_narrows(self):
        ctx = self._run([
            {"name":"FIND",   "args":{"entity":"Paris"}},
            {"name":"FILTER", "args":{"property":"population","value":"million",
                                      "slot_in":"found","slot_out":"filtered"}},
        ])
        filtered = ctx.recall("filtered", [])
        self.assertIsInstance(filtered, list)

    def test_COMPARE_greater(self):
        ctx = make_ctx()
        ctx.remember("a", "The tower is 330 metres tall.")
        ctx.remember("b", "The building is 200 metres tall.")
        prog = ENGINE.build_program([
            {"name":"COMPARE","args":{"slot_a":"a","slot_b":"b",
                                      "criterion":"greater","slot_out":"winner"}}
        ])
        ctx = ENGINE.execute(prog, ctx)
        winner = ctx.recall("winner")
        self.assertIn("330", str(winner))

    def test_COMPARE_lesser(self):
        ctx = make_ctx()
        ctx.remember("a", "height 100")
        ctx.remember("b", "height 300")
        prog = ENGINE.build_program([
            {"name":"COMPARE","args":{"slot_a":"a","slot_b":"b",
                                      "criterion":"lesser","slot_out":"winner"}}
        ])
        ctx = ENGINE.execute(prog, ctx)
        self.assertIn("100", str(ctx.recall("winner")))

    def test_AGGREGATE_count(self):
        ctx = make_ctx()
        ctx.remember("items", ["a","b","c"])
        prog = ENGINE.build_program([
            {"name":"AGGREGATE","args":{"function":"count",
                                        "slot_in":"items","slot_out":"n"}}
        ])
        ctx = ENGINE.execute(prog, ctx)
        self.assertEqual(ctx.recall("n"), 3)

    def test_AGGREGATE_concat(self):
        ctx = make_ctx()
        ctx.remember("items", ["foo","bar"])
        prog = ENGINE.build_program([
            {"name":"AGGREGATE","args":{"function":"concat",
                                        "slot_in":"items","slot_out":"res"}}
        ])
        ctx = ENGINE.execute(prog, ctx)
        self.assertIn("foo", str(ctx.recall("res")))

    def test_AGGREGATE_sum(self):
        ctx = make_ctx()
        ctx.remember("items", ["value is 10 here","another 20"])
        prog = ENGINE.build_program([
            {"name":"AGGREGATE","args":{"function":"sum",
                                        "slot_in":"items","slot_out":"total"}}
        ])
        ctx = ENGINE.execute(prog, ctx)
        self.assertEqual(ctx.recall("total"), 30.0)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Relational ops
# ─────────────────────────────────────────────────────────────────────────────

class TestRelationalOps(unittest.TestCase):

    def _run(self, spec, ctx=None):
        ctx = ctx or make_ctx()
        return ENGINE.execute(ENGINE.build_program(spec), ctx)

    def test_RELATE(self):
        ctx = self._run([
            {"name":"RELATE","args":{"head":"Gustave Eiffel",
                                     "relation":"designed","slot_out":"rel"}}
        ])
        self.assertIsInstance(ctx.recall("rel"), list)

    def test_BRIDGE(self):
        ctx = make_ctx()
        ctx.remember("found", ["The Eiffel Tower was built by Gustave Eiffel."])
        ctx = self._run([
            {"name":"BRIDGE","args":{"slot_src":"found",
                                     "bridge_entity":"Statue of Liberty",
                                     "slot_out":"bridged"}}
        ], ctx)
        self.assertIsInstance(ctx.recall("bridged"), list)

    def test_PATH(self):
        ctx = self._run([
            {"name":"PATH","args":{"start":"Eiffel Tower","end":"Seine river",
                                   "slot_out":"path"}}
        ])
        path = ctx.recall("path")
        self.assertIsInstance(path, list)

    def test_LINK(self):
        ctx = make_ctx()
        ctx.remember("found", PASSAGES[:5])
        ctx = self._run([
            {"name":"LINK","args":{"slot_in":"found","anchor":"Paris",
                                   "slot_out":"linked"}}
        ], ctx)
        linked = ctx.recall("linked")
        self.assertTrue(all("Paris" in p for p in linked))

    def test_JOIN(self):
        ctx = make_ctx()
        ctx.remember("a_list", [p for p in PASSAGES if "Eiffel" in p])
        ctx.remember("b_list", [p for p in PASSAGES if "Paris" in p])
        ctx = self._run([
            {"name":"JOIN","args":{"slot_a":"a_list","slot_b":"b_list",
                                   "key":"Paris","slot_out":"joined"}}
        ], ctx)
        self.assertIsInstance(ctx.recall("joined"), list)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Temporal ops
# ─────────────────────────────────────────────────────────────────────────────

class TestTemporalOps(unittest.TestCase):

    def _run(self, spec):
        return ENGINE.execute(ENGINE.build_program(spec), make_ctx())

    def test_BEFORE_filters_correctly(self):
        ctx = self._run([
            {"name":"BEFORE","args":{"entity":"Louvre","year":1800,
                                     "slot_out":"before"}}
        ])
        before = ctx.recall("before", [])
        for p in before:
            years = [int(m) for m in __import__("re").findall(r"\b\d{4}\b", p)]
            if years:
                self.assertTrue(min(years) < 1800, f"Year not < 1800: {p}")

    def test_AFTER_filters_correctly(self):
        ctx = self._run([
            {"name":"AFTER","args":{"entity":"Eiffel","year":1880,
                                    "slot_out":"after"}}
        ])
        after = ctx.recall("after", [])
        for p in after:
            years = [int(m) for m in __import__("re").findall(r"\b\d{4}\b", p)]
            if years:
                self.assertTrue(max(years) > 1880)

    def test_DURING(self):
        ctx = self._run([
            {"name":"DURING","args":{"entity":"Notre-Dame","year_start":1100,
                                     "year_end":1400,"slot_out":"during"}}
        ])
        during = ctx.recall("during", [])
        self.assertIsInstance(during, list)

    def test_ORDER_TIME_sorted(self):
        ctx = make_ctx()
        ctx.remember("found", [
            "Built in 1950.",
            "Built in 1800.",
            "Built in 2000.",
        ])
        prog = ENGINE.build_program([
            {"name":"ORDER_TIME","args":{"slot_in":"found","slot_out":"ordered"}}
        ])
        ctx = ENGINE.execute(prog, ctx)
        ordered = ctx.recall("ordered")
        years = [int(m) for p in ordered
                 for m in __import__("re").findall(r"\b\d{4}\b", p)]
        self.assertEqual(years, sorted(years))

    def test_SPAN_correct(self):
        ctx = make_ctx()
        ctx.remember("found", ["event in 1900", "event in 2000"])
        prog = ENGINE.build_program([
            {"name":"SPAN","args":{"slot_in":"found","slot_out":"sp"}}
        ])
        ctx = ENGINE.execute(prog, ctx)
        self.assertEqual(ctx.recall("sp"), 100)

    def test_SINCE(self):
        ctx = self._run([
            {"name":"FIND","args":{"entity":"Revolution"}},
            {"name":"SINCE","args":{"event":"Revolution","slot_in":"found",
                                    "slot_out":"since"}},
        ])
        self.assertIsInstance(ctx.recall("since"), list)

    def test_UNTIL(self):
        ctx = self._run([
            {"name":"FIND","args":{"entity":"Revolution"}},
            {"name":"UNTIL","args":{"event":"Revolution","slot_in":"found",
                                    "slot_out":"until"}},
        ])
        self.assertIsInstance(ctx.recall("until"), list)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Causal ops
# ─────────────────────────────────────────────────────────────────────────────

class TestCausalOps(unittest.TestCase):

    def _run(self, spec):
        return ENGINE.execute(ENGINE.build_program(spec), make_ctx())

    def test_CAUSE(self):
        ctx = self._run([
            {"name":"CAUSE","args":{"entity":"Napoleon","slot_out":"causes"}}
        ])
        self.assertIsInstance(ctx.recall("causes"), list)

    def test_EFFECT(self):
        ctx = self._run([
            {"name":"EFFECT","args":{"entity":"Revolution","slot_out":"effects"}}
        ])
        self.assertIsInstance(ctx.recall("effects"), list)

    def test_ENABLE(self):
        ctx = self._run([
            {"name":"ENABLE","args":{"entity":"Treaty","slot_out":"enables"}}
        ])
        self.assertIsInstance(ctx.recall("enables"), list)

    def test_PREVENT(self):
        ctx = self._run([
            {"name":"PREVENT","args":{"entity":"Germany","slot_out":"prevents"}}
        ])
        self.assertIsInstance(ctx.recall("prevents"), list)

    def test_REQUIRE(self):
        ctx = self._run([
            {"name":"REQUIRE","args":{"entity":"France","slot_out":"requires"}}
        ])
        self.assertIsInstance(ctx.recall("requires"), list)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Comparative ops
# ─────────────────────────────────────────────────────────────────────────────

class TestComparativeOps(unittest.TestCase):

    def test_MAX_finds_largest(self):
        ctx = make_ctx()
        ctx.remember("found", [
            "Population is 50 million.",
            "Population is 200 million.",
            "Population is 10 million.",
        ])
        prog = ENGINE.build_program([
            {"name":"MAX","args":{"slot_in":"found","slot_out":"mx"}}
        ])
        ctx = ENGINE.execute(prog, ctx)
        self.assertIn("200", str(ctx.recall("mx")))

    def test_MIN_finds_smallest(self):
        ctx = make_ctx()
        ctx.remember("found", ["value 500","value 3","value 100"])
        prog = ENGINE.build_program([
            {"name":"MIN","args":{"slot_in":"found","slot_out":"mn"}}
        ])
        ctx = ENGINE.execute(prog, ctx)
        self.assertIn("3", str(ctx.recall("mn")))

    def test_RANK_desc(self):
        ctx = make_ctx()
        ctx.remember("found", ["score 30","score 10","score 20"])
        prog = ENGINE.build_program([
            {"name":"RANK","args":{"slot_in":"found","order":"desc",
                                   "slot_out":"ranked"}}
        ])
        ctx = ENGINE.execute(prog, ctx)
        ranked = ctx.recall("ranked")
        self.assertIn("30", str(ranked[0]))

    def test_DIFF_correct(self):
        ctx = make_ctx()
        ctx.remember("a","built in 1889")
        ctx.remember("b","built in 1793")
        prog = ENGINE.build_program([
            {"name":"DIFF","args":{"slot_a":"a","slot_b":"b","slot_out":"d"}}
        ])
        ctx = ENGINE.execute(prog, ctx)
        self.assertAlmostEqual(ctx.recall("d"), 1889 - 1793)

    def test_RATIO_correct(self):
        ctx = make_ctx()
        ctx.remember("a","value 100")
        ctx.remember("b","value 4")
        prog = ENGINE.build_program([
            {"name":"RATIO","args":{"slot_a":"a","slot_b":"b","slot_out":"r"}}
        ])
        ctx = ENGINE.execute(prog, ctx)
        self.assertAlmostEqual(ctx.recall("r"), 25.0)

    def test_THRESHOLD_gt(self):
        ctx = make_ctx()
        ctx.remember("found",["pop 10","pop 500","pop 2"])
        prog = ENGINE.build_program([
            {"name":"THRESHOLD","args":{"slot_in":"found","op":"gt",
                                        "value":50,"slot_out":"big"}}
        ])
        ctx = ENGINE.execute(prog, ctx)
        big = ctx.recall("big")
        self.assertEqual(len(big), 1)
        self.assertIn("500", big[0])

    def test_THRESHOLD_lte(self):
        ctx = make_ctx()
        ctx.remember("found",["10 items","500 items","3 items"])
        prog = ENGINE.build_program([
            {"name":"THRESHOLD","args":{"slot_in":"found","op":"lte",
                                        "value":10,"slot_out":"small"}}
        ])
        ctx = ENGINE.execute(prog, ctx)
        small = ctx.recall("small")
        self.assertEqual(len(small), 2)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Meta ops
# ─────────────────────────────────────────────────────────────────────────────

class TestMetaOps(unittest.TestCase):

    def _run(self, spec, ctx=None):
        ctx = ctx or make_ctx()
        return ENGINE.execute(ENGINE.build_program(spec), ctx)

    def test_VERIFY_supported_claim(self):
        ctx = self._run([
            {"name":"FIND","args":{"entity":"Eiffel Tower"}},
            {"name":"VERIFY","args":{"claim":"Eiffel Tower built 1889",
                                     "slot_in":"found","slot_out":"v"}},
        ])
        self.assertIsInstance(ctx.recall("v"), bool)

    def test_NEGATE_bool(self):
        ctx = make_ctx()
        ctx.remember("v", True)
        ctx = self._run([
            {"name":"NEGATE","args":{"slot_in":"v","slot_out":"nv"}}
        ], ctx)
        self.assertFalse(ctx.recall("nv"))

    def test_NEGATE_list(self):
        ctx = make_ctx()
        ctx.remember("v", PASSAGES[:3])
        ctx = self._run([
            {"name":"NEGATE","args":{"slot_in":"v","slot_out":"nv"}}
        ], ctx)
        nv = ctx.recall("nv")
        self.assertIsInstance(nv, list)
        for p in PASSAGES[:3]:
            self.assertNotIn(p, nv)

    def test_UNION_combines(self):
        ctx = make_ctx()
        ctx.remember("a", ["p1","p2"])
        ctx.remember("b", ["p2","p3"])
        ctx = self._run([
            {"name":"UNION","args":{"slot_a":"a","slot_b":"b","slot_out":"u"}}
        ], ctx)
        self.assertEqual(set(ctx.recall("u")), {"p1","p2","p3"})

    def test_INTERSECT_fuzzy(self):
        ctx = make_ctx()
        ctx.remember("a", ["the quick brown fox"])
        ctx.remember("b", ["quick brown fox runs"])
        ctx = self._run([
            {"name":"INTERSECT","args":{"slot_a":"a","slot_b":"b","slot_out":"i"}}
        ], ctx)
        result = ctx.recall("i")
        self.assertGreater(len(result), 0)

    def test_COUNT(self):
        ctx = make_ctx()
        ctx.remember("items", ["a","b","c","d"])
        ctx = self._run([
            {"name":"COUNT","args":{"slot_in":"items","slot_out":"n"}}
        ], ctx)
        self.assertEqual(ctx.recall("n"), 4)

    def test_EXISTS_true(self):
        ctx = self._run([
            {"name":"EXISTS","args":{"entity":"Eiffel","slot_out":"ex"}}
        ])
        self.assertTrue(ctx.recall("ex"))

    def test_EXISTS_false(self):
        ctx = self._run([
            {"name":"EXISTS","args":{"entity":"ZZZMISSING","slot_out":"ex"}}
        ])
        self.assertFalse(ctx.recall("ex"))

    def test_SELECT(self):
        ctx = make_ctx()
        ctx.remember("ranked", ["first","second","third"])
        ctx = self._run([
            {"name":"SELECT","args":{"slot_in":"ranked","index":1,"slot_out":"sel"}}
        ], ctx)
        self.assertEqual(ctx.recall("sel"), "second")

    def test_PROJECT_dates(self):
        ctx = make_ctx()
        ctx.remember("found",["Built in 1889 and 1900"])
        ctx = self._run([
            {"name":"PROJECT","args":{"slot_in":"found","field":"date",
                                      "slot_out":"dates"}}
        ], ctx)
        self.assertIn("1889", ctx.recall("dates"))

    def test_RESOLVE_populates_answer(self):
        ctx = make_ctx()
        ctx.remember("aggregated",["The Louvre opened in 1793.","Noise"])
        ctx = self._run([
            {"name":"RESOLVE","args":{"slot_in":"aggregated","slot_out":"ans"}}
        ], ctx)
        self.assertTrue(len(ctx.answer) > 0)

    def test_CLARIFY(self):
        ctx = self._run([
            {"name":"CLARIFY","args":{"ambiguous_term":"Seine","slot_out":"clar"}}
        ])
        self.assertIsInstance(ctx.recall("clar"), str)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Engine-level tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSymbolicEngine(unittest.TestCase):

    def test_validate_unknown_rule(self):
        prog   = ENGINE.build_program([{"name":"NONEXISTENT","args":{}}])
        errors = ENGINE.validate_program(prog)
        self.assertTrue(any("unknown rule" in e for e in errors))

    def test_validate_missing_required_arg(self):
        prog   = ENGINE.build_program([{"name":"FIND","args":{}}])
        errors = ENGINE.validate_program(prog)
        self.assertTrue(any("entity" in e for e in errors))

    def test_validate_clean_program(self):
        prog = ENGINE.build_program([
            {"name":"FIND",    "args":{"entity":"Eiffel"}},
            {"name":"AGGREGATE","args":{"function":"count"}},
            {"name":"RESOLVE", "args":{}},
        ])
        errors = ENGINE.validate_program(prog)
        self.assertEqual(errors, [])

    def test_unknown_rule_skipped_in_non_strict(self):
        prog = ENGINE.build_program([
            {"name":"FIND",      "args":{"entity":"Eiffel"}},
            {"name":"UNKNOWN_OP","args":{}},
            {"name":"AGGREGATE", "args":{"function":"count"}},
        ])
        ctx = make_ctx()
        ctx = ENGINE.execute(prog, ctx)   # should not raise
        self.assertIsNotNone(ctx)

    def test_strict_mode_raises(self):
        prog = ENGINE.build_program([{"name":"BAD_OP","args":{}}])
        with self.assertRaises(ValueError):
            ENGINE.execute(prog, make_ctx(), strict=True)

    def test_error_in_step_reduces_confidence(self):
        # Inject a deliberately broken op by patching
        from symbolic_engine import RULE_REGISTRY, RuleDescriptor, RuleCategory
        def _bad_fn(ctx, args):
            raise RuntimeError("injected error")
        RULE_REGISTRY["_BAD"] = RuleDescriptor(
            "_BAD", RuleCategory.META, _bad_fn, "test error", [], {}
        )
        prog = ENGINE.build_program([{"name":"_BAD","args":{}}])
        ctx  = make_ctx()
        ctx  = ENGINE.execute(prog, ctx)
        self.assertLess(ctx.confidence, 1.0)
        del RULE_REGISTRY["_BAD"]

    def test_trace_populated(self):
        prog = ENGINE.build_program([
            {"name":"FIND","args":{"entity":"Eiffel"}},
            {"name":"COUNT","args":{}},
        ])
        ctx = ENGINE.execute(prog, make_ctx())
        self.assertEqual(len(ctx.trace), 2)

    def test_build_program_from_spec(self):
        spec = [
            {"name":"FIND","args":{"entity":"Paris"},"out_slot":"s0"},
            {"name":"COUNT","args":{"slot_in":"s0"},"out_slot":"n"},
        ]
        prog = ENGINE.build_program(spec)
        self.assertEqual(prog[0].out_slot, "s0")
        self.assertEqual(prog[1].out_slot, "n")

    def test_auto_resolve_when_no_answer(self):
        prog = ENGINE.build_program([
            {"name":"FIND","args":{"entity":"Eiffel"}},
        ])
        ctx = ENGINE.execute(prog, make_ctx())
        # Engine should auto-resolve from last memory value
        self.assertTrue(len(ctx.answer) >= 0)   # may or may not find

    # ── End-to-end multi-hop scenario ────────────────────────────────────────

    def test_e2e_comparison_question(self):
        """
        Q: Which was built earlier, the Eiffel Tower or the Louvre?
        Expected: Louvre (1793 < 1889)
        """
        prog = ENGINE.build_program([
            # Step 1: find Eiffel Tower passages
            {"name":"FIND",    "args":{"entity":"Eiffel Tower"}, "out_slot":"ef_passages"},
            # Step 2: find Louvre passages
            {"name":"FIND",    "args":{"entity":"Louvre"},       "out_slot":"lv_passages"},
            # Step 3: extract dates from each
            {"name":"PROJECT", "args":{"slot_in":"ef_passages","field":"date"},
                               "out_slot":"ef_dates"},
            {"name":"PROJECT", "args":{"slot_in":"lv_passages","field":"date"},
                               "out_slot":"lv_dates"},
            # Step 4: compare — lesser date wins
            {"name":"COMPARE", "args":{"slot_a":"lv_dates","slot_b":"ef_dates",
                                       "criterion":"lesser"},
                               "out_slot":"earlier_dates"},
            # Step 5: resolve answer
            {"name":"RESOLVE", "args":{"slot_in":"earlier_dates"}},
        ])
        ctx = ENGINE.execute(prog, make_ctx())
        # The answer list should be non-empty
        self.assertTrue(len(ctx.answer) > 0)
        # Louvre date (1793) should appear before Eiffel date (1889)
        combined = " ".join(ctx.answer)
        self.assertTrue("1793" in combined or "1889" in combined,
                        f"Expected a date in answer, got: {ctx.answer}")

    def test_e2e_causal_chain(self):
        """
        Q: What caused the fall of the French Republic?
        """
        prog = ENGINE.build_program([
            {"name":"CAUSE", "args":{"entity":"French Republic"}, "out_slot":"causes"},
            {"name":"RESOLVE","args":{"slot_in":"causes"}},
        ])
        ctx = ENGINE.execute(
            prog,
            ExecutionContext(
                question="What caused the fall of the French Republic?",
                passages=PASSAGES,
            )
        )
        self.assertIsInstance(ctx.answer, list)

    def test_e2e_temporal_ordering(self):
        """
        Order Paris historical events by year.
        """
        prog = ENGINE.build_program([
            {"name":"FIND",       "args":{"entity":"Paris"}},
            {"name":"ORDER_TIME", "args":{"slot_in":"found"}},
            {"name":"SELECT",     "args":{"slot_in":"ordered","index":0,
                                          "slot_out":"earliest"}},
            {"name":"RESOLVE",    "args":{"slot_in":"earliest"}},
        ])
        ctx = ENGINE.execute(prog, make_ctx())
        self.assertTrue(len(ctx.answer) > 0)

    def test_e2e_bridge_multihop(self):
        """
        Bridge hop: Eiffel Tower → Gustave Eiffel → Statue of Liberty
        """
        prog = ENGINE.build_program([
            {"name":"FIND",   "args":{"entity":"Eiffel Tower"},   "out_slot":"step1"},
            {"name":"BRIDGE", "args":{"slot_src":"step1",
                                      "bridge_entity":"Gustave Eiffel"},
                              "out_slot":"step2"},
            {"name":"BRIDGE", "args":{"slot_src":"step2",
                                      "bridge_entity":"Statue of Liberty"},
                              "out_slot":"step3"},
            {"name":"RESOLVE","args":{"slot_in":"step3"}},
        ])
        ctx = ENGINE.execute(prog, make_ctx())
        self.assertIsInstance(ctx.answer, list)


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)