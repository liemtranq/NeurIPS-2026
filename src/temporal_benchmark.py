from importlib.machinery import SourceFileLoader
import json

brain = SourceFileLoader(
    "brain",
    "Master Orchestrator Pipeline (The Brain).py"
).load_module()

ANSBMasterOrchestrator = brain.ANSBMasterOrchestrator
OrchestratorConfig = brain.OrchestratorConfig

cases = [
    {
        "id": "eiffel_bad",
        "question": "If the Eiffel Tower was built in 1889, how could a photograph of it exist in 1885?",
        "label": 1,
    },
    {
        "id": "smartphone_bad",
        "question": "How could a smartphone exist in 1990 if it was invented in 2007?",
        "label": 1,
    },
    {
        "id": "company_bad",
        "question": "How could a company founded in 2004 have signed a contract in 1999?",
        "label": 1,
    },
    {
        "id": "film_bad",
        "question": "How could a movie released in 2015 win an award in 2010?",
        "label": 1,
    },
    {
        "id": "eiffel_ok",
        "question": "If the Eiffel Tower was built in 1889, could a photograph of it exist in 1890?",
        "label": 0,
    },
    {
        "id": "smartphone_ok",
        "question": "Could a smartphone invented in 2007 exist in 2010?",
        "label": 0,
    },
    {
        "id": "company_ok",
        "question": "Could a company founded in 2004 sign a contract in 2008?",
        "label": 0,
    },
    {
        "id": "film_ok",
        "question": "Could a movie released in 2015 win an award in 2016?",
        "label": 0,
    },
]

orch = ANSBMasterOrchestrator(OrchestratorConfig(max_iterations=4))

results = []

for case in cases:
    res = orch.execute(case["question"])
    pred = 1 if any(
        item.get("trigger") and item.get("dominant_error") == "temporal"
        for item in res["trace_log"]
    ) else 0

    results.append({
        "id": case["id"],
        "question": case["question"],
        "label": case["label"],
        "pred": pred,
        "correct": pred == case["label"],
        "best_node_id": res["best_node_id"],
        "global_score": res["global_score"],
        "quality_score": res["quality_score"],
        "trace_log": res["trace_log"],
    })

acc = sum(r["correct"] for r in results) / len(results)

print("=" * 60)
print("TEMPORAL BENCHMARK RESULT")
print("=" * 60)
print(f"Accuracy: {acc:.3f} ({sum(r['correct'] for r in results)}/{len(results)})")
for r in results:
    print(f"{r['id']}: label={r['label']} pred={r['pred']} correct={r['correct']}")

with open("temporal_benchmark_results.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "accuracy": acc,
            "n": len(results),
            "results": results,
        },
        f,
        ensure_ascii=False,
        indent=2,
        default=str,
    )

print("Saved to temporal_benchmark_results.json")