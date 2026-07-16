import json
import sys
from orchestrator import ANSBMasterOrchestrator, OrchestratorConfig


def predict_from_trace(res):
    return 1 if any(
        item.get("trigger") is True
        and item.get("dominant_error") == "temporal"
        for item in res.get("trace_log", [])
    ) else 0


def run_benchmark(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cases = data["cases"] if isinstance(data, dict) and "cases" in data else data

    orch = ANSBMasterOrchestrator(OrchestratorConfig(max_iterations=4))

    results = []
    correct = 0

    for i, case in enumerate(cases, 1):
        q = case["question"]
        label = int(case["label"])

        res = orch.execute(q)
        pred = predict_from_trace(res)

        ok = pred == label
        correct += int(ok)

        results.append({
            "id": case.get("id", f"case_{i:05d}"),
            "question": q,
            "label": label,
            "pred": pred,
            "correct": ok,
            "global_score": res.get("global_score"),
            "quality_score": res.get("quality_score"),
            "best_node_id": res.get("best_node_id"),
            "trace_log": res.get("trace_log", []),
        })

        print(f"[{i}/{len(cases)}] {case.get('id', i)} label={label} pred={pred} ok={ok}")

    acc = correct / len(cases)

    fp = [r for r in results if r["label"] == 0 and r["pred"] == 1]
    fn = [r for r in results if r["label"] == 1 and r["pred"] == 0]

    summary = {
        "input": input_path,
        "n": len(cases),
        "accuracy": acc,
        "correct": correct,
        "false_positives": len(fp),
        "false_negatives": len(fn),
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print("=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Accuracy: {acc:.4f} ({correct}/{len(cases)})")
    print(f"FP: {len(fp)}")
    print(f"FN: {len(fn)}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 unified_benchmark_runner.py input.json output.json")
        raise SystemExit(1)

    run_benchmark(sys.argv[1], sys.argv[2])