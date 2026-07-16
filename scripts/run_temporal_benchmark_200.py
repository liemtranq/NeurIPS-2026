import json
from orchestrator import ANSBMasterOrchestrator, OrchestratorConfig
import re

def predict_temporal_from_question(q: str) -> int:

    text = q.lower()

    raw = re.findall(r"\b(?:\d{1,3}|1[0-9]{3}|20[0-9]{2}|21[0-9]{2})(?:s)?\b", text)

    years = [int(x.rstrip("s")) for x in raw]

    if len(years) < 2:

        if (
            text.startswith("how could")
            and ("still ongoing" in text or "under construction" in text)
            and ("exist in" in text or "photo" in text or "painting" in text)
        ):
            return 1
        return 0

    y1, y2 = years[0], years[1]

    # If origin is stated first: "If invented/founded/built in y1, how could ... in y2?"

    if text.startswith("if") and "how could" in text:

        origin_year, claim_year = y1, y2

        return int(claim_year < origin_year)

    # If claim is stated first: "How could ... in y1 if invented/founded/built in y2?"

    if text.startswith("how could"):
        origin_markers = [
            "founded", "established", "released", "published", "launched",
            "premiered", "completed", "built", "invented", "opened",
            "incorporated", "discovered", "introduced", "created", "finished"
        ]

        prefix_before_y1 = text.split(str(y1))[0]
        if "completed" in prefix_before_y1 and ("photo" in prefix_before_y1 or "painting" in prefix_before_y1):
            y1_is_origin = False
        else:
            y1_is_origin = any(marker in prefix_before_y1 for marker in origin_markers)

        if y1_is_origin:
            origin_year, claim_year = y1, y2
        else:
            claim_year, origin_year = y1, y2

        return int(claim_year < origin_year)

    # OK-form: "Could ... in y1 if it was invented/founded/built in y2?"

    if text.startswith("could") and " if " in text:

        claim_year, origin_year = y1, y2

        return int(claim_year < origin_year)

    return 0

# load dataset
with open("temporal_benchmark_200.json", "r") as f:
    data = json.load(f)

cases = data["cases"]

orch = ANSBMasterOrchestrator(OrchestratorConfig(max_iterations=4))

correct = 0
results = []

for case in cases:
    q = case["question"]
    label = case["label"]

    res = orch.execute(q)

    # prediction rule
    pred = 1 if any(
        item.get("trigger") is True and item.get("dominant_error") == "temporal"
        for item in res["trace_log"]
    ) else 0

    is_correct = (pred == label)
    if is_correct:
        correct += 1

    results.append({
        "id": case["id"],
        "question": q,   # thêm dòng này
        "label": label,
        "pred": pred,
        "correct": is_correct,
        "score": res["global_score"]
    })

acc = correct / len(cases)
fp = [r for r in results if r["label"] == 0 and r["pred"] == 1]
fn = [r for r in results if r["label"] == 1 and r["pred"] == 0]

print("False positives:", len(fp))
print("False negatives:", len(fn))

print("\nSample FP:")
for r in fp[:10]:
    print(r["id"], "| score:", r["score"])
    print(r["question"])

print("\nSample FN:")
for r in fn[:20]:
    print(r["id"], "| score:", r["score"])
    print(r["question"])

print("="*60)
print("TEMPORAL BENCHMARK 200 RESULT")
print("="*60)
print(f"Accuracy: {acc:.3f} ({correct}/{len(cases)})")

# save
with open("temporal_benchmark_200_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "accuracy": acc,
        "total": len(cases),
        "correct": correct,
        "false_positives": len(fp),
        "false_negatives": len(fn),
        "results": results
    }, f, ensure_ascii=False, indent=2)