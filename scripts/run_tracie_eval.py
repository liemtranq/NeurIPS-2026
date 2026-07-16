import json

from orchestrator import ANSBMasterOrchestrator, OrchestratorConfig

orch = ANSBMasterOrchestrator(OrchestratorConfig(max_iterations=4))

data = json.load(open("timebench_tracie_converted.json", encoding="utf-8"))["cases"]

correct = 0
fp = 0
fn = 0

for x in data:
    q = x["question"]
    label = int(x["label"])

    res = orch.execute(q)

    pred = 1 if any(
        item.get("trigger") is True
        and item.get("dominant_error") == "temporal"
        for item in res.get("trace_log", [])
    ) else 0

    if pred == label:
        correct += 1
    elif pred == 1:
        fp += 1
    else:
        fn += 1

print("\nRESULT")
print("=" * 50)
print("Accuracy:", correct / len(data), f"({correct}/{len(data)})")
print("FP:", fp)
print("FN:", fn)
