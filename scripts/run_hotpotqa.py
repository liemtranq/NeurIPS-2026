import re
import json
import string
from collections import Counter
from tqdm import tqdm
from datasets import load_dataset

from orchestrator import ANSBMasterOrchestrator, OrchestratorConfig

orch = ANSBMasterOrchestrator(OrchestratorConfig(max_iterations=4))


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return int(pred_tokens == gold_tokens)

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def build_context(example, max_docs=10):
    titles = example["context"]["title"]
    sents = example["context"]["sentences"]

    chunks = []
    for title, sentences in zip(titles, sents):
        text = " ".join(sentences)
        chunks.append(f"[{title}] {text}")

    return "\n".join(chunks[:max_docs])


def run_system(context, question):
    prompt = f"""
You are answering a HotpotQA multi-hop question.

Use the context only. Find the answer by connecting evidence across paragraphs.

Context:
{context}

Question:
{question}

Return only the shortest answer span. Do not explain.
"""

    res = orch.execute(prompt)

    if isinstance(res, dict):
        for k in ["final_answer", "answer", "output"]:
            if res.get(k):
                return str(res[k])

        # fallback for your current orchestrator
        proof = str(res.get("final_proof", ""))
        return proof

    return str(res)


def run(split="validation[:500]", output="hotpotqa_distractor_results.json"):
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)

    rows = []
    em_total = 0.0
    f1_total = 0.0

    for i, x in enumerate(tqdm(ds)):
        question = x["question"]
        gold = x["answer"]
        context = build_context(x)

        pred = run_system(context, question)

        em = exact_match_score(pred, gold)
        f1 = f1_score(pred, gold)

        em_total += float(em)
        f1_total += f1

        row = {
            "id": x.get("id", str(i)),
            "question": question,
            "answer": gold,
            "prediction": pred,
            "em": float(em),
            "f1": float(f1),
            "type": x.get("type"),
            "level": x.get("level"),
        }
        rows.append(row)

        if i < 5:
            print("\nQ:", question)
            print("PRED:", pred)
            print("GT:", gold)
            print("EM:", em, "F1:", f1)
            print("-" * 80)

    n = len(ds)
    summary = {
        "split": split,
        "n": n,
        "em": em_total / n,
        "f1": f1_total / n,
        "results": rows,
    }

    with open(output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nRESULT")
    print("=" * 60)
    print(f"EM: {summary['em']:.4f}")
    print(f"F1: {summary['f1']:.4f}")
    print("Saved:", output)


print("SANITY CHECK")
print("EM test:", exact_match_score("yes", "yes"))
print("F1 test:", f1_score("yes", "yes"))
print("="*40)

if __name__ == "__main__":
    run()
