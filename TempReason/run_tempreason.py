import json
import re

MONTHS = {
    "jan":1,"feb":2,"mar":3,"apr":4,
    "may":5,"jun":6,"jul":7,"aug":8,
    "sep":9,"oct":10,"nov":11,"dec":12
}
REV = {v:k for k,v in MONTHS.items()}


def parse_time_expr(text):
    text = text.strip().lower()

    # --- RECURSIVE CASE FIRST ---
    m = re.search(
        r"(\d+)\s+year[s]?\s+and\s+(\d+)\s+month[s]?\s+(before|after)\s+(.*)",
        text
    )
    if m:
        years = int(m.group(1))
        months = int(m.group(2))
        direction = m.group(3)
        rest = m.group(4)

        base = parse_time_expr(rest)
        if base is None:
            return None

        delta = years * 12 + months
        return base - delta if direction == "before" else base + delta


    # --- fallback recursive ---
    m = re.search(
        r"(\d+)\s+(year|month)s?\s+(before|after)\s+(.*)",
        text
    )
    if m:
        val = int(m.group(1))
        unit = m.group(2)
        direction = m.group(3)
        rest = m.group(4)

        base = parse_time_expr(rest)
        if base is None:
            return None

        delta = val * (12 if unit == "year" else 1)
        return base - delta if direction == "before" else base + delta


    # --- BASE CASE LAST ---
    m = re.search(
        r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*,\s*(\d+)",
        text
    )
    if m:
        return int(m.group(2)) * 12 + MONTHS[m.group(1)]
    


def solve(q):
    q = q.lower()

    # --- NORMALIZE ---
    q = q.replace("what is the time", "")
    q = q.replace("what is", "")
    q = q.replace("the time", "")
    q = q.replace("time", "")
    q = " ".join(q.split())

    total = parse_time_expr(q)
    if total is None:
        return None

    year = total // 12
    month = total % 12
    if month == 0:
        month = 12
        year -= 1

    return f"{REV[month].title()}, {year}"


def run_l1_only(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))

    correct = 0
    total = 0

    for i, x in enumerate(data):
        q = x["question"]

        # chỉ giữ câu có arithmetic
        if not re.search(r"(year|month)", q.lower()):
            continue

        pred = solve(q)
        gt = x["text_answers"]["text"][0]

        if i < 5:
            print("Q:", q)
            print("PRED:", pred, "| GT:", gt)
            print("-"*50)

        total += 1
        if pred == gt:
            correct += 1

    print(path, "ACC:", correct/total, f"({correct}/{total})")


# ===== RUN =====
run_l1_only("test_l1.json")