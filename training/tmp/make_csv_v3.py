import csv
import json

scores = json.load(
    open("/Volumes/main/Users/kennethstott/PycharmProjects/chonk/training/tmp/scores_v3.json")
)
lines = open(
    "/Volumes/main/Users/kennethstott/PycharmProjects/chonk"
    "/training/training_data_sft_A_python_go_20260611_v3.jsonl"
).readlines()
examples = {}
for i, line in enumerate(lines, 1):
    line = line.strip()
    if line:
        examples[i] = json.loads(line)
out_path = (
    "/Volumes/main/Users/kennethstott/PycharmProjects/chonk"
    "/training/annotated_dataset_A_20260611_v3.csv"
)
with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
    writer.writerow(
        ["id", "instruction", "input", "output", "judge_verdict", "judge_scores", "judge_notes"]
    )
    for s in scores:
        ex = examples.get(s["id"], {})
        sc = "D1:{D1} D2:{D2} D3:{D3} D4:{D4} D5:{D5} D6:{D6}".format(**s)
        codes = "|".join(s["codes"]) if s["codes"] else ""
        notes = s["notes"] if s["notes"] else ""
        combined = codes + " -- " + notes if codes and notes else codes or notes
        writer.writerow(
            [
                s["id"],
                ex.get("instruction", ""),
                ex.get("input", ""),
                ex.get("output", ""),
                s["verdict"],
                sc,
                combined,
            ]
        )
print(f"Written: {out_path} (500 rows)")
