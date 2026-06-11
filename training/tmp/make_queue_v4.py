import json

scores = json.load(
    open("/Volumes/main/Users/kennethstott/PycharmProjects/chonk/training/tmp/scores_v4.json")
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
flagged = [s for s in scores if s["verdict"] != "Accept"]
rejects = [s for s in flagged if s["verdict"] == "Reject"]
reviews = [s for s in flagged if s["verdict"] == "Review"]
tq = chr(96) * 3


def fmt(s):
    ex = examples.get(s["id"], {})
    codes_str = ", ".join(s["codes"]) if s["codes"] else "N/A"
    sc = "D1:{D1} D2:{D2} D3:{D3} D4:{D4} D5:{D5} D6:{D6}".format(**s)
    parts = [
        "## Example {} -- {}".format(s["id"], s["verdict"]),
        "",
        "**Failure categories:** " + codes_str,
        "**Scores:** " + sc,
        "**Judge notes:** " + s["notes"],
        "",
        "**Instruction:**",
        ex.get("instruction", ""),
        "",
        "**Input (Python):**",
        tq + "python",
        ex.get("input", ""),
        tq,
        "",
        "**Output (Go):**",
        tq + "go",
        ex.get("output", ""),
        tq,
        "",
        "---",
        "**Your decision:** [ ] Accept  [ ] Edit  [ ] Reject",
        "**Your note (for manifest delta):** ___",
        "",
        "---",
    ]
    return "\n".join(parts)


out = [
    "# Review Queue -- Cluster A: Code Transformation -- 2026-06-11 v4",
    "**Algorithm:** SFT",
    f"**Total flagged:** {len(flagged)} ({len(reviews)} Review, {len(rejects)} Reject)",
    "",
    "Instructions: Mark each example Accept, Edit, or Reject. Add a note for the manifest delta.",
    "",
    "---",
    "",
    "## REJECT",
    "",
]
for s in rejects:
    out.append(fmt(s))
out.append("")
out.append("## REVIEW")
out.append("")
for s in reviews:
    out.append(fmt(s))
out_path = (
    "/Volumes/main/Users/kennethstott/PycharmProjects/chonk/training/review_queue_A_20260611_v4.md"
)
with open(out_path, "w") as f:
    f.write("\n".join(out))
print(f"Written: {out_path} ({len(flagged)} flagged)")
