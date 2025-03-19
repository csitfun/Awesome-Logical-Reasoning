"""Add year month to papers without them."""
import json

with open("src/metadata.json", "r", encoding="utf-8") as f:
    papers = json.load(f)

for paper in papers:
    if "year" not in paper:
        paper["year"] = ""
    if "month" not in paper:
        paper["month"] = ""

with open("src/metadata.json", "w", encoding="utf-8") as f:
    json.dump(papers, f, ensure_ascii=False, indent=4)
