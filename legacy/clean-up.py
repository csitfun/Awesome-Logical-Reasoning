"""The utility functions to clean up the paper data."""
import json
import re
from typing import Dict, List


def parse_file(filecontent: str) -> List[Dict]:
    """Markdown to json."""
    metadata = []
    h2_current = None
    h3_current = None
    h4_current = None
    
    for line in filecontent.split("\n"):
        # empty line
        if len(line.strip()) == 0:
            continue
        # renew section
        if line.startswith("## "):
            h2_current = line[4:]   # including emoji
            h3_current = None
            h4_current = None
        elif line.startswith("### "):
            h3_current = line[5:]
            h4_current = None
        elif line.startswith("#### "):
            h4_current = line[6:]
        # paper
        else:
            is_paper = False
            paper = {}

            match = re.match(r"[0-9]+. \*\*(.+)\*\* ([A-Za-z]+) ([0-9]+)\. \[([a-zA-Z]+)\]\((.+)\)", line)
            if match:
                paper["title"] = match.group(1)
                paper["month"] = match.group(2)
                paper["year"] = match.group(3)
                paper[match.group(4)] = match.group(5)
                is_paper = True

            match = re.match(r"[0-9]+. \*\*(.+)\*\* ([A-Za-z]+) ([0-9]+)\. \[([a-zA-Z]+)\]\((.+)\) \[([a-zA-Z]+)\]\((.+)\)", line)
            if match:
                paper["title"] = match.group(1)
                paper["month"] = match.group(2)
                paper["year"] = match.group(3)
                paper[match.group(4)] = match.group(5)
                paper[match.group(6)] = match.group(7)
                is_paper = True

            match = re.match(r"[0-9]+. \*\*(.+)\*\* ([0-9]+)\. \[([a-zA-Z]+)\]\((.+)\)", line)
            if match:
                paper["title"] = match.group(1)
                paper["year"] = match.group(2)
                paper[match.group(3)] = match.group(4)
                is_paper = True

            match = re.match(r"[0-9]+. \*\*(.+)\*\* ([0-9]+) \[([a-zA-Z]+)\]\((.+)\)", line)
            if match:
                paper["title"] = match.group(1)
                paper["year"] = match.group(2)
                paper[match.group(3)] = match.group(4)
                is_paper = True

            match = re.match(r"[0-9]+. \*\*(.+)\*\* ([a-zA-Z]+) ([0-9]+) \[([a-zA-Z]+)\]\((.+)\)", line)
            if match:
                paper["title"] = match.group(1)
                paper["month"] = match.group(2)
                paper["year"] = match.group(3)
                paper[match.group(4)] = match.group(5)
                is_paper = True
            
            match = re.match(r"[0-9]+. \*\*(.+)\*\* \[([a-zA-Z]+)\]\((.+)\)", line)
            if match:
                paper["title"] = match.group(1)
                paper[match.group(2)] = match.group(3)
                is_paper = True

            match = re.match(r"[0-9]+. \*\*(.+)\*\* ([A-Za-z]+)\. ([0-9]+)\. \[([a-zA-Z]+)\]\((.+)\)", line)
            if match:
                paper["title"] = match.group(1)
                paper["month"] = match.group(2)
                paper["year"] = match.group(3)
                paper[match.group(4)] = match.group(5)
                is_paper = True

            match = re.match(r"^[0-9]+. \[([A-Za-z0-9\s\-]+)\]\((.+)\)$", line)
            if match:
                paper["title"] = match.group(1)
                paper["huggingface"] = match.group(2)
                is_paper = True

            match = re.match(r"^[0-9]+. \[([A-Za-z0-9\s\-]+)\]\((.+)\): (.+)", line)
            if match:
                paper["title"] = match.group(1)
                paper["github"] = match.group(2)
                paper["description"] = match.group(3)
                is_paper = True

            match = re.match(r"^[0-9]+. ([A-Z]+) \[[A-Za-z0-9\s\-]+\]\((.+)\)", line)
            if match:
                paper["title"] = match.group(1)
                paper["paper"] = match.group(2)
                is_paper = True
            
            match = re.match(r"^[0-9]+. \*\*(.+)\*\* ([0-9]+).$", line)
            if match:
                paper["title"] = match.group(1)
                paper["year"] = match.group(2)
                is_paper = True

            if h2_current:
                paper["h2"] = h2_current
            if h3_current:
                paper["h3"] = h3_current
            if h4_current:
                paper["h4"] = h4_current

            if is_paper:
                metadata.append(paper)
    return metadata

def pdf2intro(metadata: List[Dict]) -> List[Dict]:
    """Turn links with the pdf page into the intro page."""
    for i, paper in enumerate(metadata):
        if "paper" in paper:
            # arxiv
            pattern = re.compile(r"arxiv.org/pdf/([0-9]+\.[0-9]+)")
            replacement = r"arxiv.org/abs/\1"
            metadata[i]["paper"] = re.sub(pattern, replacement, paper["paper"])

            pattern = re.compile(r"arxiv.org/abs/([0-9]+\.[0-9]+).pdf")
            replacement = r"arxiv.org/abs/\1"
            metadata[i]["paper"] = re.sub(pattern, replacement, metadata[i]["paper"])

            # acl
            pattern = re.compile(r"aclanthology.org/([A-Za-z0-9\-\.]+)\.pdf")
            replacement = r"aclanthology.org/\1"
            metadata[i]["paper"] = re.sub(pattern, replacement, metadata[i]["paper"])

    return metadata


if __name__ == "__main__":
    # read
    filepath = "legacy/README.md"
    with open(filepath, "r") as f:
        filecontent = f.read()

    # convert
    metadata = parse_file(filecontent)
    metadata = pdf2intro(metadata)

    # write
    with open("scripts/metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)


