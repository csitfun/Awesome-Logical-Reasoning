"""This script prints the markdown for the README.md file."""
import json
from typing import Any, Dict, List

import yaml


def load_structure(yaml_file: str) -> Dict[str, Dict[str, Any]]:
    """Load the document structure from the yaml file.
    
    Args:
        yaml_file       : file name of the yaml file.
    
    Returns:
        structure       : dictionary containing the paper structure.
    """
    with open(yaml_file, "r", encoding="utf-8") as f:
        structure = yaml.safe_load(f)

    # add a paper section to each hierarchy
    for h2 in structure:
        structure[h2]['papers'] = []
        for h3 in structure[h2]:
            if h3 == 'papers':
                continue
            structure[h2][h3]['papers'] = []
            for h4 in structure[h2][h3]:
                if h4 == 'papers':
                    continue
                structure[h2][h3][h4]['papers'] = []

    return structure

def load_papers(paper_file: str, is_arrange: bool) -> List[Dict[str, Any]]:
    """Load the papers from the metadata.
    
    Args:
        paper_file      : file name of the metadata file.
        is_arrange      : flag to indicate if the papers should be arranged.

    Returns:
        papers          : list of dictionaries containing the paper metadata.
    """
    with open(paper_file, "r", encoding="utf-8") as f:
        papers = json.load(f)

    # TODO: arrange papers
    if is_arrange:
        # generate paper id for new papers

        # check redundant papers

        # find missing metadata

        # write back
        pass

    return papers

def classify_papers(structure: Dict[str, Dict[str, Any]], papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Classify papers according to the yaml structure and store the appers in a list of dicts."""
    # initialize the structure
    structured_papers = structure

    # iterate over the papers
    for paper in papers:
        # assign section
        if "h4" in paper:
            h4 = paper["h4"]
        if "h3" in paper:
            h3 = paper["h3"]
        if "h2" in paper:
            h2 = paper["h2"]

        # append paper to the structure
        if "h3" not in paper:
            structured_papers[h2]['papers'].append(paper)
        elif "h4" not in paper:
            structured_papers[h2][h3]['papers'].append(paper)
        else:
            structured_papers[h2][h3][h4]['papers'].append(paper)

    # sort the papers according to year
    for h2 in structured_papers:
        structured_papers[h2]['papers'] = sorted(structured_papers[h2]['papers'], key=lambda x: x['year'], reverse=True)
        for h3 in structured_papers[h2]:
            if h3 == 'papers':
                continue
            structured_papers[h2][h3]['papers'] = sorted(structured_papers[h2][h3]['papers'], key=lambda x: x['year'], reverse=True)
            for h4 in structured_papers[h2][h3]:
                if h4 == 'papers':
                    continue
                structured_papers[h2][h3][h4]['papers'] = sorted(structured_papers[h2][h3][h4]['papers'], key=lambda x: x['year'], reverse=True)

    return structured_papers

def generate_markdown(structured_papers: List[Dict[str, Any]]):
    """Generate the markdown file and write it to the markdown file."""
    def _format_paper(paper: Dict[str, Any]) -> str:
        formatted_paper = f"- **{paper['title']}**\n\n\t"
        if 'paper' in paper:
            formatted_paper += f"[![](https://img.shields.io/badge/📄-Paper-orange)]({paper['paper']}) "
        if 'github' in paper:
            formatted_paper += f"[![](https://img.shields.io/badge/📦-Github-purple)]({paper['github']}) "
        if 'huggingface' in paper:
            formatted_paper += f"[![](https://img.shields.io/badge/🤗-HuggingFace-yellow)]({paper['huggingface']}) "
        if 'webpage' in paper:
            formatted_paper += f"[![](https://img.shields.io/badge/🌐-Webpage-blue)]({paper['webpage']}) "
        if 'year' in paper and paper['year'] != "":
            formatted_paper += f"{paper['year']}. "
        if 'month' in paper and paper['month'] != "":
            formatted_paper += f"{paper['month']}. "
        if 'description' in paper:
            formatted_paper += f"{paper['description']}"

        formatted_paper += "\n\n"
        return formatted_paper

    with open("README.md", "w", encoding="utf-8") as f:
        f.write("""# ✨Awesome-Logical-Reasoning  [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
Paper list for logical reasoning

-- A survey paper will be released soon.

**Key Words**: premise, conclusion, argument, reasoning, inference, proposition, forward chaining, backward chaining, critical thinking, syllogism

![](https://img.shields.io/github/last-commit/csitfun/Awesome-Logical-Reasoning) Maintained by [Hanmeng Liu](https://scholar.google.com/citations?user=vjmL_9UAAAAJ&hl=en), [Ruoxi Ning](https://ruoxining.github.io)

![](https://img.shields.io/badge/PRs-Welcome-red) Welcome to contribute!

![](https://github.com/csitfun/Awesome-Logical-Reasoning/blob/main/assets/Logical_Reasoning.png)
""")
        for h2 in structured_papers:
            f.write(f"## ✨ {h2}\n")
            for paper in structured_papers[h2]['papers']:
                f.write(_format_paper(paper))
            for h3 in structured_papers[h2]:
                if h3 == 'papers':
                    continue
                f.write(f"### ✨ {h3}\n")
                for paper in structured_papers[h2][h3]['papers']:
                    f.write(_format_paper(paper))
                for h4 in structured_papers[h2][h3]:
                    if h4 == 'papers':
                        continue
                    f.write(f"#### ✨ {h4}\n")
                    for paper in structured_papers[h2][h3][h4]['papers']:
                        f.write(_format_paper(paper))

def main():
    # load yaml structure
    yaml_file = "src/structure.yml"
    structure = load_structure(yaml_file)

    # load paper metadata
    paper_file = "src/metadata.json"
    papers = load_papers(paper_file, is_arrange=True)

    # classify papers
    structured_papers = classify_papers(structure, papers)

    # generate markdown
    generate_markdown(structured_papers)

if __name__ == "__main__":
    main()
