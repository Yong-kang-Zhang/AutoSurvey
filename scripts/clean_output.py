import json
import re
import sys
from pathlib import Path


def clean_generated_survey(text: str) -> str:
    cleaned_lines = []
    in_mermaid_block = False

    for raw_line in text.splitlines():
        stripped = raw_line.strip()

        if stripped.startswith("```mermaid"):
            in_mermaid_block = True
            continue

        if in_mermaid_block:
            if stripped == "```":
                in_mermaid_block = False
            continue

        if stripped.startswith(">"):
            continue

        if re.match(r'^\*\(View\s+\d+:.*\)\*$', stripped):
            continue

        cleaned_lines.append(raw_line.rstrip())

    cleaned_text = '\n'.join(cleaned_lines)
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    return cleaned_text.strip() + '\n'


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: python scripts/clean_output.py <md_path> <json_path>")

    md_path = Path(sys.argv[1])
    json_path = Path(sys.argv[2])

    cleaned_md = clean_generated_survey(md_path.read_text(encoding="utf-8"))
    md_path.write_text(cleaned_md, encoding="utf-8")

    data = json.loads(json_path.read_text(encoding="utf-8"))
    if "survey" in data:
        data["survey"] = clean_generated_survey(data["survey"])
    json_path.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
