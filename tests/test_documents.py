import ast
import json
from pathlib import Path
import re

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def test_notebooks_are_valid_and_standalone():
    paths = sorted((ROOT / "notebooks").glob("*.ipynb"))
    assert len(paths) == 4
    for path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        nbformat.validate(nbformat.from_dict(raw))
        for cell in raw["cells"]:
            # Notebook editors may normalize custom metadata into standard nbformat
            # top-level IDs and notebook-level language_info on save.
            expected_language = "python" if cell["cell_type"] == "code" else "markdown"
            assert cell["metadata"].get("language", expected_language) == expected_language
            assert cell.get("id") or cell["metadata"].get("id")
            source = "".join(cell["source"])
            if cell["cell_type"] == "code":
                ast.parse(source)
            else:
                assert not re.search(r"\b(paper|manuscript)\b", source, re.I)


def test_manuscript_citations_and_placeholders():
    text = (ROOT / "manuscript" / "main.tex").read_text(encoding="utf-8")
    bibliography = set(re.findall(r"\\bibitem\{([^}]+)\}", text))
    citations = set()
    for block in re.findall(r"\\cite\w*\{([^}]+)\}", text):
        citations.update(block.split(","))
    assert len(bibliography) == 22
    assert citations == bibliography
    assert text.count("\\subsection{Experiment ") == 4
    for section in ("Results", "Discussion", "Conclusions"):
        assert re.search(r"\\section\{" + section + r"\}\s*\\placeholder\{", text)
