"""Package source and clean notebook copies for upload; never include result artifacts.

This changes only cloud_notebooks.zip. Notebook files on disk are not modified.
"""

from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def build_bundle(root: Path):
    destination = root / "cloud_notebooks.zip"
    sources = [root / "pyproject.toml", root / "requirements.txt", root / "README.md"]
    sources += sorted((root / "src" / "dynalign").glob("*.py"))
    sources += sorted((root / "notebooks").glob("*.ipynb"))
    manifest = {}
    with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
        for path in sources:
            relative = path.relative_to(root).as_posix()
            data = path.read_bytes()
            if path.suffix == ".ipynb":
                document = json.loads(data)
                for cell in document["cells"]:
                    cell.setdefault("metadata", {})["language"] = "python" if cell["cell_type"] == "code" else "markdown"
                    cell["metadata"]["id"] = cell["id"]
                    if cell["cell_type"] == "code":
                        cell["execution_count"], cell["outputs"] = None, []
                        cell["metadata"].pop("execution", None)
                # Do not publish local interpreter IDs or machine-specific editor metadata.
                document["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                        "language_info": {"name": "python"}}
                data = (json.dumps(document, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
            archive.writestr(relative, data)
            manifest[relative] = sha256(data).hexdigest()
        archive.writestr("source_manifest.json", json.dumps(manifest, indent=2))
    return destination


if __name__ == "__main__":
    result = build_bundle(Path(__file__).resolve().parents[1])
    print(f"Created source-only upload bundle: {result.name} ({result.stat().st_size:,} bytes)")
