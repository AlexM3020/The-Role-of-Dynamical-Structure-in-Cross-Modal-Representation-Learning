"""Exercise the shared notebook source-bootstrap fragment without network access."""

import json
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
END = "print('Source setup only: result saving remains disabled by default.')"


def bootstraps():
    sources = []
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        document = json.loads(path.read_text(encoding="utf-8"))
        code = "".join(document["cells"][1]["source"])
        assert END in code, path
        sources.append(code[:code.index(END) + len(END)])
    return sources


def test_all_notebooks_share_the_same_cloud_bootstrap():
    sources = bootstraps()
    assert len(sources) == 5
    assert all(source == sources[0] for source in sources)
    assert "REPO_REF = 'main'" in sources[0]
    assert "SOURCE_ARCHIVE = None" in sources[0]
    assert "INSTALL_DEPENDENCIES = False" in sources[0]


def test_fresh_cloud_clones_but_rerun_preserves_checkout(tmp_path):
    cloud = tmp_path / "content"
    cloud.mkdir()
    scratch = tmp_path / "notebook"
    scratch.mkdir()
    calls = []

    class Paths:
        def __new__(cls, value):
            if str(value) == "/content":
                return cloud
            if str(value) == "/kaggle/working":
                return tmp_path / "absent-kaggle"
            return Path(value)

        @staticmethod
        def cwd():
            return scratch

    def check_call(arguments):
        calls.append(arguments)
        if arguments[1] == "clone":
            source = Path(arguments[-1]) / "src" / "dynalign"
            source.mkdir(parents=True)
            (source / "revised.py").write_text("# test fixture\n", encoding="utf-8")
        return 0

    # Inject a fake cloud filesystem and Git; the notebook code itself is unchanged.
    source = bootstraps()[0]
    for statement in ("from pathlib import Path\n", "import sys\n", "import subprocess\n"):
        source = source.replace(statement, "")
    runtime = SimpleNamespace(path=[], modules={})
    git = SimpleNamespace(check_call=check_call, run=lambda *a, **k: SimpleNamespace(returncode=0, stdout="test-revision\n"))
    context = {"Path": Paths, "sys": runtime, "subprocess": git}
    exec(compile(source, "cloud-bootstrap", "exec"), context)
    assert context["ROOT"] == cloud / "dynalign_github"
    assert len(calls) == 3
    assert calls[0][1] == "clone"
    assert calls[1][3] == "fetch"
    assert "--detach" in calls[2]
    assert not list(cloud.glob("**/outputs"))
    marker = context["ROOT"] / "user-note.txt"
    marker.write_text("keep this change", encoding="utf-8")
    exec(compile(source, "cloud-bootstrap-rerun", "exec"), context)
    assert len(calls) == 3  # No fetch, reset, checkout, or installation on rerun.
    assert marker.read_text(encoding="utf-8") == "keep this change"
