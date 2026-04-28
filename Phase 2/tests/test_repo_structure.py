import pathlib
import sys

def test_repo_structure_manifest_validates_ok():
    """
    Run with: pytest -q
    This test is meant to run inside your Digifly repo.
    It will fail if structure drifts from config/structure_manifest.yaml.
    """
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from digifly.tools.validate_repo import validate_repo

    rep = validate_repo("config/structure_manifest.yaml", strict=False, cwd=repo_root)
    assert rep.ok, rep.format()
