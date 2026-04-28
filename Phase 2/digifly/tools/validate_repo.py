"""digifly.tools.validate_repo

Repo structure + API contract validator for Digifly.

Goals:
  - Catch drift in file/folder layout as you refactor.
  - Catch missing/renamed functions/modules early.
  - Provide a single command that prints a clear report and returns non-zero on failure.

Usage (from repo root):
  python -m digifly.tools.validate_repo
  python -m digifly.tools.validate_repo --manifest config/structure_manifest.yaml
  python -m digifly.tools.validate_repo --strict

You can also import and call validate_repo() in CI or notebooks.
"""

from __future__ import annotations

import argparse
import fnmatch
import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "PyYAML is required for digifly.tools.validate_repo. "
        "Install with: pip install pyyaml"
    ) from e


@dataclass
class Finding:
    level: str   # "ERROR" | "WARN" | "INFO"
    code: str
    message: str

    def __str__(self) -> str:
        return f"[{self.level}] {self.code}: {self.message}"


@dataclass
class Report:
    ok: bool
    findings: List[Finding]

    def summary_counts(self) -> Dict[str, int]:
        out = {"ERROR": 0, "WARN": 0, "INFO": 0}
        for f in self.findings:
            out[f.level] = out.get(f.level, 0) + 1
        return out

    def format(self) -> str:
        counts = self.summary_counts()
        lines: List[str] = []
        lines.append("=" * 80)
        lines.append("DIGIFLY REPO VALIDATION REPORT")
        lines.append("=" * 80)
        lines.append(f"OK: {self.ok}")
        lines.append(
            f"Counts: errors={counts.get('ERROR',0)}  warns={counts.get('WARN',0)}  info={counts.get('INFO',0)}"
        )
        lines.append("-" * 80)
        for f in self.findings:
            lines.append(str(f))
        lines.append("=" * 80)
        return "\n".join(lines)


def load_manifest(manifest_path: Path) -> Dict[str, Any]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Manifest must parse to a mapping/dict.")
    return data


def find_repo_root(start: Path, markers: List[str]) -> Optional[Path]:
    """Walk upward from `start` until all marker paths exist."""
    cur = start.resolve()
    for _ in range(50):
        if all((cur / m).exists() for m in markers):
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


def _check_paths(root: Path, items: List[Dict[str, Any]], level: str) -> List[Finding]:
    findings: List[Finding] = []
    for it in items:
        rel = it.get("path")
        typ = it.get("type", "file")
        if not rel or not isinstance(rel, str):
            findings.append(Finding("ERROR", "MANIFEST_PATH_INVALID", f"Invalid path entry: {it!r}"))
            continue
        p = root / rel
        if typ == "file":
            if not p.exists() or not p.is_file():
                findings.append(Finding(level, "MISSING_FILE", f"Missing file: {rel}"))
            else:
                findings.append(Finding("INFO", "FOUND_FILE", f"Found file: {rel}"))
        elif typ == "dir":
            if not p.exists() or not p.is_dir():
                findings.append(Finding(level, "MISSING_DIR", f"Missing directory: {rel}"))
            else:
                findings.append(Finding("INFO", "FOUND_DIR", f"Found directory: {rel}"))
        else:
            findings.append(Finding("ERROR", "MANIFEST_TYPE_INVALID", f"Unknown type '{typ}' for path: {rel}"))
    return findings


def _glob_count(root: Path, pattern: str) -> int:
    # Match against posix-like relative paths
    n = 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = str(p.relative_to(root)).replace(os.sep, "/")
        if fnmatch.fnmatch(rel, pattern):
            n += 1
    return n


def _check_globs(root: Path, rules: List[Dict[str, Any]]) -> List[Finding]:
    findings: List[Finding] = []
    for r in rules:
        pattern = r.get("glob")
        min_count = int(r.get("min_count", 0))
        if not pattern:
            findings.append(Finding("ERROR", "MANIFEST_GLOB_INVALID", f"Invalid glob rule: {r!r}"))
            continue
        n = _glob_count(root, str(pattern))
        if n < min_count:
            findings.append(Finding("ERROR", "GLOB_TOO_FEW", f"Glob '{pattern}' matched {n}, expected >= {min_count}"))
        else:
            findings.append(Finding("INFO", "GLOB_OK", f"Glob '{pattern}' matched {n} file(s)"))
    return findings


def _import_module(module: str) -> Tuple[Optional[Any], Optional[str]]:
    try:
        m = importlib.import_module(module)
        return m, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _check_python_contracts(contracts: List[Dict[str, Any]]) -> List[Finding]:
    findings: List[Finding] = []
    for c in contracts:
        module = c.get("module", "")
        if not module:
            findings.append(Finding("WARN", "CONTRACT_EMPTY_MODULE", f"Contract has empty module: {c.get('name','(unnamed)')}" ))
            continue

        m, err = _import_module(str(module))
        if m is None:
            findings.append(Finding("ERROR", "IMPORT_FAILED", f"Cannot import '{module}': {err}"))
            continue

        findings.append(Finding("INFO", "IMPORT_OK", f"Imported module '{module}'"))

        must_have = c.get("must_have", []) or []
        callable_names = set((c.get("callable", []) or []))

        for name in must_have:
            if not hasattr(m, name):
                findings.append(Finding("ERROR", "MISSING_ATTR", f"Module '{module}' missing attribute '{name}'"))
            else:
                findings.append(Finding("INFO", "FOUND_ATTR", f"Module '{module}' has '{name}'"))
                if name in callable_names:
                    obj = getattr(m, name)
                    if not callable(obj):
                        findings.append(Finding("ERROR", "NOT_CALLABLE", f"'{module}.{name}' exists but is not callable"))
                    else:
                        findings.append(Finding("INFO", "CALLABLE_OK", f"'{module}.{name}' is callable"))
    return findings


def validate_repo(
    manifest_path: Path | str = "config/structure_manifest.yaml",
    *,
    strict: bool = False,
    cwd: Optional[Path] = None,
) -> Report:
    """Validate a Digifly repo against a structure/API manifest."""
    cwd_path = (cwd or Path.cwd()).resolve()
    mp = Path(manifest_path)
    if not mp.is_absolute():
        mp = (cwd_path / mp).resolve()

    manifest = load_manifest(mp)
    markers = manifest.get("project_root_markers") or ["digifly/__init__.py"]
    if not isinstance(markers, list) or not all(isinstance(x, str) for x in markers):
        markers = ["digifly/__init__.py"]

    repo_root = find_repo_root(mp.parent, list(markers))
    findings: List[Finding] = []

    if repo_root is None:
        findings.append(Finding("ERROR", "ROOT_NOT_FOUND", f"Could not locate repo root using markers: {markers} starting from {mp.parent}"))
        return Report(ok=False, findings=findings)

    findings.append(Finding("INFO", "ROOT_FOUND", f"Repo root: {repo_root}"))

    required = manifest.get("required_paths") or []
    recommended = manifest.get("recommended_paths") or []
    globs = manifest.get("glob_rules") or []
    py_contracts = manifest.get("python_contracts") or []

    # Paths
    findings.extend(_check_paths(repo_root, list(required), "ERROR"))
    rec_level = "ERROR" if strict else "WARN"
    findings.extend(_check_paths(repo_root, list(recommended), rec_level))

    # Globs
    if globs:
        findings.extend(_check_globs(repo_root, list(globs)))

    # Python API contracts
    if py_contracts:
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        findings.extend(_check_python_contracts(list(py_contracts)))

    ok = not any(f.level == "ERROR" for f in findings)
    return Report(ok=ok, findings=findings)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Validate Digifly repo structure and API contracts.")
    p.add_argument("--manifest", default="config/structure_manifest.yaml", help="Path to structure manifest YAML")
    p.add_argument("--strict", action="store_true", help="Treat recommended paths as errors")
    args = p.parse_args(argv)

    rep = validate_repo(args.manifest, strict=args.strict)
    print(rep.format())
    return 0 if rep.ok else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
