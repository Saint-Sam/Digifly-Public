from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union


_EXPORT_INDEX_FILENAME = ".phase2_export_index.json"
_EXPORT_INDEX_CACHE: Dict[str, Dict[int, Dict[str, Any]]] = {}


def _coerce_root(swc_root: Union[str, Path]) -> Path:
    return Path(swc_root).expanduser().resolve()


def _export_index_path(swc_root: Union[str, Path]) -> Path:
    return (_coerce_root(swc_root) / _EXPORT_INDEX_FILENAME).resolve()


def _extract_id_hint(path: Path) -> Optional[int]:
    parent = path.parent.name.strip()
    if parent.isdigit():
        return int(parent)
    stem = path.stem
    for pattern in (
        r"^(\d+)_",
        r"^(\d+)$",
        r".*?(\d{4,}).*",
    ):
        match = re.match(pattern, stem)
        if match:
            try:
                return int(match.group(1))
            except Exception:
                return None
    return None


def _score_syn_csv_candidate_for_nid(path: Path, nid: int) -> tuple[int, int]:
    name = path.name
    parent_is_nid = path.parent.name == str(int(nid))
    score = 9
    if name == f"{nid}_synapses_new.csv":
        score = 0 if parent_is_nid else 1
    elif name.endswith("_synapses_new.csv") and str(nid) in name:
        score = 2 if parent_is_nid else 3
    elif str(nid) in path.stem:
        score = 4
    return score, len(str(path))


def _score_swc_candidate_for_nid(path: Path, nid: int) -> tuple[int, int]:
    name = path.name
    stem = path.stem
    parent_is_nid = path.parent.name == str(int(nid))
    score = 99
    if name == f"{nid}_axodendro_with_synapses.swc":
        score = 0 if parent_is_nid else 1
    elif "with_synapses" in stem and str(nid) in stem:
        score = 2 if parent_is_nid else 3
    elif name == f"{nid}_healed_final.swc":
        score = 4 if parent_is_nid else 5
    elif name == f"{nid}_healed.swc":
        score = 6 if parent_is_nid else 7
    elif name == f"{nid}.swc":
        score = 8
    elif str(nid) in stem:
        score = 9
    return score, len(str(path))


def _read_meta_payload(meta_path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {"meta_path": str(meta_path), "meta_parse_error": True}
    if not isinstance(data, dict):
        return {"meta_path": str(meta_path), "meta_parse_error": True}
    out = {"meta_path": str(meta_path)}
    dataset = data.get("neuprint_dataset")
    if dataset not in (None, ""):
        out["neuprint_dataset"] = str(dataset)
    return out


def _iter_export_files(swc_root: Path) -> Iterable[Path]:
    by_id_root = (swc_root / "by_id").resolve()
    if by_id_root.exists() and by_id_root.is_dir():
        for child in by_id_root.iterdir():
            if not child.is_dir():
                continue
            for entry in child.rglob("*"):
                if entry.is_file():
                    yield entry.resolve()
        return
    for pattern in ("*_synapses_new.csv", "*.swc", "*_export_meta.json"):
        yield from (p.resolve() for p in swc_root.rglob(pattern))


def _build_export_index(swc_root: Union[str, Path]) -> Dict[int, Dict[str, Any]]:
    root = _coerce_root(swc_root)
    index: Dict[int, Dict[str, Any]] = {}
    for path in _iter_export_files(root):
        hint = _extract_id_hint(path)
        if hint is None:
            continue
        entry = index.setdefault(int(hint), {})
        name = path.name
        suffix = path.suffix.lower()
        if name.endswith("_synapses_new.csv"):
            prev = entry.get("syn_csv")
            if prev is None or _score_syn_csv_candidate_for_nid(path, hint) < _score_syn_csv_candidate_for_nid(Path(prev), hint):
                entry["syn_csv"] = str(path)
        elif suffix == ".swc":
            prev = entry.get("swc")
            if prev is None or _score_swc_candidate_for_nid(path, hint) < _score_swc_candidate_for_nid(Path(prev), hint):
                entry["swc"] = str(path)
        elif name.endswith("_export_meta.json"):
            entry.update(_read_meta_payload(path))
    return index


def _save_export_index_to_disk(swc_root: Union[str, Path], index: Dict[int, Dict[str, Any]]) -> None:
    root = _coerce_root(swc_root)
    payload = {
        "swc_root": str(root),
        "entries": {str(int(k)): v for k, v in sorted(index.items())},
    }
    _export_index_path(root).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_export_index_from_disk(swc_root: Union[str, Path]) -> Optional[Dict[int, Dict[str, Any]]]:
    path = _export_index_path(swc_root)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        return None
    out: Dict[int, Dict[str, Any]] = {}
    for raw_key, raw_value in entries.items():
        try:
            key = int(raw_key)
        except Exception:
            continue
        if isinstance(raw_value, dict):
            out[key] = dict(raw_value)
    return out


def invalidate_export_path_cache(swc_root: Union[str, Path]) -> None:
    _EXPORT_INDEX_CACHE.pop(str(_coerce_root(swc_root)), None)


def refresh_export_path_index(swc_root: Union[str, Path]) -> Dict[int, Dict[str, Any]]:
    root = _coerce_root(swc_root)
    index = _build_export_index(root)
    _EXPORT_INDEX_CACHE[str(root)] = index
    try:
        _save_export_index_to_disk(root, index)
    except Exception:
        pass
    print(f"[paths] indexed export tree ids={len(index)} root={root}")
    return index


def _get_export_index(swc_root: Union[str, Path]) -> Dict[int, Dict[str, Any]]:
    root = _coerce_root(swc_root)
    key = str(root)
    cached = _EXPORT_INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    loaded = _load_export_index_from_disk(root)
    if loaded is None:
        loaded = refresh_export_path_index(root)
    else:
        print(f"[paths] loaded export index ids={len(loaded)} root={root}")
    _EXPORT_INDEX_CACHE[key] = loaded
    return loaded


def _upsert_export_entry(swc_root: Union[str, Path], nid: int, **updates: Any) -> None:
    root = _coerce_root(swc_root)
    index = _get_export_index(root)
    entry = index.setdefault(int(nid), {})
    for key, value in updates.items():
        if value is None:
            continue
        entry[key] = value
    _EXPORT_INDEX_CACHE[str(root)] = index
    try:
        _save_export_index_to_disk(root, index)
    except Exception:
        pass


def export_index_entry(swc_root: Union[str, Path], nid: int) -> Dict[str, Any]:
    return dict(_get_export_index(swc_root).get(int(nid), {}))


def _targeted_syn_csv_search(swc_root: Path, nid: int) -> Optional[Path]:
    for pat in (
        f"**/{nid}/{nid}_synapses_new.csv",
        f"**/{nid}_synapses_new.csv",
        f"**/*{nid}*synapses_new*.csv",
    ):
        hits = list(swc_root.glob(pat))
        if hits:
            hits.sort(key=lambda p: _score_syn_csv_candidate_for_nid(p, int(nid)))
            return hits[0].resolve()
    return None


def _targeted_swc_search(swc_root: Path, nid: int) -> Optional[Path]:
    patterns = [
        f"**/{nid}/{nid}_axodendro_with_synapses.swc",
        f"**/{nid}_axodendro_with_synapses.swc",
        f"**/*{nid}*with_synapses*.swc",
        f"**/{nid}/{nid}_healed_final.swc",
        f"**/{nid}_healed_final.swc",
        f"**/{nid}/{nid}_healed.swc",
        f"**/{nid}_healed.swc",
        f"**/{nid}.swc",
        f"**/*{nid}*.swc",
    ]
    for pat in patterns:
        hits = list(swc_root.glob(pat))
        if hits:
            hits.sort(key=lambda p: _score_swc_candidate_for_nid(p, int(nid)))
            return hits[0].resolve()
    return None


def _syn_csv_path(swc_root: Union[str, Path], nid: int) -> Optional[Path]:
    root = _coerce_root(swc_root)
    entry = export_index_entry(root, int(nid))
    syn_csv = entry.get("syn_csv")
    if syn_csv:
        path = Path(str(syn_csv)).expanduser().resolve()
        if path.exists():
            return path
    fallback = _targeted_syn_csv_search(root, int(nid))
    if fallback is not None:
        _upsert_export_entry(root, int(nid), syn_csv=str(fallback))
        return fallback
    return None


def _edges_filename_variants(pre_id: int):
    return [
        f"edges_ego_{int(pre_id)}__rawsyn.csv",
        f"edges_ego_{int(pre_id)}_w≥6__rawsyn.csv",
        f"edges_ego_{int(pre_id)}_w>=6__rawsyn.csv",
    ]


def _find_swc(swc_root: str, nid: int) -> str:
    nid = int(nid)
    root = _coerce_root(swc_root)
    if root.is_file() and root.suffix.lower() == ".swc":
        if str(nid) in root.name:
            return str(root)
        root = root.parent.resolve()

    entry = export_index_entry(root, nid)
    swc_path = entry.get("swc")
    if swc_path:
        path = Path(str(swc_path)).expanduser().resolve()
        if path.exists():
            return str(path)

    fallback = _targeted_swc_search(root, nid)
    if fallback is not None:
        _upsert_export_entry(root, nid, swc=str(fallback))
        return str(fallback)
    raise FileNotFoundError(f"No SWC for id={nid} under {root}")
