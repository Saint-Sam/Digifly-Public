from __future__ import annotations

import os
import time
import json
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence
from datetime import datetime, timezone

import numpy as np
import pandas as pd


DEFAULT_NEUPRINT_SERVER = "https://neuprint.janelia.org"
DEFAULT_NEUPRINT_DATASET = "manc:v1.2.1"
DEFAULT_MALE_CNS_DATASET = "male-cns:v0.9"
DEFAULT_NEUPRINT_TOKEN_FILE = "~/.neuprint_token"
REPO_NEUPRINT_TOKEN_FILE = "Neuprint Token.txt"
PHASE1_ROOT = Path(__file__).resolve().parent
CANONICAL_SYNAPSE_COLUMNS = ["pre_id", "post_id", "x", "y", "z", "type"]
DEFAULT_PHASE1_EXPORT_WORKERS = 4
DEFAULT_PHASE1_PROGRESS_EVERY = 25
KNOWN_NEUPRINT_DATASET_VERSIONS = {
    "manc": ["v1.0", "v1.2.1", "v1.2.3"],
    "male-cns": ["v0.9"],
}
DATASET_FAMILY_ALIASES = {
    "manc": "manc",
    "male-cns": "male-cns",
    "male_cns": "male-cns",
    "malecns": "male-cns",
}


def phase1_path(path: str | Path) -> Path:
    """Return an absolute path, resolving relative paths under this Phase 1 folder."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (PHASE1_ROOT / candidate).resolve()


def _coerce_int_ids(values: Iterable[Any], *, name: str = "body ids") -> List[int]:
    out: List[int] = []
    seen: set[int] = set()
    for raw in values:
        val = int(raw)
        if val in seen:
            continue
        seen.add(val)
        out.append(val)
    if not out:
        raise ValueError(f"{name} must contain at least one body id")
    return out


def normalize_neuprint_dataset_family(dataset_family: str | None = None) -> str:
    text = str(dataset_family or os.environ.get("NEUPRINT_DATASET") or DEFAULT_NEUPRINT_DATASET).strip().lower()
    if not text:
        return "manc"
    if ":" in text:
        text = text.split(":", 1)[0].strip()
    return DATASET_FAMILY_ALIASES.get(text, text)


def known_neuprint_dataset_versions(dataset_family: str | None = None) -> List[str]:
    family = normalize_neuprint_dataset_family(dataset_family)
    return list(KNOWN_NEUPRINT_DATASET_VERSIONS.get(family, []))


def default_neuprint_dataset_version(dataset_family: str | None = None) -> str:
    family = normalize_neuprint_dataset_family(dataset_family)
    if family == "male-cns":
        return DEFAULT_MALE_CNS_DATASET.split(":", 1)[1]
    return DEFAULT_NEUPRINT_DATASET.split(":", 1)[1]


def build_neuprint_dataset_name(dataset_family: str | None, version: str | None = None) -> str:
    family = normalize_neuprint_dataset_family(dataset_family)
    version_text = str(version or "").strip()
    if not version_text:
        version_text = default_neuprint_dataset_version(family)
    if ":" in version_text:
        version_text = version_text.split(":", 1)[1].strip()
    if not version_text.lower().startswith("v"):
        version_text = f"v{version_text}"
    return f"{family}:{version_text}"


def resolve_neuprint_dataset_choice(dataset: str | None = None, version: str | None = None) -> str:
    text = str(dataset or "").strip()
    if not text:
        return build_neuprint_dataset_name("manc", version)
    if ":" in text:
        family, raw_version = text.split(":", 1)
        return build_neuprint_dataset_name(family, version or raw_version)
    family = normalize_neuprint_dataset_family(text)
    return build_neuprint_dataset_name(family, version)


def normalize_neuprint_dataset(dataset: str | None = None) -> str:
    return resolve_neuprint_dataset_choice(dataset or os.environ.get("NEUPRINT_DATASET"))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _candidate_token_file_paths(token_file: str | Path | None = None) -> List[Path]:
    repo_token_path = (Path(__file__).resolve().parent / REPO_NEUPRINT_TOKEN_FILE).resolve()
    raw_candidates: List[str | Path | None] = [
        token_file,
        os.environ.get("NEUPRINT_TOKEN_FILE"),
        DEFAULT_NEUPRINT_TOKEN_FILE,
        repo_token_path,
    ]
    out: List[Path] = []
    seen: set[Path] = set()
    for raw in raw_candidates:
        if raw is None:
            continue
        path = Path(str(raw)).expanduser().resolve()
        if path in seen:
            continue
        seen.add(path)
        out.append(path)
    return out


def _read_token_from_file(token_file: str | Path | None = None) -> str:
    for path in _candidate_token_file_paths(token_file):
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8").strip()
        if text:
            return text
    return ""


def export_metadata_path(export_root: str | Path, body_id: int) -> Path:
    return (phase1_path(export_root) / "by_id" / str(int(body_id)) / f"{int(body_id)}_export_meta.json").resolve()


def _write_export_metadata(
    *,
    export_root: str | Path,
    body_id: int,
    dataset: str,
    server: str,
    upsample_nm: float,
    min_conf: float,
    batch_size: int,
) -> Path:
    meta_path = export_metadata_path(export_root, int(body_id))
    payload = {
        "body_id": int(body_id),
        "neuprint_dataset": normalize_neuprint_dataset(dataset),
        "neuprint_server": str(server).strip() or DEFAULT_NEUPRINT_SERVER,
        "upsample_nm": float(upsample_nm),
        "min_conf": float(min_conf),
        "batch_size": int(batch_size),
        "generated_at_utc": _utc_now_iso(),
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return meta_path


def _resolve_neuprint_connection_settings(
    *,
    server: str | None = None,
    dataset: str | None = None,
    token: str | None = None,
    token_file: str | Path | None = None,
    client=None,
) -> tuple[str, str, str]:
    client_server = getattr(client, "server", None)
    client_dataset = getattr(client, "dataset", None)
    client_token = getattr(client, "token", None)

    server_use = str(server or client_server or os.environ.get("NEUPRINT_SERVER") or DEFAULT_NEUPRINT_SERVER).strip()
    dataset_use = normalize_neuprint_dataset(dataset or client_dataset)
    token_use = str(token or client_token or os.environ.get("NEUPRINT_TOKEN") or _read_token_from_file(token_file) or "").strip()
    if not token_use:
        raise RuntimeError(
            "No neuPrint token found. Set NEUPRINT_TOKEN, set NEUPRINT_TOKEN_FILE, "
            f"or place the token in {Path(DEFAULT_NEUPRINT_TOKEN_FILE).expanduser()} "
            f"or the local ignored file {(Path(__file__).resolve().parent / REPO_NEUPRINT_TOKEN_FILE).resolve()}."
        )
    return server_use, dataset_use, token_use


def _load_neuprint_client(
    *,
    server: str | None = None,
    dataset: str | None = None,
    token: str | None = None,
    token_file: str | Path | None = None,
):
    try:
        from neuprint import Client
    except Exception as exc:  # pragma: no cover - import depends on user environment
        raise RuntimeError(
            "neuprint-python is not available in this environment. Install it in the Jupyter/kernel "
            "environment used for Phase 1/Phase 2, then retry the on-demand export."
        ) from exc

    server_use, dataset_use, token_use = _resolve_neuprint_connection_settings(
        server=server,
        dataset=dataset,
        token=token,
        token_file=token_file,
    )
    return Client(server_use, dataset=dataset_use, token=token_use)


def _make_client_getter(
    *,
    client=None,
    server: str | None = None,
    dataset: str | None = None,
    token: str | None = None,
    token_file: str | Path | None = None,
    workers: int = 1,
) -> Callable[[], Any]:
    workers_use = max(1, int(workers))
    if client is not None and workers_use <= 1:
        return lambda: client

    server_use, dataset_use, token_use = _resolve_neuprint_connection_settings(
        server=server,
        dataset=dataset,
        token=token,
        token_file=token_file,
        client=client,
    )
    local_state = threading.local()

    def get_client():
        cached = getattr(local_state, "client", None)
        if cached is None:
            local_state.client = _load_neuprint_client(
                server=server_use,
                dataset=dataset_use,
                token=token_use,
            )
            cached = local_state.client
        return cached

    return get_client


def _run_body_jobs(
    body_ids: Sequence[int],
    *,
    job: Callable[[int], Dict[str, Any]],
    workers: int,
    progress_label: str,
    progress_every: int,
) -> List[Dict[str, Any]]:
    ids = [int(x) for x in body_ids]
    total = len(ids)
    workers_use = max(1, int(workers))
    progress_every_use = max(1, int(progress_every))

    if total <= 1 or workers_use <= 1:
        results: List[Dict[str, Any]] = []
        for idx, body_id in enumerate(ids, start=1):
            results.append(job(int(body_id)))
            if idx % progress_every_use == 0 or idx == total:
                print(f"[phase1] {progress_label} {idx}/{total} done")
        return results

    ordered: List[Dict[str, Any] | None] = [None] * total
    completed = 0
    with ThreadPoolExecutor(max_workers=workers_use) as executor:
        future_map = {
            executor.submit(job, int(body_id)): idx
            for idx, body_id in enumerate(ids)
        }
        for future in as_completed(future_map):
            idx = future_map[future]
            ordered[idx] = future.result()
            completed += 1
            if completed % progress_every_use == 0 or completed == total:
                print(f"[phase1] {progress_label} {completed}/{total} done")
    return [row for row in ordered if row is not None]


def _fetch_healed_skeleton(client, body_id: int, upsample_nm: float) -> pd.DataFrame:
    try:
        from neuprint.skeleton import heal_skeleton, upsample_skeleton
    except Exception as exc:  # pragma: no cover - import depends on user environment
        raise RuntimeError(
            "neuprint.skeleton helpers are not available in this environment."
        ) from exc

    skel = client.fetch_skeleton(int(body_id), heal=False, format="pandas")
    if skel is None or len(skel) == 0:
        raise RuntimeError(f"empty skeleton for {int(body_id)}")
    skel = heal_skeleton(skel, max_distance=np.inf, root_parent=-1)
    skel = upsample_skeleton(skel, max_segment_length=float(upsample_nm))
    if skel is None or len(skel) == 0:
        raise RuntimeError(f"empty skeleton after heal/upsample for {int(body_id)}")
    return skel


def _to_ordered_swc_table(skel: pd.DataFrame) -> pd.DataFrame:
    required = {"rowId", "link", "x", "y", "z"}
    missing = required.difference(skel.columns)
    if missing:
        raise RuntimeError(f"skeleton missing required columns: {sorted(missing)}")

    work = skel.copy()
    if "radius" not in work.columns:
        work["radius"] = 0.01

    def parent_id(value: object) -> int:
        if pd.isna(value):
            return -1
        try:
            return int(value)
        except Exception:
            return -1

    row_ids = [int(rid) for rid in work["rowId"].tolist()]
    row_id_set = set(row_ids)
    parent_by_id: Dict[int, int] = {}
    children: Dict[int, List[int]] = defaultdict(list)
    for _, row in work.iterrows():
        rid = int(row["rowId"])
        parent = parent_id(row["link"])
        parent_by_id[rid] = parent
        if parent != rid:
            children[parent].append(rid)

    for key in children:
        children[key].sort()

    roots = [
        rid
        for rid in row_ids
        if parent_by_id.get(rid, -1) < 0
        or parent_by_id.get(rid) not in row_id_set
        or parent_by_id.get(rid) == rid
    ]

    ordered: List[int] = []
    seen: set[int] = set()

    def visit(start_id: int) -> None:
        stack = [int(start_id)]
        while stack:
            node_id = stack.pop()
            if node_id in seen:
                continue
            seen.add(node_id)
            ordered.append(node_id)
            for child_id in reversed(children.get(node_id, [])):
                if child_id not in seen:
                    stack.append(child_id)

    for root in sorted(set(roots)):
        visit(root)

    for rid in row_ids:
        if rid not in seen:
            visit(rid)

    work = work.set_index("rowId").loc[ordered].reset_index()
    work[["x", "y", "z", "radius"]] = work[["x", "y", "z", "radius"]].astype(float) / 1000.0
    bad_radius = ~np.isfinite(work["radius"]) | (work["radius"] <= 0)
    work.loc[bad_radius, "radius"] = 0.01
    work["swc_type"] = 3
    work["new_id"] = np.arange(1, len(work) + 1, dtype=int)
    id_map = dict(zip(work["rowId"].astype(int), work["new_id"].astype(int)))

    def map_parent(row: pd.Series) -> int:
        p = parent_id(row["link"])
        if p < 0 or p == int(row["rowId"]):
            return -1
        mapped_parent = id_map.get(p)
        if mapped_parent is None or int(mapped_parent) >= int(row["new_id"]):
            return -1
        return int(mapped_parent)

    work["new_parent"] = work.apply(map_parent, axis=1).astype(int)
    return work


def _write_swc(skel: pd.DataFrame, body_id: int, out_path: Path) -> None:
    table = _to_ordered_swc_table(skel)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# bodyId {int(body_id)}\n")
        for _, row in table.iterrows():
            handle.write(
                f"{int(row['new_id'])} {int(row['swc_type'])} "
                f"{row['x']:.3f} {row['y']:.3f} {row['z']:.3f} "
                f"{row['radius']:.3f} {int(row['new_parent'])}\n"
            )


def _canonicalize_synapse_rows(df: pd.DataFrame) -> pd.DataFrame:
    df_use = df.copy()
    for col in CANONICAL_SYNAPSE_COLUMNS:
        if col not in df_use.columns:
            df_use[col] = pd.NA
    df_use = df_use[CANONICAL_SYNAPSE_COLUMNS]
    if not df_use.empty:
        for col in ("pre_id", "post_id"):
            df_use[col] = pd.to_numeric(df_use[col], errors="coerce").astype("Int64")
        for col in ("x", "y", "z"):
            df_use[col] = pd.to_numeric(df_use[col], errors="coerce")
        df_use["type"] = df_use["type"].astype(str).str.lower()
        df_use = df_use[df_use["type"].isin({"pre", "post"})].copy()
    return df_use


def update_synapse_csvs_with_coords(
    *,
    base_out: str | Path,
    body_ids: Sequence[int],
    client=None,
    dataset: str | None = None,
    server: str | None = None,
    token: str | None = None,
    token_file: str | Path | None = None,
    min_conf: float = 0.4,
    batch_size: int = 10000,
    skip_existing: bool = True,
    workers: int = 1,
    progress_every: int = DEFAULT_PHASE1_PROGRESS_EVERY,
) -> Dict[str, Any]:
    base_out_use = phase1_path(base_out)
    ids = _coerce_int_ids(body_ids, name="body ids")
    get_client = _make_client_getter(
        client=client,
        server=server,
        dataset=dataset,
        token=token,
        token_file=token_file,
        workers=workers,
    )

    pre_cypher_tmpl = """
    MATCH (pre:Neuron {{bodyId: {bid}}})-[:Contains]->(:SynapseSet)-[:Contains]->(preSyn:Synapse),
          (preSyn)-[:SynapsesTo]->(postSyn:Synapse)<-[:Contains]-(:SynapseSet)<-[:Contains]-(post:Neuron)
    WHERE preSyn.type = 'pre'
      AND postSyn.type = 'post'
      AND preSyn.confidence  >= {min_conf}
      AND postSyn.confidence >= {min_conf}
    RETURN pre.bodyId AS pre_id,
           post.bodyId AS post_id,
           preSyn.location.x AS x,
           preSyn.location.y AS y,
           preSyn.location.z AS z,
           'pre' AS type
    SKIP {skip} LIMIT {limit}
    """

    post_cypher_tmpl = """
    MATCH (pre:Neuron)-[:Contains]->(:SynapseSet)-[:Contains]->(preSyn:Synapse),
          (preSyn)-[:SynapsesTo]->(postSyn:Synapse)<-[:Contains]-(:SynapseSet)<-[:Contains]-(post:Neuron {{bodyId: {bid}}})
    WHERE preSyn.type = 'pre'
      AND postSyn.type = 'post'
      AND preSyn.confidence  >= {min_conf}
      AND postSyn.confidence >= {min_conf}
    RETURN pre.bodyId AS pre_id,
           post.bodyId AS post_id,
           postSyn.location.x AS x,
           postSyn.location.y AS y,
           postSyn.location.z AS z,
           'post' AS type
    SKIP {skip} LIMIT {limit}
    """

    def fetch_side(body_id: int, tmpl: str, client_use) -> pd.DataFrame:
        chunks: List[pd.DataFrame] = []
        skip = 0
        while True:
            cypher = tmpl.format(
                bid=int(body_id),
                min_conf=float(min_conf),
                skip=int(skip),
                limit=int(batch_size),
            )
            df_chunk = client_use.fetch_custom(cypher)
            if df_chunk is None or df_chunk.empty:
                break
            chunks.append(df_chunk)
            if len(df_chunk) < int(batch_size):
                break
            skip += int(batch_size)
        if not chunks:
            return pd.DataFrame(columns=CANONICAL_SYNAPSE_COLUMNS)
        return _canonicalize_synapse_rows(pd.concat(chunks, ignore_index=True))

    def export_one(body_id: int) -> Dict[str, Any]:
        out_dir = (base_out_use / "by_id" / str(int(body_id))).resolve()
        csv_path = (out_dir / f"{int(body_id)}_synapses_new.csv").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        if skip_existing and csv_path.exists():
            try:
                probe = pd.read_csv(csv_path, nrows=5)
                if set(CANONICAL_SYNAPSE_COLUMNS).issubset(probe.columns):
                    return {
                        "body_id": int(body_id),
                        "status": "already_exists",
                        "csv_path": str(csv_path),
                        "row_count": int(pd.read_csv(csv_path).shape[0]),
                    }
            except Exception:
                pass

        t0 = time.perf_counter()
        client_use = get_client()
        df_pre = fetch_side(int(body_id), pre_cypher_tmpl, client_use)
        df_post = fetch_side(int(body_id), post_cypher_tmpl, client_use)
        df_all = _canonicalize_synapse_rows(pd.concat([df_pre, df_post], ignore_index=True))
        df_all.to_csv(csv_path, index=False)
        return {
            "body_id": int(body_id),
            "status": "exported",
            "csv_path": str(csv_path),
            "row_count": int(len(df_all)),
            "elapsed_s": float(time.perf_counter() - t0),
        }

    results = _run_body_jobs(
        ids,
        job=export_one,
        workers=workers,
        progress_label="synapse csv exports",
        progress_every=progress_every,
    )

    return {
        "base_out": str(base_out_use),
        "body_ids": [int(x) for x in ids],
        "results": results,
    }


def export_healed_swcs(
    *,
    body_ids: Sequence[int],
    export_root: str | Path,
    client=None,
    dataset: str | None = None,
    server: str | None = None,
    token: str | None = None,
    token_file: str | Path | None = None,
    upsample_nm: float = 2000.0,
    skip_existing: bool = True,
    workers: int = 1,
    progress_every: int = DEFAULT_PHASE1_PROGRESS_EVERY,
) -> Dict[str, Any]:
    export_root_use = phase1_path(export_root)
    ids = _coerce_int_ids(body_ids, name="body ids")
    get_client = _make_client_getter(
        client=client,
        server=server,
        dataset=dataset,
        token=token,
        token_file=token_file,
        workers=workers,
    )

    def export_one(body_id: int) -> Dict[str, Any]:
        out_dir = (export_root_use / "by_id" / str(int(body_id))).resolve()
        swc_path = (out_dir / f"{int(body_id)}_healed.swc").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        if skip_existing and swc_path.exists():
            return {"body_id": int(body_id), "status": "already_exists", "swc_path": str(swc_path)}

        t0 = time.perf_counter()
        client_use = get_client()
        skel = _fetch_healed_skeleton(client_use, int(body_id), upsample_nm=float(upsample_nm))
        _write_swc(skel, int(body_id), swc_path)
        return {
            "body_id": int(body_id),
            "status": "exported",
            "swc_path": str(swc_path),
            "elapsed_s": float(time.perf_counter() - t0),
        }

    rows = _run_body_jobs(
        ids,
        job=export_one,
        workers=workers,
        progress_label="swc exports",
        progress_every=progress_every,
    )

    return {
        "export_root": str(export_root_use),
        "body_ids": [int(x) for x in ids],
        "results": rows,
    }


def ensure_phase2_neuron_exports(
    *,
    body_ids: Sequence[int],
    export_root: str | Path,
    client=None,
    dataset: str | None = None,
    server: str | None = None,
    token: str | None = None,
    token_file: str | Path | None = None,
    upsample_nm: float = 2000.0,
    min_conf: float = 0.4,
    batch_size: int = 10000,
    skip_existing: bool = True,
    workers: int = 1,
    progress_every: int = DEFAULT_PHASE1_PROGRESS_EVERY,
) -> Dict[str, Any]:
    ids = _coerce_int_ids(body_ids, name="body ids")
    server_use, dataset_use, token_use = _resolve_neuprint_connection_settings(
        server=server,
        dataset=dataset,
        token=token,
        token_file=token_file,
        client=client,
    )
    swc_report = export_healed_swcs(
        body_ids=ids,
        export_root=export_root,
        client=client,
        dataset=dataset_use,
        server=server_use,
        token=token_use,
        token_file=token_file,
        upsample_nm=float(upsample_nm),
        skip_existing=bool(skip_existing),
        workers=int(workers),
        progress_every=int(progress_every),
    )
    syn_report = update_synapse_csvs_with_coords(
        base_out=export_root,
        body_ids=ids,
        client=client,
        dataset=dataset_use,
        server=server_use,
        token=token_use,
        token_file=token_file,
        min_conf=float(min_conf),
        batch_size=int(batch_size),
        skip_existing=bool(skip_existing),
        workers=int(workers),
        progress_every=int(progress_every),
    )
    meta_paths = []
    for body_id in ids:
        meta_paths.append(
            str(
                _write_export_metadata(
                    export_root=export_root,
                    body_id=int(body_id),
                    dataset=dataset_use,
                    server=server_use,
                    upsample_nm=float(upsample_nm),
                    min_conf=float(min_conf),
                    batch_size=int(batch_size),
                )
            )
        )
    return {
        "export_root": str(phase1_path(export_root)),
        "body_ids": [int(x) for x in ids],
        "neuprint_dataset": dataset_use,
        "neuprint_server": server_use,
        "metadata_paths": meta_paths,
        "swc_report": swc_report,
        "synapse_report": syn_report,
    }
