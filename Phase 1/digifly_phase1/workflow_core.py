"""Compatibility core for the Phase 1 notebook workflows.

This module keeps the proven Phase 1 notebook behavior importable while the
public notebook becomes a smaller launcher. The menu-facing entry points are:

- Choice 1: `option_20_build_and_map`
- Choice 3: `export_all_neuroncriteria_template`
- Choice 4: `run_pathfinding_option_26`
- Choice 5: `export_glia_volume_manc_v121`
- Choice 6: `report_dataset_label_coverage`
- Choice 7: `find_neurons_near_reference_skeletons`

Choice 2 lives in `filter_ids_by_size_and_export_swc.py` and shares the same
Phase 1 output-root rules through its caller.
"""

from __future__ import annotations

import neuprint as neu
from neuprint import Client, NeuronCriteria, fetch_neurons, fetch_synapses, fetch_simple_connections
import json, math, sys
from pathlib import Path
import pandas as pd
import navis.interfaces.neuprint as neu
from collections import defaultdict, deque
import networkx as nx
import numpy as np
import ipywidgets as widgets
import navis
import matplotlib.pyplot as plt
import re
from navis.interfaces.neuprint import fetch_skeletons
from matplotlib.backends.backend_pdf import PdfPages
from neuprint import NeuronCriteria
from scipy.stats import pearsonr, linregress
from scipy.spatial import cKDTree
import networkx as nx
from itertools import combinations, permutations
import networkx as nx
import numpy as np
import warnings
from IPython.display import display, clear_output
import ipywidgets as widgets
import neuprint as neu
import os, numpy as np, pandas as pd
from neuprint import NeuronCriteria, fetch_neurons, fetch_simple_connections, fetch_synapse_connections
from neuprint.skeleton import (
    fetch_skeleton, heal_skeleton,
    reorient_skeleton, upsample_skeleton,
    attach_synapses_to_skeleton,
    skeleton_segments
)





from neuprint import NeuronCriteria, fetch_neurons, fetch_synapse_connections
from neuprint.skeleton import heal_skeleton, upsample_skeleton

from .token_store import get_neuprint_token, phase1_root


# =============================================================================
# Shared Phase 1 state and output roots
# =============================================================================

PHASE1_ROOT = phase1_root()
PHASE1_OUTPUTS_ROOT = PHASE1_ROOT / "outputs"


def phase1_path(path: str | Path) -> Path:
    """Return an absolute path, resolving relative paths under `Phase 1`."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (PHASE1_ROOT / candidate).resolve()


def phase1_output_path(path: str | Path) -> Path:
    """Resolve a user-facing output path with Phase 1 as the relative root."""
    return phase1_path(path)

client = None
navis_client = None


def set_active_client(np_client):
    """Register the active neuPrint client for legacy helpers that use globals."""
    global client, navis_client
    client = np_client
    navis_client = np_client
    try:
        neu.set_default_client(np_client)
    except Exception:
        pass
    return np_client


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
_point_re = re.compile(r"Point\{[^}]*X=([\d\.\-]+),\s*Y=([\d\.\-]+),\s*Z=([\d\.\-]+)\}")
from scipy.stats import binomtest
import os
import pandas as pd
from neuprint import NeuronCriteria, fetch_neurons
import pandas as pd
import matplotlib.pyplot as plt
from neuprint import NeuronCriteria, fetch_simple_connections, fetch_neurons
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from neuprint import NeuronCriteria, fetch_neurons, fetch_simple_connections
from sklearn.mixture import GaussianMixture
import neuprint as neu
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.graph_objects as go
import trimesh
import numpy as np
from neuprint import NeuronCriteria


# =============================================================================
# Choice 1 configuration: SWC export, synapse CSVs, and synapse-mapped SWCs
# =============================================================================

UPSAMPLE_NM   = 2000
BASE_OUT      = str(phase1_output_path("export_swc"))
UNLABELED_EXPORT_ROOT = str(phase1_output_path("Glia IDs"))
SEED_PCT      = 99.0
GROW_PCT      = 95.0
BRANCH_MAX_LEN= 20
FALLBACK_MIN_SEEDS = 1
ALLOW_SIGN_FLIPS = False
MIN_SCALE = 1e-4; MAX_SCALE = 100.0
MIN_SCALE = 1e-4; MAX_SCALE = 100.0
PRE_PCTL   = 95
POST_PCTL  = 90
MIN_CUTOFF = 0.05
MAX_CUTOFF = 3.0
K_NEIGH      = 32
SNAP_TO_NODE = 0.0
MERGE_T_EPS  = 0.0
PRE_WINS     = False
LOCK_SOMA    = True
LOCK_TRUNK   = False
FORCE_ACCEPT_ALL = True
USE_NEUPRINT_SOMA = True
SOMA_SEED_MULT = 1.1   # nodes within somaRadius * this are marked soma
SOMA_GROW_MULT = 1.6   # optional: include neighbors out to this multiple
DN_SOMA_FALLBACK = True
DN_SOMA_ANCHOR   = "max_z"   # anchor = node with largest Z
DN_RAD_PCTL      = 99.5      # base radius ~ big neurite size
DN_RAD_MULT      = 2.0       # scale it up to soma-ish
DN_SEED_MULT     = 0.8       # seed ball radius = SEED_MULT * base_r
DN_GROW_MULT     = 1.6       # grow ball radius = GROW_MULT * base_r
DN_MIN_R_UM      = 0.4
DN_MAX_R_UM      = 8.0
DN_FORCE_TIP_CAP = True   # always add a terminal soma cap for DNs without somaLocation
MN_RAD_PCTL     = 99.7    # when neuprint radius is missing, base_r from large neurites
MN_RAD_MULT     = 2.2     # scale the pctl radius to “soma-ish”
MN_SEED_MULT    = 1.1     # seed ball = this × base soma radius
MN_GROW_MULT    = 1.6     # grow ball = this × base soma radius
MN_MIN_SEED_UM  = 0.01     # never seed smaller than this
MN_MIN_GROW_UM  = 0.01    # never grow smaller than this
MN_MAX_GROW_UM  = 30.0    # cap absurd growth
MN_GROW_RADIUS_GATE = 0.55   # only grow onto nodes with radius >= 0.55 * base_r
import os as _os  # at the top once


# Choice 1 helper functions. These preserve the notebook's original skeleton
# export and synapse-mapping behavior; the public menu calls the entry point
# `option_20_build_and_map()` below.
def _edges_ego_csv_path(bid, out_dir):
    # Exact filename pattern: edges_ego_<bid>__rawsyn.csv
    from pathlib import Path
    return Path(out_dir) / f"edges_ego_{int(bid)}__rawsyn.csv"
def _option20_outdir_for_bid(bid, base_out, client):
    """
    Decide where Option 20 should read/write data for a given bodyId.

    Priority:
      1) If an SWC for this bodyId already exists anywhere under base_out,
         use that folder (avoids creating 'No/None' or duplicate trees).
      2) Otherwise, query neuPrint for metadata and choose:
           - family:  AN / IN / DN / MN / SN / UNKNOWN
           - folder:  for MN → full instance if available, else type
                      for others → type (e.g. IN12B002); if missing, fall
                      back to instance prefix (before first '_'); if still
                      missing, use 'by_id'.
    """
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from neuprint import NeuronCriteria, fetch_neurons

    bid = int(bid)
    base_out = Path(base_out)

    # ------------------------------------------------------------------
    # 1) If this bodyId already has an SWC somewhere, use that folder
    # ------------------------------------------------------------------
    swc_patterns = [
        "**/*_healed_final.swc",
        "**/*_healed.swc",
        "**/*with_synapses*.swc",
        "**/*.swc",
    ]
    for pat in swc_patterns:
        for p in base_out.glob(pat):
            try:
                b = int(p.name.split("_", 1)[0])
            except Exception:
                continue
            if b == bid:
                # Reuse the existing canonical folder for this neuron.
                return p.parent

    # ------------------------------------------------------------------
    # 2) No existing SWC → infer location from neuPrint metadata
    # ------------------------------------------------------------------
    fam = "UNKNOWN"
    folder_name = "by_id"   # safe fallback if everything else is missing

    try:
        res = fetch_neurons(NeuronCriteria(bodyId=bid), client=client)
        df = res[0] if isinstance(res, tuple) else res
    except Exception:
        df = None

    if df is not None and not df.empty:
        row = df.iloc[0]

        inst  = row.get("instance")
        ntype = row.get("type")

        inst  = "" if (inst  is None or (isinstance(inst, float) and np.isnan(inst)))  else str(inst)
        ntype = "" if (ntype is None or (isinstance(ntype, float) and np.isnan(ntype))) else str(ntype)

        # ---- family: AN / IN / DN / MN / SN / UNKNOWN ----
        for prefix in ("AN", "IN", "DN", "MN", "SN"):
            if ntype.startswith(prefix) or inst.startswith(prefix):
                fam = prefix
                break

        # ---- folder within family ----
        if fam == "MN":
            # Motor neurons use the full instance when available, e.g. MNwm19_PDMNa_R.
            if inst:
                folder_name = inst
            elif ntype:
                folder_name = ntype
        else:
            # For AN / IN / DN / SN: use TYPE (IN12B002, DNxl048, etc.)
            if ntype:
                folder_name = ntype
            elif inst:
                # If only instance is present, strip suffixes after first '_'
                # e.g. "IN12B002_T1_R" → "IN12B002"
                folder_name = inst.split("_", 1)[0]

    # guardrails: never let fam become weird values like "No" or None
    valid_fams = {"AN", "IN", "DN", "MN", "SN", "UNKNOWN"}
    if fam not in valid_fams:
        fam = "UNKNOWN"
    if not folder_name:
        folder_name = "by_id"

    fam_dir = base_out / fam
    return fam_dir / folder_name / str(bid)
def _list_outgoing_partners(client, body_id, min_total_weight=1):
    """
    Return a list of bodyIds that the given body_id connects TO (outgoing partners).
    Uses a tiny custom Cypher that is fast and doesn’t time out.
    """
    cypher = (
        "MATCH (n:Neuron {bodyId: $bid})-[c:ConnectsTo]->(m:Neuron) "
        "WHERE c.weight >= $w "
        "RETURN m.bodyId AS bodyId_post"
    )
    try:
        df = client.fetch_custom(cypher, format='pandas')  # uses default dataset from client
        # Fallback for neuPrint clients that require params via string formatting.
        if isinstance(df, tuple):  # some versions return (df, msg)
            df = df[0]
        if df is None or df.empty:
            return []
        col = 'bodyId_post' if 'bodyId_post' in df.columns else list(df.columns)[0]
        return sorted(set(map(int, df[col].values)))
    except Exception:
        # Fallback: try string interpolation if params aren’t supported
        try:
            cy2 = (
                f"MATCH (n:Neuron {{bodyId: {int(body_id)}}})-[c:ConnectsTo]->(m:Neuron) "
                f"WHERE c.weight >= {int(min_total_weight)} "
                f"RETURN m.bodyId AS bodyId_post"
            )
            df2 = client.fetch_custom(cy2, format='pandas')
            if isinstance(df2, tuple):
                df2 = df2[0]
            if df2 is None or df2.empty:
                return []
            col = 'bodyId_post' if 'bodyId_post' in df2.columns else list(df2.columns)[0]
            return sorted(set(map(int, df2[col].values)))
        except Exception as e2:
            print(f"[TRACE] partner list fetch failed for {body_id}: {e2}")
            return []
def _ensure_dir(p):
    _os.makedirs(p, exist_ok=True); return p
import re
from collections import defaultdict
def _sanitize_label(name: str) -> str:
    # Make safe folder names; keep dots/underscores
    import re
    name = (name or "UNKNOWN").strip().replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9._\-]+", "", name) or "UNKNOWN"
def _instance_map_for_ids(client, ids: list[int]) -> dict[int, str]:
    """Return {bodyId: instance_label} for the given IDs (fallbacks if missing)."""
    from neuprint import fetch_neurons, NeuronCriteria
    inst = {}
    if not ids: 
        return inst
    df, _ = fetch_neurons(NeuronCriteria(bodyId=ids), client=client)
    if df is None or df.empty:
        return {int(b): f"by_id" for b in ids}
    # Prefer 'instance', then fall back to 'name'/'cellType'/'type'
    for _, r in df.iterrows():
        bid = int(r.get("bodyId"))
        label = None
        for col in ("instance", "name", "cellType", "type", "systematicType"):
            if col in df.columns and pd.notna(r.get(col)):
                label = str(r.get(col))
                break
        inst[bid] = _sanitize_label(label) if label else f"by_id"
    # Ensure all ids present
    for b in ids:
        inst.setdefault(int(b), "by_id")
    return inst
def _fetch_pre_by_partner_chunks(client, base_id: int, partners: list[int],
                                 chunk_size: int = 50, min_conf: float = 0.4):
    """
    Robustly fetch presynaptic (outgoing) synapse coordinates from base_id → partners.

    This is used inside _export_one_neuron() to gather all presynaptic sites of `base_id`
    that target any partner in `partners`. It chunks requests to avoid Cypher timeouts
    and returns a DataFrame with standardized columns:
        x, y, z, type='pre', bodyId_post

    Returns an empty DataFrame if none found (never raises on 404/timeout).
    """
    import pandas as pd
    from neuprint import fetch_synapse_connections, NeuronCriteria

    if not partners:
        return pd.DataFrame(columns=["x", "y", "z", "type", "bodyId_post"])

    all_rows = []
    for i in range(0, len(partners), chunk_size):
        sub = partners[i:i + chunk_size]
        try:
            df = fetch_synapse_connections(
                source_criteria=NeuronCriteria(bodyId=int(base_id)),
                target_criteria=NeuronCriteria(bodyId=sub),
                client=client
            )
        except Exception as e:
            print(f"[TRACE] fetch_synapse_connections failed chunk {i//chunk_size} for {base_id}: {e}")
            continue

        if df is None or df.empty:
            continue

        # Handle schema differences
        cols = list(df.columns)
        coord_candidates = [
            ("pre_x", "pre_y", "pre_z"),
            ("x_pre", "y_pre", "z_pre"),
            ("preX", "preY", "preZ")
        ]
        for cx, cy, cz in coord_candidates:
            if cx in cols and cy in cols and cz in cols:
                df = df.rename(columns={cx: "x", cy: "y", cz: "z"})
                break
        if "bodyId_post" not in df.columns:
            # sometimes "bodyId_post" is "post_bodyId"
            for alt in ("post_bodyId", "bodyId_postsyn", "post_id"):
                if alt in df.columns:
                    df["bodyId_post"] = df[alt]
                    break

        # Filter by confidence if available
        if "confidence" in df.columns:
            df = df[df["confidence"].astype(float) >= float(min_conf)]

        # Reduce columns
        keep = ["x", "y", "z", "bodyId_post"]
        df = df[[c for c in keep if c in df.columns]].copy()
        df["type"] = "pre"
        all_rows.append(df)

    if not all_rows:
        return pd.DataFrame(columns=["x", "y", "z", "type", "bodyId_post"])

    out = pd.concat(all_rows, ignore_index=True)
    return out
def _pick_type_col(df):
    for c in ["type","cellType","instance","class","name"]:
        if c in df.columns: return c
    raise RuntimeError("fetch_neurons() returned no recognizable type column")
def soma_check(client, body_id, baseline_swc_path):
    df, _ = fetch_neurons(NeuronCriteria(bodyId=[int(body_id)]), client=client)
    if df is None or df.empty:
        return {"status":"NA", "reason":"no metadata"}

    loc = df.iloc[0].get("somaLocation", None)
    rad = df.iloc[0].get("somaRadius", None)

    # parse somaLocation → µm
    soma_um = None
    if isinstance(loc, (list, tuple)) and len(loc) == 3:
        soma_um = np.asarray(loc, dtype=float) / 1000.0
    elif isinstance(loc, dict) and all(k in loc for k in ("x","y","z")):
        soma_um = np.array([loc["x"], loc["y"], loc["z"]], dtype=float) / 1000.0
    elif isinstance(loc, str):
        try:
            j = json.loads(loc)
            if isinstance(j, (list, tuple)) and len(j) == 3:
                soma_um = np.asarray(j, dtype=float) / 1000.0
            elif isinstance(j, dict) and all(k in j for k in ("x","y","z")):
                soma_um = np.array([j["x"], j["y"], j["z"]], dtype=float) / 1000.0
        except Exception:
            pass

    if soma_um is None:
        return {"status":"NA", "reason":"no somaLocation"}

    thr_um = max((float(rad)/1000.0 if pd.notna(rad) else 0.0) * 2.0, 1.0)

    _, recs = _parse_swc(baseline_swc_path)
    soma_xyz = np.array([[r[2], r[3], r[4]] for r in recs if r[1] == 1], float)
    all_xyz  = np.array([[r[2], r[3], r[4]] for r in recs], float)

    dmin = float(np.min(np.linalg.norm((soma_xyz if len(soma_xyz) else all_xyz) - soma_um, axis=1)))
    status = "OK" if (len(soma_xyz) and dmin <= thr_um) else ("FAIL" if len(soma_xyz) else "FAIL")
    return {"status":status, "n_soma":int(len(soma_xyz)), "d_um":dmin, "thr_um":thr_um}
def _list_types_by_prefix(prefix, client, pattern=None, extra_cols=("instance","name","cellType","systematicType")):
    """
    Returns {label: [bodyIds]} where *label* is the primary type column.
    If `pattern` is provided, it is used as a regex (case-insensitive) and
    applied as 'contains' rather than 'starts with'.
    A few extra label columns are searched for the same pattern.
    """
    df,_ = fetch_neurons(NeuronCriteria(status="Traced"), client=client)
    if df.empty:
        return {}

    tcol = _pick_type_col(df)
    cols = [tcol] + [c for c in extra_cols if c in df.columns]
    df = df[cols + ["bodyId"]].copy()

    # Build a single boolean mask over all available label columns
    if pattern is None:
        # legacy: startswith()
        mask = df[tcol].astype(str).str.upper().str.startswith(prefix.upper())
    else:
        rx = re.compile(pattern, flags=re.IGNORECASE)
        mlist = []
        for c in cols:
            mlist.append(df[c].astype(str).str.contains(rx, na=False))
        mask = np.logical_or.reduce(mlist)

    df = df[mask].dropna(subset=[tcol])
    out = defaultdict(list)
    for _, r in df.iterrows():
        out[str(r[tcol])].append(int(r["bodyId"]))
    return dict(out)
from neuprint import fetch_custom
MN_REGEX_DEFAULT = r"(?i)(MN|motor|motoneuron|MNfl)"  # or set to r"(?i)MN" to be strict
def _fetch_mn_candidates(client, regex=MN_REGEX_DEFAULT, limit=None):
    """
    Find MNs by (a) regex hits in label fields, (b) class contains 'motor',
    or (c) presence of Target. Returns a DataFrame with bodyId + labels.
    """
    label_fields = ["type","class","instance","name","cellType","systematicType","Target"]
    field_expr = ", ".join([f"n.{f} AS {f}" for f in label_fields])

    # Guard: escape single quotes so Cypher stays valid if regex contains them
    rx = str(regex).replace("'", "\\'")

    pred = f"""
      (n.type           =~ '{rx}' OR
       n.instance       =~ '{rx}' OR
       n.name           =~ '{rx}' OR
       n.cellType       =~ '{rx}' OR
       n.systematicType =~ '{rx}' OR
       n.class          =~ '(?i).*motor.*' OR
       exists(n.Target))
    """

    cypher = f"""
    MATCH (n:Neuron)
    WHERE n.status='Traced' AND {pred}
    RETURN n.bodyId AS bodyId, {field_expr}
    {"LIMIT "+str(int(limit)) if limit else ""}
    """

    # --- robust fetch: handle None and (df, msg) tuple cases ---
    res = fetch_custom(cypher, client=client)
    if isinstance(res, tuple):
        df = res[0]
    else:
        df = res

    if df is None:
        df = pd.DataFrame(columns=["bodyId"])

    if df.empty:
        return pd.DataFrame(columns=["bodyId"])

    df = df.drop_duplicates("bodyId").copy()

    # Clean empties → NaN (avoid 'nan' strings)
    for c in label_fields:
        if c in df.columns:
            df[c] = df[c].replace({"": np.nan, "None": np.nan, "none": np.nan})

    return df
import time
from pathlib import Path
NP_RETRIES = 4
NP_BACKOFF_SEC = 2.0   # exponential backoff base
try:
    from neuprint import fetch_synapse_connections, NeuronCriteria
except Exception:
    fetch_synapse_connections = None
    NeuronCriteria = None
def _export_one_neuron(body_id, out_dir, client, upsample_nm=UPSAMPLE_NM):
    _ensure_dir(out_dir)
    swc_path = _os.path.join(out_dir, f"{body_id}_healed.swc")
    syn_path = _os.path.join(out_dir, f"{body_id}_synapses.csv")

    if _os.path.exists(swc_path) and os.path.exists(syn_path):
        print(f"  • {body_id}: already exported")
        return swc_path, syn_path, True

    skel = client.fetch_skeleton(int(body_id), heal=False, format="pandas")
    skel = heal_skeleton(skel, max_distance=np.inf, root_parent=-1)
    skel = upsample_skeleton(skel, max_segment_length=upsample_nm)

    # Reorder parents before children without recursive DFS; large healed
    # skeletons can exceed Python's call stack.
    def _parent_id(value):
        if pd.isna(value):
            return -1
        try:
            return int(value)
        except Exception:
            return -1

    row_ids = [int(rid) for rid in skel["rowId"].tolist()]
    row_id_set = set(row_ids)
    parent_by_id = {}
    children = defaultdict(list)
    for _, r in skel.iterrows():
        rid = int(r["rowId"])
        parent = _parent_id(r["link"])
        parent_by_id[rid] = parent
        if parent != rid:
            children[parent].append(rid)
    for k in children:
        children[k].sort()

    roots = [
        rid for rid in row_ids
        if parent_by_id.get(rid, -1) < 0
        or parent_by_id.get(rid) not in row_id_set
        or parent_by_id.get(rid) == rid
    ]
    order = []
    seen = set()

    def visit(start_id):
        stack = [int(start_id)]
        while stack:
            rid = stack.pop()
            if rid in seen:
                continue
            seen.add(rid)
            order.append(rid)
            for child_id in reversed(children.get(rid, [])):
                if child_id not in seen:
                    stack.append(child_id)

    for root in sorted(set(roots)):
        visit(root)
    for rid in row_ids:
        if rid not in seen:
            visit(rid)

    skel = skel.set_index("rowId").loc[order].reset_index()

    # nm -> µm for SWC
    skel[["x","y","z","radius"]] = skel[["x","y","z","radius"]].astype(float)/1000.0
    skel.loc[skel["radius"]<=0,"radius"]=0.01
    skel["swc_type"]=3
    skel["new_id"]=np.arange(1, len(skel)+1)
    id_map = dict(zip(skel["rowId"], skel["new_id"]))

    def _new_parent(row):
        parent = _parent_id(row["link"])
        if parent < 0 or parent == int(row["rowId"]):
            return -1
        mapped_parent = id_map.get(parent)
        if mapped_parent is None or int(mapped_parent) >= int(row["new_id"]):
            return -1
        return int(mapped_parent)

    skel["new_parent"] = skel.apply(_new_parent, axis=1)
    with open(swc_path,"w") as f:
        f.write(f"# bodyId {body_id}\n")
        for _,r in skel.iterrows():
            f.write(f"{int(r['new_id'])} {int(r['swc_type'])} {r['x']:.3f} {r['y']:.3f} {r['z']:.3f} {r['radius']:.3f} {int(r['new_parent'])}\n")

    # synapses (raw nm)
    # --- SAFE synapse export: posts fast-path + pres via partner chunks ---
    from neuprint import NeuronCriteria, SynapseCriteria, fetch_synapse_connections
    
    # posts
    try:
        post_df = fetch_synapse_connections(
            target_criteria=NeuronCriteria(bodyId=[int(body_id)]),
            synapse_criteria=SynapseCriteria(confidence=0.4),
            client=client
        )
        if post_df is not None and not post_df.empty:
            syn_post = (post_df[['x_post','y_post','z_post']]
                        .rename(columns={'x_post':'x','y_post':'y','z_post':'z'}))
            syn_post['type'] = 'post'
        else:
            syn_post = pd.DataFrame(columns=['x','y','z','type'])
    except Exception as e:
        print(f"[TRACE] posts fetch failed for {body_id}: {e}")
        syn_post = pd.DataFrame(columns=['x','y','z','type'])
    
    # pres (chunk by partners to avoid giant custom-cypher)
    try:
        partners = _list_outgoing_partners(client, int(body_id), min_total_weight=1)
        if partners:
            syn_pre = _fetch_pre_by_partner_chunks(client, int(body_id), partners,
                                                   chunk_size=50, min_conf=0.4)
            if syn_pre is None:
                syn_pre = pd.DataFrame(columns=['x','y','z','type'])
        else:
            print(f"[TRACE] no outgoing partners for {body_id} (min_total_weight=1)")
            syn_pre = pd.DataFrame(columns=['x','y','z','type'])
    except Exception as e:
        print(f"[TRACE] pres partner-chunk fetch failed for {body_id}: {e}")
        syn_pre = pd.DataFrame(columns=['x','y','z','type'])
    
    # Merge and save. nm→µm conversion is handled later during mapping.
    syn = pd.concat([syn_post, syn_pre], ignore_index=True)
    syn.to_csv(syn_path, index=False)
    
    print(f"  • {body_id}: wrote SWC & synapses  (post={len(syn_post)}, pre={len(syn_pre)})")
    return swc_path, syn_path, False
def _parse_swc(path):
    header, recs = [], []
    with open(path) as f:
        for L in f:
            if not L.strip() or L.lstrip().startswith('#'):
                header.append(L); continue
            parts = L.split()
            if len(parts)<7:
                header.append(L); continue
            nid=int(float(parts[0])); ntype=int(float(parts[1]))
            x,y,z = map(float, parts[2:5])
            r = float(parts[5]); pid = int(float(parts[6]))
            recs.append([nid, ntype, x,y,z, r, pid, parts])
    return header, recs
def _bbox(arr): return np.min(arr,axis=0), np.max(arr,axis=0)
def _bbox_corners(lo,hi):
    xs=[lo[0],hi[0]]; ys=[lo[1],hi[1]]; zs=[lo[2],hi[2]]
    return np.array([[x,y,z] for x in xs for y in ys for z in zs], dtype=float)
def _best_align_params(syn_xyz, swc_xyz):
    syn_xyz = np.asarray(syn_xyz, dtype=float)
    swc_xyz = np.asarray(swc_xyz, dtype=float)

    if syn_xyz.size == 0 or swc_xyz.size == 0:
        return 1.0, (0, 1, 2), (1, 1, 1), np.zeros(3, dtype=float), float("nan")

    lo_s, hi_s = _bbox(syn_xyz); lo_w, hi_w = _bbox(swc_xyz)
    range_s = np.maximum(hi_s - lo_s, 1e-9); range_w = np.maximum(hi_w - lo_w, 1e-9)

    best = None
    for perm in permutations([0, 1, 2], 3):
        rs = range_s[list(perm)]
        flip_opts = ([(1, 1, 1)] if not ALLOW_SIGN_FLIPS
                     else [(sx, sy, sz) for sx in (1, -1) for sy in (1, -1) for sz in (1, -1)])
        for flips in flip_opts:
            s_axis = range_w / rs
            s = float(np.median(s_axis))
            if not np.isfinite(s):
                continue
            if not (MIN_SCALE <= s <= MAX_SCALE):
                continue
            syn_corners = _bbox_corners(lo_s[list(perm)], hi_s[list(perm)]) * np.array(flips) * s
            shift = lo_w - syn_corners.min(axis=0)
            syn_c_aligned = syn_corners + shift
            swc_corners = _bbox_corners(lo_w, hi_w)
            rmse = float(np.sqrt(np.mean((np.sort(syn_c_aligned, axis=0) - np.sort(swc_corners, axis=0)) ** 2)))
            key = (rmse, s, perm, flips, shift)
            if best is None or rmse < best[0]:
                best = key

    if best is None:
        valid = range_s > 1e-6
        if np.any(valid):
            axis_scales = range_w[valid] / range_s[valid]
            axis_scales = axis_scales[np.isfinite(axis_scales)]
            s = float(np.median(axis_scales)) if len(axis_scales) else 1.0
        else:
            s = 1.0
        s = float(np.clip(s, MIN_SCALE, MAX_SCALE))
        perm = (0, 1, 2)
        flips = (1, 1, 1)

        syn_corners = _bbox_corners(lo_s, hi_s) * s
        shift = lo_w - syn_corners.min(axis=0)
        syn_c_aligned = syn_corners + shift
        swc_corners = _bbox_corners(lo_w, hi_w)
        rmse = float(np.sqrt(np.mean((np.sort(syn_c_aligned, axis=0) - np.sort(swc_corners, axis=0)) ** 2)))
        print(f"[align] fallback used for degenerate synapse bbox (scale={s:.6g}, rmse={rmse:.6g})")
        return s, perm, flips, shift, rmse

    rmse, s, perm, flips, shift = best
    return s, perm, flips, shift, rmse
def _project_points_to_segments(points, A, B, mid_tree, seg_a_idx, seg_b_idx, k=K_NEIGH):
    if len(points)==0:
        return (np.array([],dtype=int),)*2 + (np.array([],dtype=float),)*2
    k_eff=min(k,len(A)); d0,idx0=mid_tree.query(points,k=k_eff)
    if k_eff==1: idx0=idx0[:,None]
    A_c=A[idx0]; B_c=B[idx0]; P=points[:,None,:]
    D=B_c-A_c; DD=np.sum(D*D,axis=2); DD=np.where(DD==0,1e-12,DD)
    t=np.sum((P-A_c)*D,axis=2)/DD; t_clip=np.clip(t,0.0,1.0)
    C=A_c+t_clip[...,None]*D
    dist=np.linalg.norm(P-C,axis=2)
    best_k=np.argmin(dist,axis=1)
    seg_idx=idx0[np.arange(len(points)),best_k]
    t_best=t_clip[np.arange(len(points)),best_k]
    d_best=dist[np.arange(len(points)),best_k]
    closest=C[np.arange(len(points)),best_k]
    return seg_idx,t_best,d_best,closest
def _write_baseline_swc(input_header, input_file, output_file, region_set):
    with open(output_file,"w") as fout, open(input_file) as fin:
        for L in input_header: fout.write(L)
        for L in fin:
            if not L.strip() or L.lstrip().startswith('#'): continue
            parts=L.split()
            if len(parts)<7: fout.write(L); continue
            nid=int(float(parts[0]))
            parts[1]='1' if nid in region_set else '2'
            fout.write(' '.join(parts)+'\n')
def _run_mapping_for(body_id, out_dir):
    INPUT_SWC    = _os.path.join(out_dir, f"{body_id}_healed.swc")
    SYN_CSV      = _os.path.join(out_dir, f"{body_id}_synapses_new.csv")
    LEGACY_SYN_CSV = _os.path.join(out_dir, f"{body_id}_synapses.csv")
    BASELINE_SWC = _os.path.join(out_dir, f"{body_id}_healed_final.swc")
    AUG_SWC      = _os.path.join(out_dir, f"{body_id}_axodendro_with_synapses.swc")
    MAP_CSV      = _os.path.join(out_dir, f"{body_id}_axodendro_with_synapses__synmap.csv")
    REPORT_JSON  = _os.path.join(out_dir, f"{body_id}_axodendro_with_synapses__report.json")
    CHECK_CSV   = _os.path.join(out_dir, f"{body_id}_synmap_node_label_check.csv")
    _bid_arg = body_id 
    # skip if already mapped
    if _os.path.exists(MAP_CSV) and _os.path.exists(AUG_SWC) and _os.path.exists(REPORT_JSON):
        df = pd.read_csv(MAP_CSV)
        acc = df[df["accepted"].astype(str).str.lower().isin(["true","t","1","yes","y"])]
        pre_tot = (df["syn_type"]=="pre").sum()
        post_tot= (df["syn_type"]=="post").sum()
        pre_acc = (acc["syn_type"]=="pre").sum()
        post_acc= (acc["syn_type"]=="post").sum()
        try:
            sc = soma_check(client, body_id, BASELINE_SWC)
        except Exception as _e:
            sc = {"status":"NA", "reason":f"error: {_e}"}
        return {"bid":body_id,"pre_rows":pre_tot,"post_rows":post_tot,
                "accepted_pre":pre_acc,"accepted_post":post_acc,
                "paths":{"map":MAP_CSV,"aug":AUG_SWC,"report":REPORT_JSON},
                "soma": sc}
        
    # Ensure mapping input exists. Older exports write <bid>_synapses.csv only.
    if not _os.path.exists(SYN_CSV):
        if _os.path.exists(LEGACY_SYN_CSV):
            try:
                syn_legacy = pd.read_csv(LEGACY_SYN_CSV)
                required = {"x", "y", "z", "type"}
                if not required.issubset(syn_legacy.columns):
                    raise ValueError(f"legacy synapse CSV missing columns {required}")
                syn_legacy = syn_legacy[["x", "y", "z", "type"]]
                syn_legacy.to_csv(SYN_CSV, index=False)
                print(f"  • {body_id}: created { _os.path.basename(SYN_CSV) } from legacy synapses.csv")
            except Exception as e:
                raise FileNotFoundError(
                    f"Could not prepare mapping synapse CSV for bodyId {body_id}: {e}"
                ) from e
        else:
            raise FileNotFoundError(
                f"Missing mapping synapse CSV: {SYN_CSV} (and no legacy file at {LEGACY_SYN_CSV})"
            )

    # 1) load swc
    header, recs = _parse_swc(INPUT_SWC)
    if not recs: raise RuntimeError(f"{INPUT_SWC}: no SWC records")
    nodes  = {r[0]: r for r in recs}
    order  = [r[0] for r in recs]
    coords = np.array([[r[2], r[3], r[4]] for r in recs], float)

    # adjacency
    adj = defaultdict(list)
    for nid, ntype, x, y, z, r, pid, parts in recs:
        if pid != -1 and pid in nodes:
            adj[nid].append(pid)
            adj[pid].append(nid)

    # ── Try NeuPrint somaLocation / somaRadius first (optional, safe fallback) ──
    soma_xyz_hint = None   # (µm, 3)
    soma_r_hint   = None   # (µm, float)

    # parse bodyId from header if present
    body_id = None
    for L in header:
        if "bodyId" in L:
            try:
                import re
                m = re.search(r'bodyId\s+(\d+)', L)
                if m: body_id = int(m.group(1))
            except Exception:
                pass
            break
    if body_id is None:
        body_id = _bid_arg
    if body_id is not None:
        try:
            # If a neuprint.Client named "client" exists, use it; else try env vars.
            from neuprint import Client, NeuronCriteria, fetch_neurons
            _client = globals().get("client", None)
            if _client is None:
                import os
                np_dataset = os.environ.get("NEUPRINT_DATASET")
                np_token   = get_neuprint_token(required=False)
                if np_dataset and np_token:
                    _client = Client("https://neuprint.janelia.org",
                                     dataset=np_dataset, token=np_token)
            if _client is not None:
                meta, _ = fetch_neurons(NeuronCriteria(bodyId=body_id), client=_client)
                if len(meta):
                    loc = meta.iloc[0].get("somaLocation", None)
                    rad = meta.iloc[0].get("somaRadius",  None)
                    # NeuPrint returns nm → convert to µm
                    if isinstance(loc, (list, tuple)) and len(loc) == 3:
                        soma_xyz_hint = np.asarray(loc, float) / 1000.0
                    if pd.notnull(rad):
                        soma_r_hint = float(rad) / 1000.0
        except Exception as _e:
            # Silent fallback: use percentile method below.
            pass

    # ── Build soma region: NeuPrint‐guided if possible, else percentile method ──

    radii  = np.array([nodes[n][5] for n in order], float)

    # Resolve neuron type once when a client/body_id is available.
    ntype = None
    try:
        _client_nt = globals().get("client", None)
        if _client_nt is None:
            import os as _os_tmp
            _ds = _os_tmp.environ.get("NEUPRINT_DATASET"); _tk = get_neuprint_token(required=False)
            if _ds and _tk:
                from neuprint import Client as _NPClient
                _client_nt = _NPClient("https://neuprint.janelia.org", dataset=_ds, token=_tk)
        if _client_nt is not None and body_id is not None:
            from neuprint import fetch_neurons, NeuronCriteria
            df_nt, _ = fetch_neurons(NeuronCriteria(bodyId=int(body_id)), client=_client_nt)
            if len(df_nt):
                for c in ("type","systematicType","class","instance"):
                    if c in df_nt.columns and pd.notna(df_nt.iloc[0][c]):
                        ntype = str(df_nt.iloc[0][c])
                        break
    except Exception:
        pass


    # ── Build soma region (MN-aware): NeuPrint‐guided if possible, else MN ball / percentile ──
    from collections import deque

    # Use the best-effort neuron type fetch to set the MN flag.
    ntu  = (ntype or "").upper()
    is_mn = ntu.startswith("MN") or ("MOTOR" in ntu)

    # helper: largest connected component among seed nodes, then neighbor + distance-ball grow
    def _grow_ball_region(seeds_set, in_ball_set):
        # LCC among seeds
        visited, comps = set(), []
        for s in seeds_set:
            if s in visited: 
                continue
            comp, Q = set(), deque([s])
            while Q:
                v = Q.popleft()
                if v in comp: 
                    continue
                comp.add(v); visited.add(v)
                for nb in adj[v]:
                    if nb in seeds_set and nb not in comp:
                        Q.append(nb)
            comps.append(comp)
        initial = max(comps, key=len) if comps else set(seeds_set)
        # include all 1-hop neighbors
        region_local = set(initial)
        for v in list(initial):
            for nb in adj[v]:
                region_local.add(nb)
        # BFS grow within in_ball
        Q = deque(region_local)
        while Q:
            v = Q.popleft()
            for nb in adj[v]:
                if nb not in region_local and nb in in_ball_set:
                    region_local.add(nb); Q.append(nb)
        return region_local

    if soma_xyz_hint is not None:
        # anchor around provided soma center
        # ---- MN-aware soma region (NeuPrint-guided) ----
        is_mn = (str(ntype).upper().startswith("MN") or "MOTOR" in str(ntype).upper())
        
        d2s = np.linalg.norm(coords - soma_xyz_hint[None, :], axis=1)
        seed_idx = int(np.argmin(d2s))
        seed_nid = order[seed_idx]
        
        radii = np.array([nodes[n][5] for n in order], float)
        # base_r: prefer neuprint soma radius; else robust neurite size for MN
        if is_mn:
            base_r = soma_r_hint if (soma_r_hint and soma_r_hint > 0) else (
                (np.percentile(radii, MN_RAD_PCTL) if len(radii) else 1.0) * MN_RAD_MULT
            )
            SEED_RAD = max(MN_SEED_MULT * base_r, MN_MIN_SEED_UM)
            GROW_RAD = min(max(MN_GROW_MULT * base_r, MN_MIN_GROW_UM), MN_MAX_GROW_UM)
            gate_r   = MN_GROW_RADIUS_GATE * base_r
        else:
            base_r  = (soma_r_hint if (soma_r_hint and soma_r_hint > 0)
                       else (np.percentile(radii, 99.0) if len(radii) else 1.0))
            SEED_RAD = max(SOMA_SEED_MULT * base_r, 0.5)
            GROW_RAD = max(SOMA_GROW_MULT * base_r, 1.0)
            gate_r   = 0.0  # no extra gating for non-MN
        
        # seeds: within SEED_RAD of soma center
        seeds = {order[i] for i, d in enumerate(d2s) if d <= SEED_RAD} or {seed_nid}
        
        # LCC of seeds
        visited, comps = set(), []
        Q = deque()
        for s in seeds:
            if s in visited: continue
            comp = set(); Q.clear(); Q.append(s)
            while Q:
                v = Q.popleft()
                if v in comp: continue
                comp.add(v); visited.add(v)
                for nb in adj[v]:
                    if nb in seeds and nb not in comp: Q.append(nb)
            comps.append(comp)
        initial = max(comps, key=len) if comps else set(seeds)
        
        # Start region
        region = set(initial)
        
        # Neighbor include (MN: only if neighbor is fat enough)
        for v in list(initial):
            for nb in adj[v]:
                if is_mn:
                    if nodes[nb][5] >= gate_r:
                        region.add(nb)
                else:
                    region.add(nb)
        
        # BFS grow but require BOTH: inside GROW_RAD AND (for MN) node radius ≥ gate_r
        in_ball = {order[i] for i, d in enumerate(d2s) if d <= GROW_RAD}
        Q = deque(region)
        while Q:
            v = Q.pop()
            for nb in adj[v]:
                if nb in region: 
                    continue
                if nb in in_ball and (not is_mn or nodes[nb][5] >= gate_r):
                    region.add(nb); Q.append(nb)


    elif ntu.startswith("DN"):
        # DN fallback (unchanged)
        idx_max_z = int(np.argmax(coords[:, 2]))
        anchor_nid = order[idx_max_z]
        anchor_xyz = coords[idx_max_z]
        d2a = np.linalg.norm(coords - anchor_xyz[None, :], axis=1)

        radii   = np.array([nodes[n][5] for n in order], float)
        base_r  = float(np.clip(np.percentile(radii, DN_RAD_PCTL) * DN_RAD_MULT,
                                DN_MIN_R_UM, DN_MAX_R_UM))
        seed_r  = DN_SEED_MULT * base_r
        grow_r  = DN_GROW_MULT * base_r

        seeds   = {order[i] for i, d in enumerate(d2a) if d <= seed_r} or {anchor_nid}
        in_ball = {order[i] for i, d in enumerate(d2a) if d <= grow_r}
        region  = _grow_ball_region(seeds, in_ball)

        print(f"  SOMA  : DN fallback @maxZ nid={anchor_nid} "
              f"(seed≈{seed_r:.2f}µm, grow≈{grow_r:.2f}µm, nodes={len(region)})")

    elif is_mn:
        # MN fallback with NO somaLocation → “fat-neurite ball” (better than percentile for MNs)
        radii   = np.array([nodes[n][5] for n in order], float)
        k       = max(10, int(0.002 * len(order)))
        idx_top = np.argsort(-radii)[:k] if len(radii) else []
        center  = coords[idx_top].mean(axis=0) if len(idx_top) else coords.mean(axis=0)
        d2c     = np.linalg.norm(coords - center[None, :], axis=1)

        base_r  = float(np.percentile(radii, MN_RAD_PCTL)) if len(radii) else MN_MIN_SEED_UM
        SEED_RAD = max(MN_SEED_MULT * base_r, MN_MIN_SEED_UM)
        GROW_RAD = float(np.clip(MN_GROW_MULT * base_r, MN_MIN_GROW_UM, MN_MAX_GROW_UM))

        seeds   = {order[i] for i, d in enumerate(d2c) if d <= SEED_RAD} or {order[int(np.argmin(d2c))]}
        in_ball = {order[i] for i, d in enumerate(d2c) if d <= GROW_RAD}
        region  = _grow_ball_region(seeds, in_ball)

        # small guard: expand once if still tiny
        if len(region) < 20:
            GROW_RAD = float(np.clip(GROW_RAD * 1.3, MN_MIN_GROW_UM, MN_MAX_GROW_UM))
            in_ball  = {order[i] for i, d in enumerate(d2c) if d <= GROW_RAD}
            region   = _grow_ball_region(seeds, in_ball)

    else:
        # Original percentile-based fallback (non-DN, non-MN)
        radii    = np.array([nodes[n][5] for n in order], float)
        seed_thr = np.percentile(radii, SEED_PCT) if len(radii) else 0.0
        grow_thr = np.percentile(radii, GROW_PCT) if len(radii) else 0.0
        seeds    = {nid for nid in nodes if nodes[nid][5] >= seed_thr}
        if not seeds:
            idx_sorted = np.argsort(-radii)
            seeds = {order[i] for i in idx_sorted[:min(FALLBACK_MIN_SEEDS, len(order))]}
        # LCC among seeds, add neighbors, then grow by radius threshold
        visited, comps = set(), []
        for s in seeds:
            if s in visited: 
                continue
            comp, Q = set(), deque([s])
            while Q:
                v = Q.popleft()
                if v in comp: 
                    continue
                comp.add(v); visited.add(v)
                for nb in adj[v]:
                    if nb in seeds and nb not in comp:
                        Q.append(nb)
            comps.append(comp)
        initial = max(comps, key=len) if comps else set(seeds)
        region  = set(initial)
        for v in list(initial):
            for nb in adj[v]:
                region.add(nb)
        Q = deque(region)
        while Q:
            v = Q.popleft()
            for nb in adj[v]:
                if nb not in region and nodes[nb][5] >= grow_thr:
                    region.add(nb); Q.append(nb)


    # ── Reabsorb tiny branches ──
    non_region = set(nodes) - region
    entry_pts  = [nb for v in region for nb in adj[v] if nb in non_region]
    seen = set(); branches = []
    for entry in entry_pts:
        if entry in seen: continue
        comp, Q = set(), deque([entry])
        while Q:
            v = Q.popleft()
            if v in comp: continue
            comp.add(v); seen.add(v)
            for nb in adj[v]:
                if nb in non_region and nb not in comp: Q.append(nb)
        branches.append(comp)
    for b in [b for b in branches if len(b) <= BRANCH_MAX_LEN]:
        region |= b

    _write_baseline_swc(header, INPUT_SWC, BASELINE_SWC, region)

    # 2) align synapses (raw nm -> swc µm)
    syn = pd.read_csv(SYN_CSV)
    need={'x','y','z','type'}
    if not need.issubset(syn.columns): raise ValueError(f"{SYN_CSV} missing {need}")
    raw_xyz = syn[['x','y','z']].to_numpy(float)
    syn_types = syn['type'].astype(str).str.lower().values

    finite_mask = np.isfinite(raw_xyz).all(axis=1)
    if not finite_mask.all():
        dropped = int((~finite_mask).sum())
        if dropped:
            print(f"[align] dropping {dropped} synapse row(s) with non-finite xyz")
        raw_xyz = raw_xyz[finite_mask]
        syn_types = syn_types[finite_mask]

    pre_mask = syn_types=='pre'; post_mask = syn_types=='post'

    if len(raw_xyz) == 0:
        s_scale, s_perm, s_flip, s_shift, rmse = 1.0, (0, 1, 2), (1, 1, 1), np.zeros(3, dtype=float), float("nan")
        aligned = raw_xyz.copy()
        print(f"[align] {body_id}: no valid synapse coordinates; writing empty mapping rows")
    else:
        s_scale, s_perm, s_flip, s_shift, rmse = _best_align_params(raw_xyz, coords)
        aligned = raw_xyz[:, s_perm] * np.array(s_flip) * s_scale + s_shift

    # 3) project to segments (from baseline)
    base_header, base_recs = _parse_swc(BASELINE_SWC)
    base_order = [r[0] for r in base_recs]
    base_coords = np.array([[r[2],r[3],r[4]] for r in base_recs], float)
    base_types  = np.array([r[1] for r in base_recs], int)
    base_parent = np.array([r[6] for r in base_recs], int)
    nid2idx = {nid:i for i,nid in enumerate(base_order)}
    seg_a_idx, seg_b_idx, seg_pairs = [], [], []
    for i,nid in enumerate(base_order):
        pid = base_parent[i]
        if pid == -1 or pid not in nid2idx: continue
        j = nid2idx[pid]
        seg_a_idx.append(i); seg_b_idx.append(j); seg_pairs.append((nid,pid))
    seg_a_idx = np.array(seg_a_idx,int); seg_b_idx=np.array(seg_b_idx,int)
    A=base_coords[seg_a_idx]; B=base_coords[seg_b_idx]
    mid=0.5*(A+B); mid_tree=cKDTree(mid)
    pre_pts  = aligned[pre_mask];  post_pts = aligned[post_mask]
    pre_seg, pre_t, pre_d, pre_c   = _project_points_to_segments(pre_pts,  A,B,mid_tree,seg_a_idx,seg_b_idx)
    post_seg,post_t,post_d,post_c  = _project_points_to_segments(post_pts, A,B,mid_tree,seg_a_idx,seg_b_idx)

    # cutoffs (still compute for report)
    def _robust_cutoff(d,pctl,floor=MIN_CUTOFF,ceil=MAX_CUTOFF):
        if len(d)==0: return ceil
        return float(np.clip(np.percentile(d,pctl), floor, ceil))
    pre_cut  = _robust_cutoff(pre_d,  PRE_PCTL)
    post_cut = _robust_cutoff(post_d, POST_PCTL)

    if FORCE_ACCEPT_ALL:
        pre_keep  = np.ones(len(pre_d), dtype=bool)
        post_keep = np.ones(len(post_d), dtype=bool)
    else:
        pre_keep  = pre_d  <= pre_cut
        post_keep = post_d <= post_cut

    # 4) assign to endpoints or insert
    node={}
    for r in base_recs:
        nid,t,x,y,z,r_,pid,parts=r
        node[nid]={"type":int(t),"x":float(x),"y":float(y),"z":float(z),"r":float(r_),"pid":int(pid),"parts":parts[:]}

    max_nid = max(node.keys()) if node else 0
    def _radius_interp(r1,r2,t): return (1-t)*r1 + t*r2

    plans=[]
    def _plan(seg_idx_arr,t_arr,d_arr,c_arr,keep,label):
        out=[]
        for i,keepi in enumerate(keep):
            if not keepi: continue
            sidx=int(seg_idx_arr[i]); t=float(t_arr[i]); cxyz=c_arr[i]
            i_child=seg_a_idx[sidx]; i_parent=seg_b_idx[sidx]
            cA=A[sidx]; cB=B[sidx]
            dA=np.linalg.norm(cxyz-cA); dB=np.linalg.norm(cxyz-cB)
            snap=None
            if dA<=SNAP_TO_NODE or t<=MERGE_T_EPS: snap=("child", i_child)
            elif dB<=SNAP_TO_NODE or (1.0-t)<=MERGE_T_EPS: snap=("parent", i_parent)
            out.append({"seg_idx":sidx,"t":t,"closest":cxyz,"snap":snap,"label":label})
        return out
    plans += _plan(pre_seg,  pre_t,  pre_d,  pre_c,  pre_keep,  "pre")
    plans += _plan(post_seg, post_t, post_d, post_c, post_keep, "post")

    insertions_per_seg=defaultdict(list)
    for P in plans:
        if P["snap"] is None:
            key_t = round(P["t"], 9)
            insertions_per_seg[(P["seg_idx"], key_t)].append(P)

    inserted_nodes={}
    seg_to_inserts_sorted=defaultdict(list)
    for (sidx,key_t), items in insertions_per_seg.items():
        labels={it["label"] for it in items}
        label=('pre' if ('pre' in labels and PRE_WINS) else
               ('post' if 'post' in labels else list(labels)[0]))
        t=float(np.mean([it["t"] for it in items]))
        cxyz=np.mean([it["closest"] for it in items],axis=0)
        seg_to_inserts_sorted[sidx].append({"key_t":key_t,"t":t,"label":label,"xyz":cxyz})
    for sidx in list(seg_to_inserts_sorted.keys()):
        seg_to_inserts_sorted[sidx].sort(key=lambda d:d["t"], reverse=True)

    new_nodes_count=0
    for sidx, arr in seg_to_inserts_sorted.items():
        child_idx=seg_a_idx[sidx]; parent_idx=seg_b_idx[sidx]
        child_nid=base_order[child_idx]; parent_nid=base_order[parent_idx]
        rA=node[child_nid]["r"]; rB=node[parent_nid]["r"]
        last_parent=parent_nid
        for item in arr:
            t=item["t"]; xyz=item["xyz"]; label=item["label"]
            max_nid+=1; new_nid=max_nid; new_nodes_count+=1
            rr=_radius_interp(rA,rB,t)
            node[new_nid]={"type":4 if label=='pre' else 5,
                           "x":float(xyz[0]),"y":float(xyz[1]),"z":float(xyz[2]),
                           "r":float(rr),"pid":int(last_parent),
                           "parts":[str(new_nid), str(4 if label=='pre' else 5),
                                    f"{xyz[0]:.6f}", f"{xyz[1]:.6f}", f"{xyz[2]:.6f}",
                                    f"{rr:.6f}", str(last_parent)]}
            inserted_nodes[(sidx, round(t,9))]=new_nid
            last_parent=new_nid
        if arr:
            node[child_nid]["pid"]=last_parent

    # endpoint snaps → mark labels (spawn children if both labels collide)
    endpoint_hits = defaultdict(set)
    for P in plans:
        if P["snap"] is not None:
            _, idx_ep = P["snap"]
            nid_ep = base_order[idx_ep]
            endpoint_hits[nid_ep].add(P["label"])
    
    def _tiny_offset_vec(nid):
        # make a tiny, stable visual offset so markers aren’t exactly on top of the endpoint
        # use parent direction if possible; else a small +Z nudge
        i = nid2idx.get(nid, None)
        if i is None:
            return np.array([0.0, 0.0, 0.2])
        pid = base_parent[i]
        if pid == -1 or pid not in nid2idx:
            return np.array([0.0, 0.0, 0.2])
        v = base_coords[i] - base_coords[nid2idx[pid]]
        n = np.linalg.norm(v)
        if n < 1e-6:
            return np.array([0.0, 0.0, 0.2])
        return (v / n) * 0.2  # 0.2 µm outward
    
    for nid, labels in endpoint_hits.items():
        # keep soma locked if requested
        if LOCK_SOMA and node[nid]["type"] == 1:
            continue
    
        labs = sorted(labels)  # e.g., ['post'] or ['pre','post']
        if labs == ['pre']:
            node[nid]["type"] = 4
        elif labs == ['post']:
            node[nid]["type"] = 5
        elif labs == ['post','pre']:
            # keep the endpoint’s original label (usually 2), and add two small marker children: one PRE and one POST
            base_xyz = np.array([node[nid]["x"], node[nid]["y"], node[nid]["z"]], float)
            base_r   = float(node[nid]["r"])
            dv       = _tiny_offset_vec(nid)
    
            for lab in ('pre','post'):
                new_t = 4 if lab == 'pre' else 5
                new_xyz = base_xyz + dv if lab == 'pre' else base_xyz - dv
                new_r   = base_r * 0.9
    
                max_nid += 1
                child_id = max_nid
                node[child_id] = {
                    "type": new_t,
                    "x": float(new_xyz[0]), "y": float(new_xyz[1]), "z": float(new_xyz[2]),
                    "r": float(new_r), "pid": int(nid),
                    "parts": [str(child_id), str(new_t),
                              f"{new_xyz[0]:.6f}", f"{new_xyz[1]:.6f}", f"{new_xyz[2]:.6f}",
                              f"{new_r:.6f}", str(nid)]
                }


    # --- DN tip-soma cap (only if no somaLocation AND it's a DN AND no type=1 yet) ---
    is_dn        = (str(ntype).upper().startswith("DN"))
    has_soma_meta= (soma_xyz_hint is not None)
    has_type1    = any(d["type"] == 1 for d in node.values())
    
    should_cap = is_dn and (not has_soma_meta) and (DN_FORCE_TIP_CAP or not has_type1)
    if should_cap:
        # leaves (zero children)
        child_count = {nid: 0 for nid in base_order}
        for pid in base_parent:
            if pid != -1 and pid in child_count:
                child_count[pid] += 1
        leaf_idxs = [i for i, nid in enumerate(base_order) if child_count[nid] == 0] or [int(np.argmax(base_coords[:,2]))]
    
        # rostral-most leaf
        tip_idx = max(leaf_idxs, key=lambda i: base_coords[i, 2])
        tip_nid = base_order[tip_idx]
        tip_xyz = base_coords[tip_idx]
    
        # stable outward direction (grandparent→tip; PCA fallback)
        pid   = base_parent[tip_idx]
        gpid  = base_parent[nid2idx[pid]] if (pid in nid2idx and base_parent[nid2idx[pid]] in nid2idx) else None
        if pid in nid2idx:
            v1 = tip_xyz - base_coords[nid2idx[pid]]
        else:
            v1 = np.array([0.0, 0.0, 1.0])
        if gpid is not None:
            v2 = tip_xyz - base_coords[nid2idx[gpid]]
            vec = v1 + 0.5*v2
        else:
            vec = v1
        nrm = np.linalg.norm(vec) or 1.0
        dirv = vec / nrm
        if dirv[2] < 0:  # push outward along +Z-ish if ambiguous
            dirv = -dirv
    
        # radius guess from fat neurite, then clamp by clearance to other nodes
        radii_arr = np.array([node[n]["r"] for n in base_order], float)
        r_guess   = np.percentile(radii_arr, DN_RAD_PCTL) if len(radii_arr) else DN_MIN_R_UM
        r_guess   = float(np.clip(r_guess * DN_RAD_MULT, DN_MIN_R_UM, DN_MAX_R_UM))
    
        d_nodes = np.linalg.norm(base_coords - tip_xyz, axis=1)
        d_nodes[tip_idx] = np.inf
        if pid in nid2idx: d_nodes[nid2idx[pid]] = np.inf
        clear = float(np.min(d_nodes)) if np.isfinite(np.min(d_nodes)) else np.inf
        soma_r = min(r_guess, 0.45*clear) if np.isfinite(clear) else r_guess
        soma_r = float(np.clip(soma_r, DN_MIN_R_UM, DN_MAX_R_UM))
    
        # center far enough past tip so connector doesn’t pierce the sphere
        center = tip_xyz + dirv * (1.25 * soma_r)
    
        # insert soma as child of tip (terminal), big radius
        max_nid += 1
        soma_nid = max_nid
        node[soma_nid] = {
            "type": 1,
            "x": float(center[0]), "y": float(center[1]), "z": float(center[2]),
            "r": float(soma_r), "pid": int(tip_nid),
            "parts": [str(soma_nid), "1",
                      f"{center[0]:.6f}", f"{center[1]:.6f}", f"{center[2]:.6f}",
                      f"{soma_r:.6f}", str(tip_nid)]
        }
    
        # hide any stray sticks inside the sphere by relabeling them soma
        d_center = np.linalg.norm(base_coords - center[None, :], axis=1)
        inside = np.where(d_center <= 2*soma_r)[0]
        for i in inside:
            nid_i = base_order[i]
            if node[nid_i]["type"] != 1:
                node[nid_i]["type"] = 1
    
        print(f"  SOMA  : DN tip cap nid={soma_nid} r≈{soma_r:.2f}µm (offset {1.25*soma_r:.2f}µm), "
              f"hid {len(inside)} in-sphere nodes")

    # 5) write augmented SWC
    base_header = base_header  # keep header
    final_order = base_order[:] + [nid for nid in range(max(base_order)+1, max_nid+1)]
    def _line(nid):
        d=node[nid]; parts=d["parts"][:]; parts[1]=str(d["type"]); parts[6]=str(d["pid"])
        return ' '.join(parts)
    with open(AUG_SWC,"w") as fout:
        for L in base_header: fout.write(L)
        for nid in final_order: fout.write(_line(nid)+"\n")

    # mapping CSV
    records=[]
    def _add_rows(points, seg_idx, t, d, closest, keep_mask, label, start_idx):
        row=start_idx
        for i,keep in enumerate(keep_mask):
            raw = raw_xyz[pre_mask if label=='pre' else post_mask][i]
            ali = (aligned[pre_mask] if label=='pre' else aligned[post_mask])[i]
            if not keep:
                records.append({"syn_index":row,"syn_type":label,
                                "orig_x":raw[0],"orig_y":raw[1],"orig_z":raw[2],
                                "aligned_x":ali[0],"aligned_y":ali[1],"aligned_z":ali[2],
                                "accepted":False})
                row+=1; continue
            sidx=int(seg_idx[i]); tt=float(t[i]); dd=float(d[i]); cxyz=closest[i]
            i_child=seg_a_idx[sidx]; i_parent=seg_b_idx[sidx]
            cA=A[sidx]; cB=B[sidx]
            dA=np.linalg.norm(cxyz-cA); dB=np.linalg.norm(cxyz-cB)
            if dA<=SNAP_TO_NODE or tt<=MERGE_T_EPS:
                used_nid = base_order[i_child]; mode="snap_child"
            elif dB<=SNAP_TO_NODE or (1.0-tt)<=MERGE_T_EPS:
                used_nid = base_order[i_parent]; mode="snap_parent"
            else:
                key_t=round(tt,9)
                used_nid = inserted_nodes.get((sidx,key_t))
                mode="inserted" if used_nid is not None else "snap_unknown"
            lab_final = node[used_nid]["type"] if used_nid is not None else None
            records.append({"syn_index":row,"syn_type":label,
                            "orig_x":raw[0],"orig_y":raw[1],"orig_z":raw[2],
                            "aligned_x":ali[0],"aligned_y":ali[1],"aligned_z":ali[2],
                            "accepted":True,
                            "seg_child":seg_pairs[sidx][0],"seg_parent":seg_pairs[sidx][1],
                            "t_on_segment":tt,"dist_um":dd,
                            "used_node_id":used_nid,"used_node_mode":mode,"used_node_label":lab_final})
            row+=1
        return row
    r_next=0
    r_next=_add_rows(pre_pts,  pre_seg,  pre_t,  pre_d,  pre_c,  pre_keep,  "pre",  r_next)
    _     =_add_rows(post_pts, post_seg, post_t, post_d, post_c, post_keep, "post", r_next)

    map_cols = [
        "syn_index", "syn_type",
        "orig_x", "orig_y", "orig_z",
        "aligned_x", "aligned_y", "aligned_z",
        "accepted",
        "seg_child", "seg_parent", "t_on_segment", "dist_um",
        "used_node_id", "used_node_mode", "used_node_label",
    ]
    df = pd.DataFrame.from_records(records)
    if df.empty:
        df = pd.DataFrame(columns=map_cols)
    else:
        for col in map_cols:
            if col not in df.columns:
                df[col] = pd.NA
        df = df[map_cols]
    df.to_csv(MAP_CSV, index=False)

    # node-label check summary
    acc = df[df["accepted"]==True]
    def _to_bool(v): return str(v).strip().lower() in ("true","t","1","yes","y")
    pre_tot = (df["syn_type"]=="pre").sum()
    post_tot= (df["syn_type"]=="post").sum()
    pre_acc = (acc["syn_type"]=="pre").sum()
    post_acc= (acc["syn_type"]=="post").sum()

    # per-node conflicts
    acc_nodes = acc.dropna(subset=["used_node_id"]).copy()
    acc_nodes["used_node_id"] = acc_nodes["used_node_id"].astype(int, errors="ignore")
    node_groups = acc_nodes.groupby("used_node_id", dropna=True)
    rows=[]; inconsistent=[]
    for nid,g in node_groups:
        types_here=sorted(g["syn_type"].unique().tolist())
        labels_here=sorted(g["used_node_label"].dropna().unique().tolist())
        label = labels_here[0] if labels_here else None
        if len(labels_here)>1:
            inconsistent.append({"used_node_id":nid,"labels_found":labels_here,"syn_types":types_here})
        if set(types_here)=={"pre","post"}:
            expect_true=4; expect_false=5
        elif types_here==["pre"]:
            expect_true=expect_false=4
        elif types_here==["post"]:
            expect_true=expect_false=5
        else:
            expect_true=expect_false=None
        rows.append({"used_node_id":nid,"syn_types_at_node":",".join(types_here),
                     "used_node_label":label,"expect_if_PRE_WINS_true":expect_true,
                     "expect_if_PRE_WINS_false":expect_false,
                     "ok_if_true":(label==expect_true),"ok_if_false":(label==expect_false)})
    node_check = pd.DataFrame(rows)
    node_check.to_csv(CHECK_CSV, index=False)


    # ---- Per-neuron console summary + SOMA check ----
    print("\nSummary:")
    print(f"  PRE   : {pre_acc}/{pre_tot} accepted")
    print(f"  POST  : {post_acc}/{post_tot} accepted")
    print(f"  New SWC nodes inserted: {new_nodes_count}")

    # soma validation against NeuPrint metadata
    try:
        sc = soma_check(client, body_id, BASELINE_SWC)  # uses global `client`
    except Exception as _e:
        sc = {"status":"NA", "reason":f"error: {_e}"}

    if sc["status"] == "NA":
        print("  SOMA  : NA (no somaLocation in metadata)")
    else:
        print(f"  SOMA  : {sc['status']} (min d={sc['d_um']:.3f} µm, n1={sc.get('n_soma',0)}, thr≤{sc['thr_um']:.3f})")

    
    # report json
    report = {
        "files":{"input_swc":INPUT_SWC,"baseline_swc":BASELINE_SWC,"augmented_swc":AUG_SWC,
                 "syn_csv":SYN_CSV,"mapping_csv":MAP_CSV,"node_check_csv":CHECK_CSV},
        "soma_detection":{"seed_pct":SEED_PCT,"grow_pct":GROW_PCT,
                          "branch_max_len":BRANCH_MAX_LEN,"soma_region_nodes":int(len(region))},
        "alignment":{"scale":float(s_scale),"perm":s_perm,"flips":s_flip,
                     "shift":list(map(float,s_shift))},
        "projection":{"k_neighbors":K_NEIGH,"snap_to_node_um":SNAP_TO_NODE,
                      "merge_t_eps":MERGE_T_EPS,"cutoffs_um":{"pre":float(pre_cut),"post":float(post_cut)},
                      "force_accept_all":FORCE_ACCEPT_ALL},
        "labeling_rules":{"pre_wins":PRE_WINS,"lock_soma":LOCK_SOMA,"lock_trunk":LOCK_TRUNK},
        "counts":{"swc_nodes_baseline":int(len(base_recs)),"swc_nodes_augmented":int(len(final_order)),
                  "inserted_nodes":int(new_nodes_count),
                  "syn_total":int(len(syn)),"syn_pre_total":int(pre_mask.sum()),"syn_post_total":int(post_mask.sum()),
                  "pre_accepted":int(pre_acc),"post_accepted":int(post_acc)}
    }
    with open(REPORT_JSON,"w") as f: json.dump(report,f,indent=2)

    return {"bid":body_id,
            "pre_rows":int(pre_tot), "post_rows":int(post_tot),
            "accepted_pre":int(pre_acc), "accepted_post":int(post_acc),
            "paths":{"map":MAP_CSV,"aug":AUG_SWC,"report":REPORT_JSON},
            "soma": sc}
STRICT_UNLABELED_EXTRA_FIELDS = [
    "name", "cellType", "systematicType", "label", "class", "subclass",
    "superclass", "supertype", "celltypePredictedNt", "predictedNt",
    "consensusNt", "otherNt", "modality", "prefix", "target",
    "transmission", "synonyms", "description", "locationType",
]
def _build_unlabeled_where_clause(var_name="n", strict=False):
    fields = ["type", "instance"]
    if strict:
        fields += STRICT_UNLABELED_EXTRA_FIELDS
    return " AND ".join([
        f"trim(coalesce({var_name}.{fld}, '')) = ''"
        for fld in fields
    ])
def fetch_unlabeled_body_ids(client, strict=False):
    where_clause = _build_unlabeled_where_clause("n", strict=strict)
    q = f"""
    MATCH (n:Neuron)
    WHERE n.bodyId IS NOT NULL
      AND {where_clause}
    RETURN n.bodyId AS bodyId
    ORDER BY bodyId ASC
    """
    df = client.fetch_custom(q)
    if df is None or df.empty or "bodyId" not in df.columns:
        return []
    vals = pd.to_numeric(df["bodyId"], errors="coerce").dropna().astype(int).tolist()
    return sorted(set(vals))
def _extract_point_xyz(val):
    """Best-effort parse of a point-like neuPrint value -> (x,y,z) in NM."""
    def _nan3():
        return (np.nan, np.nan, np.nan)

    if val is None:
        return _nan3()

    if isinstance(val, float) and np.isnan(val):
        return _nan3()

    if isinstance(val, dict):
        for keys in (("x", "y", "z"), ("X", "Y", "Z")):
            if all(k in val for k in keys):
                try:
                    return (float(val[keys[0]]), float(val[keys[1]]), float(val[keys[2]]))
                except Exception:
                    return _nan3()

    if isinstance(val, (list, tuple)) and len(val) >= 3:
        try:
            return (float(val[0]), float(val[1]), float(val[2]))
        except Exception:
            return _nan3()

    s = str(val).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return _nan3()

    # Pattern like Point{X=..., Y=..., Z=...}
    try:
        m = _point_re.search(s)
        if m:
            return (float(m.group(1)), float(m.group(2)), float(m.group(3)))
    except Exception:
        pass

    # Try literal dict/list representation
    try:
        import ast
        parsed = ast.literal_eval(s)
        if parsed is not None:
            return _extract_point_xyz(parsed)
    except Exception:
        pass

    # Generic x/y/z key-value fallback
    try:
        import re
        rx = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
        qpat = r"[\"']?"
        mx = re.search(rf"{qpat}x{qpat}\s*[:=]\s*({rx})", s, flags=re.IGNORECASE)
        my = re.search(rf"{qpat}y{qpat}\s*[:=]\s*({rx})", s, flags=re.IGNORECASE)
        mz = re.search(rf"{qpat}z{qpat}\s*[:=]\s*({rx})", s, flags=re.IGNORECASE)
        if mx and my and mz:
            return (float(mx.group(1)), float(my.group(1)), float(mz.group(1)))
    except Exception:
        pass

    return _nan3()
def _append_flat_xyz_from_point_columns(df, point_cols):
    if df is None or df.empty:
        return df

    for col in point_cols:
        xcol = f"{col}_x_nm"
        ycol = f"{col}_y_nm"
        zcol = f"{col}_z_nm"

        if col not in df.columns:
            df[xcol] = np.nan
            df[ycol] = np.nan
            df[zcol] = np.nan
            continue

        xyz = df[col].apply(_extract_point_xyz)
        df[xcol] = xyz.apply(lambda t: t[0])
        df[ycol] = xyz.apply(lambda t: t[1])
        df[zcol] = xyz.apply(lambda t: t[2])

    return df
def _swc_coord_summary(swc_path):
    out = {
        "swc_root_x_um": np.nan,
        "swc_root_y_um": np.nan,
        "swc_root_z_um": np.nan,
        "swc_bbox_min_x_um": np.nan,
        "swc_bbox_min_y_um": np.nan,
        "swc_bbox_min_z_um": np.nan,
        "swc_bbox_max_x_um": np.nan,
        "swc_bbox_max_y_um": np.nan,
        "swc_bbox_max_z_um": np.nan,
        "swc_bbox_center_x_um": np.nan,
        "swc_bbox_center_y_um": np.nan,
        "swc_bbox_center_z_um": np.nan,
    }

    try:
        if swc_path is None or (isinstance(swc_path, float) and np.isnan(swc_path)):
            return out

        sp = Path(str(swc_path)).expanduser()
        if not sp.exists():
            return out

        xs, ys, zs = [], [], []
        root = None

        with open(sp, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 7:
                    continue

                x = float(parts[2]); y = float(parts[3]); z = float(parts[4])
                pid = int(float(parts[6]))

                xs.append(x); ys.append(y); zs.append(z)
                if pid == -1 and root is None:
                    root = (x, y, z)

        if not xs:
            return out

        arr = np.column_stack([np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), np.asarray(zs, dtype=float)])
        lo = np.min(arr, axis=0)
        hi = np.max(arr, axis=0)
        ctr = 0.5 * (lo + hi)

        if root is None:
            root = (float(arr[0, 0]), float(arr[0, 1]), float(arr[0, 2]))

        out.update({
            "swc_root_x_um": float(root[0]),
            "swc_root_y_um": float(root[1]),
            "swc_root_z_um": float(root[2]),
            "swc_bbox_min_x_um": float(lo[0]),
            "swc_bbox_min_y_um": float(lo[1]),
            "swc_bbox_min_z_um": float(lo[2]),
            "swc_bbox_max_x_um": float(hi[0]),
            "swc_bbox_max_y_um": float(hi[1]),
            "swc_bbox_max_z_um": float(hi[2]),
            "swc_bbox_center_x_um": float(ctr[0]),
            "swc_bbox_center_y_um": float(ctr[1]),
            "swc_bbox_center_z_um": float(ctr[2]),
        })
    except Exception:
        return out

    return out
def _append_swc_coordinate_columns(df, swc_col="healed_final_swc"):
    if df is None or df.empty:
        return df

    if swc_col not in df.columns:
        return df

    feats = df[swc_col].apply(_swc_coord_summary)
    feat_df = pd.DataFrame(feats.tolist())

    for col in feat_df.columns:
        df[col] = feat_df[col]

    return df


# =============================================================================
# Choice 1 entry point
# =============================================================================

def option_20_build_and_map(
    client,
    base_out=BASE_OUT,
    unlabeled_export_root=UNLABELED_EXPORT_ROOT,
):
    """Run Choice 1: export healed SWCs, synapse CSVs, and mapped SWCs.

    Relative output roots are resolved under the public `Phase 1` folder. This
    keeps notebook launches independent of the current working directory while
    preserving explicit absolute paths for manual/advanced runs.
    """
    base_out = phase1_output_path(base_out)
    unlabeled_export_root = phase1_output_path(unlabeled_export_root)

    print("\n[Option 20] Enter a body ID, a body ID list (comma/space separated),")
    print("            or AN/IN/DN/MN/SN/ALL, UNLABELED, UNLABELED_STRICT.")
    choice = input("Selection: ").strip()
    rollup = []
    edges_session_ids = []
    soma_counts = {"OK": 0, "FAIL": 0, "NA": 0}
    from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
    import threading
    stats_lock = threading.Lock()

    def run_one(bid, out_dir=None):
        """Option-20 per-bodyId worker.
    
        - Ensures healed SWC + mapped-with-synapses exist in the canonical
          Option-20 folder for this BID.
        - Explicitly skips edges_ego build.
        - Ensures <bid>_synapses_new.csv exists AND has pre_id/post_id/x/y/z/type.
          If not, it is rebuilt using batched custom-Cypher via
          update_synapse_csvs_with_coords().
        """
        from pathlib import Path
        import pandas as pd
    
        bid = int(bid)
    
        # Default Option-20 folder: <base_out>/<FAMILY or TYPE>/<bid>/,
        # unless caller overrides with a custom out_dir.
        out_dir = Path(out_dir or _option20_outdir_for_bid(bid, base_out, client))
        out_dir.mkdir(parents=True, exist_ok=True)
    
        healed_final = out_dir / f"{bid}_healed_final.swc"
        mapped_swc   = out_dir / f"{bid}_axodendro_with_synapses.swc"
        map_csv      = out_dir / f"{bid}_axodendro_with_synapses__synmap.csv"
    
        syn_new_csv    = out_dir / f"{bid}_synapses_new.csv"
        syn_legacy_csv = out_dir / f"{bid}_synapses.csv"

        def _ensure_synapses_new_csv():
            required_xyz = {"x", "y", "z", "type"}

            if syn_new_csv.exists():
                try:
                    probe_new = pd.read_csv(syn_new_csv, nrows=5)
                    if required_xyz.issubset(probe_new.columns):
                        return True
                    missing = required_xyz.difference(probe_new.columns)
                    print(f"  ↻ {bid}: {syn_new_csv.name} missing {missing}; rebuilding")
                except Exception as e:
                    print(f"  ↻ {bid}: could not read {syn_new_csv.name} ({e}); rebuilding")

            if syn_legacy_csv.exists():
                try:
                    legacy = pd.read_csv(syn_legacy_csv)
                    if required_xyz.issubset(legacy.columns):
                        if "pre_id" not in legacy.columns:
                            legacy["pre_id"] = pd.NA
                        if "post_id" not in legacy.columns:
                            legacy["post_id"] = pd.NA
                        legacy = legacy[["pre_id", "post_id", "x", "y", "z", "type"]]
                        legacy.to_csv(syn_new_csv, index=False)
                        print(f"  • {bid}: wrote {syn_new_csv.name} from {syn_legacy_csv.name}")
                        return True
                    print(f"  ↻ {bid}: {syn_legacy_csv.name} missing coordinate/type columns; rebuilding")
                except Exception as e:
                    print(f"  ↻ {bid}: failed converting {syn_legacy_csv.name} ({e}); rebuilding")

            try:
                update_synapse_csvs_with_coords(
                    base_out=out_dir,
                    min_conf=0.4,
                    skip_existing=False,
                    body_ids=[bid],
                    client=client,
                )
            except Exception as e:
                print(f"[update_synapse_csvs_with_coords] WARNING for {bid}: {e}")

            if syn_new_csv.exists():
                try:
                    probe_new = pd.read_csv(syn_new_csv, nrows=5)
                    return required_xyz.issubset(probe_new.columns)
                except Exception:
                    return False
            return False

        # ------------------------------------------------------------------
        # 1) SWC + mapping (same behavior as before)
        # ------------------------------------------------------------------
        if not (healed_final.exists() and mapped_swc.exists() and map_csv.exists()):
            swc_path, syn_path, _ = _export_one_neuron(bid, out_dir, client)

            if not _ensure_synapses_new_csv():
                print(f"[ERROR] {bid}: unable to prepare {syn_new_csv.name}; skipping mapping")
                return

            try:
                summary = _run_mapping_for(bid, out_dir)
            except Exception as e:
                print(f"[Option 20] WARNING {bid}: mapping failed ({e})")
                return
    
            with stats_lock:
                soma_status = summary.get("soma", {}).get("status", "NA")
                if soma_status not in soma_counts:
                    soma_status = "NA"
                soma_counts[soma_status] += 1
                rollup.append(summary)
        else:
            print(f"  ✓ Skipping {bid}: SWC+mapping already present ({out_dir})")
    
        # ------------------------------------------------------------------
        # 2) edges_ego: completely disabled (no large custom queries)
        # ------------------------------------------------------------------
        ego_csv_path = Path(_edges_ego_csv_path(bid, out_dir))
        if ego_csv_path.exists():
            print(f"[edges_ego] {ego_csv_path.name} already exists (leaving as-is)")
        else:
            print(f"[edges_ego] Skipping edges_ego build for {bid} "
                  f"(disabled to avoid large custom queries)")
    
        with stats_lock:
            edges_session_ids.append(bid)
    
        # ------------------------------------------------------------------
        # 3) Ensure <bid>_synapses_new.csv has pre_id/post_id/x/y/z/type
        # ------------------------------------------------------------------
        required_cols = {"pre_id", "post_id", "x", "y", "z", "type"}
        syn_csv = out_dir / f"{bid}_synapses_new.csv"
        needs_rebuild = True
    
        if syn_csv.exists():
            try:
                probe = pd.read_csv(syn_csv, nrows=5)
                if required_cols.issubset(probe.columns):
                    print(f"  • {bid}: {syn_csv.name} already has required columns — keeping")
                    needs_rebuild = False
                else:
                    missing = required_cols.difference(probe.columns)
                    print(f"  ↻ {bid}: {syn_csv.name} missing {missing}; will rebuild")
            except Exception as e:
                print(f"  ↻ {bid}: could not read existing {syn_csv.name} ({e}); will rebuild")
    
        if needs_rebuild:
            try:
                update_synapse_csvs_with_coords(
                    base_out=out_dir,
                    min_conf=0.4,
                    skip_existing=False,   # force rebuild even if file exists
                    body_ids=[bid],
                    client=client,         # use the live neuprint.Client
                )
            except Exception as e:
                print(f"[update_synapse_csvs_with_coords] WARNING for {bid}: {e}")

        

    import re as _re_opt20
    from pathlib import Path

    try:
        from tqdm.auto import tqdm as _tqdm_opt20
    except Exception:
        _tqdm_opt20 = None
    import time as _time_opt20

    def _ask_parallel_workers(default_workers=8, max_allowed=32):
        if default_workers < 1:
            default_workers = 1
        default_workers = int(min(max_allowed, max(1, default_workers)))
        ans = input(f"Max parallel workers [1-{max_allowed}] [{default_workers}]: ").strip()
        if not ans:
            return default_workers
        try:
            w = int(ans)
        except Exception:
            print(f"[Option 20] Invalid worker count '{ans}', using {default_workers}")
            return default_workers
        if w < 1:
            w = 1
        if w > max_allowed:
            print(f"[Option 20] Capping workers to {max_allowed}")
            w = max_allowed
        return w

    def _run_many(body_ids, out_dir_fn=None, run_label="batch", max_workers=1):
        ids = [int(b) for b in body_ids]
        if not ids:
            return

        total = len(ids)
        started_at = _time_opt20.time()

        if _tqdm_opt20 is not None:
            pbar = _tqdm_opt20(
                total=total,
                desc=f"Option 20 [{run_label}]",
                unit="neuron",
                position=0,
                leave=True,
                dynamic_ncols=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )
        else:
            pbar = None
            print(f"[Option 20] Starting '{run_label}' for {total} neurons...")

        def _mark_progress(n=1):
            if pbar is not None:
                pbar.update(n)
                return
            # text fallback with ETA every 25 steps
            _mark_progress.done += n
            if _mark_progress.done % 25 == 0 or _mark_progress.done == total:
                elapsed = max(1e-9, _time_opt20.time() - started_at)
                rate = _mark_progress.done / elapsed
                remain = max(0, total - _mark_progress.done)
                eta_s = remain / rate if rate > 1e-12 else float("inf")
                if eta_s == float("inf"):
                    eta_txt = "unknown"
                else:
                    h = int(eta_s // 3600)
                    m = int((eta_s % 3600) // 60)
                    s = int(eta_s % 60)
                    eta_txt = f"{h:02d}:{m:02d}:{s:02d}"
                print(f"[Option 20] Progress: {_mark_progress.done}/{total} | ETA {eta_txt}")

        _mark_progress.done = 0

        try:
            if max_workers <= 1:
                for bid in ids:
                    out_dir = out_dir_fn(bid) if out_dir_fn else None
                    run_one(bid, out_dir=out_dir)
                    _mark_progress(1)
                return

            print(f"[Option 20] Parallel run '{run_label}' with max_workers={max_workers} on {total} neurons")
            max_inflight = max(1, int(max_workers) * 4)

            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                ids_iter = iter(ids)
                inflight = {}

                def _submit_next(n=1):
                    submitted = 0
                    while submitted < n:
                        try:
                            bid_next = next(ids_iter)
                        except StopIteration:
                            break
                        out_dir_next = out_dir_fn(bid_next) if out_dir_fn else None
                        fut_next = ex.submit(run_one, bid_next, out_dir_next)
                        inflight[fut_next] = bid_next
                        submitted += 1
                    return submitted

                _submit_next(max_inflight)

                while inflight:
                    done_set, _ = wait(inflight.keys(), return_when=FIRST_COMPLETED)
                    for fut in done_set:
                        bid = inflight.pop(fut)
                        try:
                            fut.result()
                        except Exception as e:
                            print(f"[Option 20] WARNING {bid}: worker failed ({e})")
                    _mark_progress(len(done_set))
                    _submit_next(len(done_set))
        finally:
            if pbar is not None:
                pbar.close()


    # Parse possible explicit list of bodyIds: "10311, 10477 12749"
    id_tokens = [t for t in _re_opt20.split(r"[,\s]+", choice) if t]

    choice_key = choice.strip().upper()
    unlabeled_mode_map = {
        "UNLABELED": False,
        "UNLABELLED": False,
        "UL": False,
        "UNLABELED_STRICT": True,
        "UNLABELLED_STRICT": True,
        "UL_STRICT": True,
        "UNLABELED-STRICT": True,
    }

    if choice_key in unlabeled_mode_map:
        strict_mode = unlabeled_mode_map[choice_key]
        mode_label = "strict" if strict_mode else "unstrict"
        unlabeled_ids = fetch_unlabeled_body_ids(client, strict=strict_mode)

        print(f"\n[Option 20] {mode_label} unlabeled bodyIds found: {len(unlabeled_ids)}")
        if not unlabeled_ids:
            print("No neurons matched this unlabeled filter.")
            return

        default_root = unlabeled_export_root
        root_in = input(f"Output root for {mode_label} unlabeled exports [{default_root}]: ").strip()
        out_root = phase1_output_path(root_in or default_root)
        out_root.mkdir(parents=True, exist_ok=True)
        print(f"[Option 20] Saving per-neuron folders under: {out_root}")

        default_workers = 8 if len(unlabeled_ids) > 1 else 1
        max_workers = _ask_parallel_workers(default_workers=default_workers)
        _run_many(
            unlabeled_ids,
            out_dir_fn=lambda bid: (out_root / str(int(bid))),
            run_label=f"unlabeled_{mode_label}",
            max_workers=max_workers,
        )

        default_master = out_root / f"unlabeled_{mode_label}_master_metadata.csv"
        master_in = input(f"Master metadata CSV path [{default_master}]: ").strip()
        master_csv = phase1_output_path(master_in) if master_in else default_master

        try:
            df_master = export_all_neuroncriteria_template(
                csv_out=str(master_csv),
                criteria_kwargs=None,
                body_ids=unlabeled_ids,
                np_client=client,
            )

            if "bodyId" in df_master.columns:
                body_series = pd.to_numeric(df_master["bodyId"], errors="coerce")
                df_master["option20_out_dir"] = [
                    (str(out_root / str(int(b)))) if pd.notna(b) else pd.NA
                    for b in body_series
                ]
                df_master["healed_final_swc"] = [
                    (str(out_root / str(int(b)) / f"{int(b)}_healed_final.swc")) if pd.notna(b) else pd.NA
                    for b in body_series
                ]
                df_master["mapped_swc"] = [
                    (str(out_root / str(int(b)) / f"{int(b)}_axodendro_with_synapses.swc")) if pd.notna(b) else pd.NA
                    for b in body_series
                ]
                df_master["synapses_new_csv"] = [
                    (str(out_root / str(int(b)) / f"{int(b)}_synapses_new.csv")) if pd.notna(b) else pd.NA
                    for b in body_series
                ]
                df_master["unlabeled_mode"] = mode_label

                point_cols = ["location", "avgLocation", "somaLocation", "tosomaLocation", "rootLocation"]
                df_master = _append_flat_xyz_from_point_columns(df_master, point_cols)
                df_master = _append_swc_coordinate_columns(df_master, swc_col="healed_final_swc")

                df_master.to_csv(master_csv, index=False)

            print(f"[master_csv] Wrote {master_csv} ({len(df_master)} rows)")
        except Exception as e:
            print(f"[master_csv] WARNING: failed to write master metadata CSV: {e}")

    elif id_tokens and all(tok.isdigit() for tok in id_tokens):
        id_list = sorted({int(tok) for tok in id_tokens})
        custom_single_out = None

        if len(id_list) == 1:
            custom_single_in = input(
                "Optional save folder for this neuron (leave blank for default): "
            ).strip()
            if custom_single_in:
                custom_single_out = str(phase1_output_path(custom_single_in))
                print(f"[Option 20] Custom output folder: {custom_single_out}")

        print(f"\n[Option 20] Processing explicit bodyId list ({len(id_list)} neurons):")
        max_workers = 1 if len(id_list) == 1 else _ask_parallel_workers(default_workers=min(8, len(id_list)))
        _run_many(
            id_list,
            out_dir_fn=(lambda bid: custom_single_out if len(id_list) == 1 else None),
            run_label="explicit_body_ids",
            max_workers=max_workers,
        )
    else:
        # Family / ALL mode
        FAM_SET = {"AN", "IN", "DN", "MN", "SN"}
        fam_in = choice.upper()

        if fam_in == "ALL":
            fams = ["AN", "IN", "DN", "MN", "SN"]
        else:
            fams = [fam_in]

        for fam in fams:
            if fam not in FAM_SET:
                print("Use a body ID (or list), AN/IN/DN/MN/SN, ALL, UNLABELED, or UNLABELED_STRICT")
                return

        for fam in fams:
            print(f"\n=== Family {fam} ===")
            if fam == "MN":
                mn_df = _fetch_mn_candidates(client, regex=MN_REGEX_DEFAULT)
                if mn_df.empty:
                    print("  (No MN candidates found)")
                    continue
                all_ids = sorted(mn_df["bodyId"].astype(int).tolist())
                inst_map = _instance_map_for_ids(client, all_ids)
                tmap = {}
                for bid in all_ids:
                    label = inst_map.get(bid, "by_id")
                    tmap.setdefault(label, []).append(bid)
            else:
                tmap = _list_types_by_prefix(fam, client, pattern=rf"{fam}")

            if not tmap:
                print(f"  (No types for {fam})")
                continue

            # Print by "instance_label", but do not use it for grouping.
            # to build paths – run_one() always defers to _option20_outdir_for_bid.
            for instance_label, body_ids in sorted(tmap.items()):
                print(f"\n  • Instance {instance_label} ({len(body_ids)} neuron"
                      f"{'s' if len(body_ids) != 1 else ''})")
                for bid in sorted(set(body_ids)):
                    run_one(bid)
    # --- Rollup summary ---
    if not rollup:
        print("\nNo neurons processed.")
        return

    probs = []
    for r in rollup:
        ok_pre  = (r["accepted_pre"]  == r["pre_rows"])
        ok_post = (r["accepted_post"] == r["post_rows"])
        if not (ok_pre and ok_post):
            probs.append(r)

    print("\n==== Synapse-mapping summary ====")
    print(f"Processed: {len(rollup)} neurons")
    tot_pre   = sum(r["pre_rows"] for r in rollup)
    tot_post  = sum(r["post_rows"] for r in rollup)
    tot_pre_a = sum(r["accepted_pre"] for r in rollup)
    tot_post_a= sum(r["accepted_post"] for r in rollup)
    print(f"  PRE  : {tot_pre_a}/{tot_pre} accepted")
    print(f"  POST : {tot_post_a}/{tot_post} accepted")
    if probs:
        print("\n  Mismatches:")
        for r in probs:
            print(f"   - {r['bid']}: pre {r['accepted_pre']}/{r['pre_rows']}, "
                  f"post {r['accepted_post']}/{r['post_rows']}")
            print(f"       map: {r['paths']['map']}")
    else:
        print("  All neurons matched (accepted == total).")
        print(f"  SOMA : OK={soma_counts['OK']}  FAIL={soma_counts['FAIL']}  NA={soma_counts['NA']}")

    # --- edges_ego merge: usually empty now, but keep existing logic ---
    try:
        from pathlib import Path

        ego_ids = sorted({int(b) for b in edges_session_ids})
        if ego_ids:
            base_dir = Path(base_out)
            per_files = []
            for bid in ego_ids:
                pattern = f"**/{bid}/{bid}_edges_rawsyn.csv"
                hits = list(base_dir.glob(pattern))
                if not hits:
                    hits = list(base_dir.glob(f"**/{bid}_edges_rawsyn.csv"))
                if hits:
                    per_files.append(hits[0])

            if per_files:
                frames = [pd.read_csv(p) for p in per_files if p.exists()]
                if frames:
                    ego_df = pd.concat(frames, ignore_index=True)
                    ego_name = f"edges_ego_{'_'.join(map(str, ego_ids))}__rawsyn.csv"
                    ego_csv  = base_dir / ego_name
                    ego_df.to_csv(ego_csv, index=False)
                    print(f"[edges_csv] Wrote {ego_csv.name} ({len(ego_df)} rows)")
                else:
                    print("[edges_csv] No per-neuron edges files found to merge.")
            else:
                print("[edges_csv] No per-neuron edges files registered.")
        else:
            print("[edges_csv] No neurons processed; no ego file written.")
    except Exception as e:
        print(f"[edges_csv] WARNING: ego merge failed: {e}")


def update_synapse_csvs_with_coords(
    base_out=BASE_OUT,
    min_conf=0.4,
    skip_existing=True,
    body_ids=None,
    client=None,
    batch_size=10000,
):
    """
    For each bodyId BID, write <BID>_synapses_new.csv in its Option-20 folder.

    Uses batched custom Cypher queries, so even large neurons are fetched
    chunk-by-chunk instead of one huge 20k+ pull.

    Output columns:
        pre_id, post_id, x, y, z, type  ('pre' or 'post')

    - 'pre' rows  = synapses where BID is presynaptic (coords on BID side)
    - 'post' rows = synapses where BID is postsynaptic (coords on BID side)
    """
    # Called by Choice 1 when synapse CSVs are missing or stale.
    import os, time
    from pathlib import Path
    import pandas as pd
    from neuprint import Client

    base_out = phase1_output_path(base_out)
    if not base_out.exists():
        print(f"[WARN] Base directory {base_out} not found.")
        return

    # ---- neuPrint client ----
    if client is None:
        base_client = globals().get("client", None)
        if base_client is not None and isinstance(base_client, Client):
            client = base_client
        else:
            server  = os.environ.get("NEUPRINT_SERVER", "https://neuprint.janelia.org")
            dataset = os.environ.get("NEUPRINT_DATASET", None)
            token   = get_neuprint_token(required=False)
            if not dataset or not token:
                print("[ERROR] Missing dataset/token. Set NEUPRINT_DATASET and save a neuPrint token.")
                return
            client = Client(server, dataset=dataset, token=token)

    # ---- SWC discovery → prefer TYPE over INSTANCE folders ----
    def _scan_swc_ids(root: Path):
        """
        Scan for SWCs and return [(bid, folder), ...].

        Preference:
          1) TYPE folders   e.g. IN/IN12B002/10070
          2) INSTANCE only if no TYPE folder exists   e.g. IN/IN12B002_T1_R/10070
        """
        pats = [
            "**/*_healed_final.swc",
            "**/*_healed.swc",
            "**/*with_synapses*.swc",
            "**/*.swc",
        ]
        type_hits = {}
        inst_hits = {}

        for pat in pats:
            for p in root.glob(pat):
                try:
                    bid = int(p.name.split("_", 1)[0])
                except Exception:
                    continue
                parent = p.parent
                # heuristic: type folders have no '_' in their name,
                # instance folders do (IN12B002_T1_R, DNp01(GF)_R, etc.)
                if "_" in parent.name:
                    inst_hits.setdefault(bid, parent)
                else:
                    type_hits.setdefault(bid, parent)

        hits = []
        # prefer TYPE if present
        for bid, folder in type_hits.items():
            hits.append((bid, folder))
        # add INSTANCE only when no TYPE folder was observed.
        for bid, folder in inst_hits.items():
            if bid not in type_hits:
                hits.append((bid, folder))
        return hits

    # ---- build body list and id→dir map ----
    if body_ids is None:
        swc_hits = _scan_swc_ids(base_out)
        if not swc_hits:
            print(f"[INFO] No SWC files found under {base_out}")
            return
        id_to_dir = {bid: folder for bid, folder in swc_hits}
        body_list = sorted(id_to_dir.keys())
    else:
        body_list = sorted({int(b) for b in body_ids})
        swc_hits = dict(_scan_swc_ids(base_out))
        id_to_dir = {
            bid: (swc_hits.get(bid) or (base_out / "by_id" / str(bid)))
            for bid in body_list
        }

    # ---- skip logic based on existing _synapses_new.csv ----
    required_cols = {"pre_id", "post_id", "x", "y", "z", "type"}
    if skip_existing:
        keep = []
        for bid in body_list:
            out_dir = Path(id_to_dir[bid]); out_dir.mkdir(parents=True, exist_ok=True)
            csv_path = out_dir / f"{bid}_synapses_new.csv"
            if csv_path.exists():
                try:
                    df0 = pd.read_csv(csv_path, nrows=5)
                    if required_cols.issubset(df0.columns):
                        print(f"  • {bid}: {csv_path.name} already has required columns — skipping")
                        continue
                    else:
                        print(f"  ↻ {bid}: {csv_path.name} missing some columns; will rebuild")
                except Exception as e:
                    print(f"  ↻ {bid}: could not read existing {csv_path.name} ({e}); will rebuild")
            keep.append(bid)
        body_list = keep

    if not body_list:
        print("[INFO] Synapse coordinate update: nothing to do.")
        return

    # ---- batched custom-Cypher templates (Python .format, no params dict) ----
    PRE_CYPHER_TMPL = """
    MATCH (pre:Neuron {{bodyId: {bid}}})-[:Contains]->(:SynapseSet)-[:Contains]->(preSyn:Synapse),
          (preSyn)-[:SynapsesTo]->(postSyn:Synapse)<-[:Contains]-(:SynapseSet)<-[:Contains]-(post:Neuron)
    WHERE preSyn.type = 'pre'
      AND postSyn.type = 'post'
      AND preSyn.confidence  >= {min_conf}
      AND postSyn.confidence >= {min_conf}
    RETURN pre.bodyId  AS pre_id,
           post.bodyId AS post_id,
           preSyn.location.x AS x,
           preSyn.location.y AS y,
           preSyn.location.z AS z,
           'pre' AS type
    SKIP {skip} LIMIT {limit}
    """

    POST_CYPHER_TMPL = """
    MATCH (pre:Neuron)-[:Contains]->(:SynapseSet)-[:Contains]->(preSyn:Synapse),
          (preSyn)-[:SynapsesTo]->(postSyn:Synapse)<-[:Contains]-(:SynapseSet)<-[:Contains]-(post:Neuron {{bodyId: {bid}}})
    WHERE preSyn.type = 'pre'
      AND postSyn.type = 'post'
      AND preSyn.confidence  >= {min_conf}
      AND postSyn.confidence >= {min_conf}
    RETURN pre.bodyId  AS pre_id,
           post.bodyId AS post_id,
           postSyn.location.x AS x,
           postSyn.location.y AS y,
           postSyn.location.z AS z,
           'post' AS type
    SKIP {skip} LIMIT {limit}
    """

    def _fetch_side(bid: int, tmpl: str, side_label: str):
        """Fetch 'pre' or 'post' rows for one neuron in batches."""
        all_chunks = []
        skip = 0
        total = 0
        while True:
            cypher = tmpl.format(
                bid=int(bid),
                min_conf=float(min_conf),
                skip=int(skip),
                limit=int(batch_size),
            )
            df_chunk = client.fetch_custom(cypher)
            if df_chunk is None or df_chunk.empty:
                if skip == 0:
                    print(f"[{side_label.upper()}] {bid}: no rows")
                break

            n_chunk = len(df_chunk)
            total += n_chunk
            all_chunks.append(df_chunk)

            print(f"[{side_label.upper()}] {bid}: fetched {total} (chunk {n_chunk})")

            if n_chunk < batch_size:
                break
            skip += batch_size

        if not all_chunks:
            return pd.DataFrame(columns=["pre_id", "post_id", "x", "y", "z", "type"])
        return pd.concat(all_chunks, ignore_index=True)

    # ---- main loop ----
    print(f"[INFO] Exporting synapse coords for {len(body_list)} neuron(s) "
          f"(batch_size={int(batch_size)})")

    for bid in body_list:
        out_dir = Path(id_to_dir[bid]); out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"{bid}_synapses_new.csv"

        print(f"\n[update_synapse_csvs_with_coords] Neuron {bid}")
        t0 = time.time()

        df_pre  = _fetch_side(bid, PRE_CYPHER_TMPL,  "pre")
        df_post = _fetch_side(bid, POST_CYPHER_TMPL, "post")
        df_all = pd.concat([df_pre, df_post], ignore_index=True)

        if df_all.empty:
            print(f"  ! {bid}: no synapse pairs fetched; writing empty CSV")
            df_all = pd.DataFrame(columns=["pre_id", "post_id", "x", "y", "z", "type"])
        else:
            df_all = df_all[["pre_id", "post_id", "x", "y", "z", "type"]]

        df_all.to_csv(csv_path, index=False)
        dt = time.time() - t0
        print(f"  • {bid}: wrote {len(df_all)} rows → {csv_path}  ({dt:0.1f}s)")

    print("\n[INFO] Synapse coordinate update complete (batched custom-Cypher).")


# =============================================================================
# Choice 3 metadata export, also reused by Choice 1 unlabeled batches
# =============================================================================

def export_all_neuroncriteria_template(
    csv_out: str = "all_neurons_neuroncriteria_template.csv",
    criteria_kwargs: dict | None = None,
    body_ids: list[int] | None = None,
    np_client=None,
    chunk_size: int = 2000,
):
    """
    Export neuron metadata in a NeuronCriteria-style template.

    - If `body_ids` is None, exports all neurons matching `criteria_kwargs`.
    - If `body_ids` is provided, exports only those neuron IDs (chunked).
    - Used directly by Choice 3 and internally by Choice 1's unlabeled mode.
    """

    if criteria_kwargs is None:
        criteria_kwargs = {}

    if np_client is None:
        np_client = navis_client

    if body_ids is None:
        # 1a) Build criteria + fetch all matching neurons
        nc = NeuronCriteria(**criteria_kwargs)
        # IMPORTANT: avoid printing nc directly; some neuPrint clients have a broken __repr__.
        print("[INFO] Fetching neurons with NeuronCriteria and kwargs:", criteria_kwargs)
        df_neurons, _ = fetch_neurons(nc, client=np_client)
    else:
        # 1b) Fetch explicit bodyId list in chunks
        ids = sorted({int(b) for b in body_ids})
        if not ids:
            df_neurons = pd.DataFrame(columns=["bodyId"])
        else:
            chunk_size = max(1, int(chunk_size))
            print(f"[INFO] Fetching neurons for explicit bodyId list: {len(ids)} ids (chunk_size={chunk_size})")
            frames = []
            for i in range(0, len(ids), chunk_size):
                chunk = ids[i:i + chunk_size]
                df_chunk, _ = fetch_neurons(NeuronCriteria(bodyId=chunk), client=np_client)
                if df_chunk is not None and not df_chunk.empty:
                    frames.append(df_chunk)

            if frames:
                df_neurons = pd.concat(frames, ignore_index=True)
            else:
                df_neurons = pd.DataFrame(columns=["bodyId"])

            if "bodyId" not in df_neurons.columns:
                df_neurons["bodyId"] = pd.Series(dtype="Int64")

            found_ids = set(pd.to_numeric(df_neurons["bodyId"], errors="coerce").dropna().astype(int).tolist())
            missing_ids = [bid for bid in ids if bid not in found_ids]
            if missing_ids:
                missing_df = pd.DataFrame({"bodyId": missing_ids})
                df_neurons = pd.concat([df_neurons, missing_df], ignore_index=True, sort=False)
                print(f"[WARN] {len(missing_ids)} requested bodyId(s) were not returned; adding placeholder rows.")

    print(f"[INFO] Retrieved {len(df_neurons)} neurons and {len(df_neurons.columns)} raw columns.")

    # 2) NeuronCriteria-style template fields to include as columns.
    template_fields = [
        "bodyId",
        "type",
        "instance",
        "status",
        "statusLabel",
        "rois",
        "inputRois",
        "outputRois",
        "group",
        "serial",
        "zapbenchId",
        "cropped",
        "flywireType",
        "hemibrainType",
        "mancType",
        "birthtime",
        "cellBodyFiber",
        "celltypePredictedNt",
        "class_",            # mapped from 'class' if present
        "consensusNt",
        "description",
        "dimorphism",
        "entryNerve",
        "exitNerve",
        "flywireId",
        "fruDsx",
        "hemibrainBodyId",
        "hemilineage",
        "itoleeHl",
        "locationType",
        "longTract",
        "matchingNotes",
        "mcnsSerial",
        "modality",
        "ntReference",
        "origin",
        "otherNt",
        "otherNtReference",
        "predictedNt",
        "prefix",
        "receptorType",
        "rootSide",
        "serialMotif",
        "somaNeuromere",
        "somaSide",
        "source",
        "subclass",
        "subclassabbr",
        "superclass",
        "supertype",
        "synonyms",
        "systematicType",
        "tag",
        "target",
        "transmission",
        "trumanHl",
        "vfbId",
        "label",
        "somaLocation",
        "tosomaLocation",
        "rootLocation",
        "soma",
    ]

    # 3) Handle special cases / aliases from the raw dataframe
    if "class" in df_neurons.columns and "class_" not in df_neurons.columns:
        df_neurons["class_"] = df_neurons["class"]

    # 4) Ensure all template fields exist as columns
    for col in template_fields:
        if col not in df_neurons.columns:
            df_neurons[col] = np.nan

    # 5) Reorder columns: template_fields first, then any extra columns from the DB
    ordered_cols = (
        [c for c in template_fields if c in df_neurons.columns]
        + [c for c in df_neurons.columns if c not in template_fields]
    )
    df_neurons = df_neurons[ordered_cols]

    # 6) Write CSV under Phase 1 unless the caller supplied an absolute path.
    csv_out = phase1_output_path(csv_out)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    df_neurons.to_csv(csv_out, index=False)
    print(
        f"[INFO] Wrote {len(df_neurons)} neurons × {len(df_neurons.columns)} columns "
        f"to '{csv_out}'."
    )

    return df_neurons


# =============================================================================
# Choice 4 pathfinding helpers
# =============================================================================

MALE_CNS_DATASET = "male-cns:v0.9" #'manc:v1.2.1'    
MALE_CNS_SERVER  = "https://neuprint.janelia.org"
def get_male_cns_client():
    """
    Lazily construct the Choice 4 male-cns client without disturbing the
    active MANC client used by the other Phase 1 choices.
    """
    global male_cns_client
    try:
        return male_cns_client
    except NameError:
        print(f"[neuprint] Creating client for dataset '{MALE_CNS_DATASET}'")
        # Token handling is centralized through get_neuprint_token().
        male_cns_client = neu.Client(
            MALE_CNS_SERVER,
            dataset=MALE_CNS_DATASET,
            token=get_neuprint_token(required=True)

        )
        return male_cns_client


def _parse_bodyid_list(text):
    """
    Parse comma/space-separated bodyId list into ints.
    """
    ids = []
    for tok in text.replace(",", " ").split():
        tok = tok.strip()
        if not tok:
            continue
        try:
            ids.append(int(tok))
        except ValueError:
            print(f"[WARN] Skipping non-integer token: {tok!r}")
    return ids
def _choose_neuron_set(prompt, default_type=None):
    """
    Ask user how they want to specify a neuron set:
    - bodyId list
    - type (exact)
    - instance (regex)
    
    Returns a list of bodyIds (may be empty if nothing found).
    """
    client = get_male_cns_client()

    print("\n" + "=" * 72)
    print(prompt)
    print("-" * 72)
    print("   1) By bodyId list")
    print("   2) By neuron type (exact match)")
    print("   3) By neuron instance (regex match)")
    mode = input("Select mode [1/2/3, default 2]: ").strip() or "2"

    if mode == "1":
        raw = input("Enter one or more bodyIds (comma/space separated): ").strip()
        ids = _parse_bodyid_list(raw)
        if not ids:
            print("[WARN] No valid bodyIds parsed.")
        else:
            print(f"[INFO] Using {len(ids)} bodyId(s): {ids}")
        return ids

    elif mode == "2":
        t = input(f"Neuron type (e.g. 'AN08B098', 'DNp01')"
                  f"{' [' + default_type + ']' if default_type else ''}: ").strip()
        if not t and default_type:
            t = default_type
        if not t:
            print("[WARN] No type provided, returning empty list.")
            return []
    
        # NEW: allow vague token like 'MN' to mean "contains MN"
        t_regex = _as_contains_regex(t)
        use_regex = (t_regex != t) or any(ch in t for ch in r".^$*+?{}[]\|()")
    
        crit = neu.NeuronCriteria(type=t_regex, regex=use_regex, status="Traced")
        neurons_df = neu.fetch_neurons(crit, omit_rois=True, client=client)
        if neurons_df.empty:
            print(f"[WARN] No neurons found with type='{t}' (regex={use_regex}) in {MALE_CNS_DATASET}")
            return []
    
        print(f"[INFO] Found {len(neurons_df)} neuron(s) for type query '{t}' (regex={use_regex})")
        print(neurons_df[["bodyId", "instance", "type"]].head())
        return neurons_df["bodyId"].tolist()


    else:  # mode == "3"
        inst = input("Neuron instance (exact or regex): ").strip()
        if not inst:
            print("[WARN] No instance provided, returning empty list.")
            return []
    
        # NEW: allow vague token like 'MN' to mean "contains MN"
        inst_regex = _as_contains_regex(inst)
        use_regex = (inst_regex != inst) or any(ch in inst for ch in r".^$*+?{}[]\|()")
    
        crit = neu.NeuronCriteria(instance=inst_regex, regex=use_regex, status="Traced")
        neurons_df = neu.fetch_neurons(crit, omit_rois=True, client=client)
        if neurons_df.empty:
            print(f"[WARN] No neurons found with instance query='{inst}' (regex={use_regex}) "
                  f"in {MALE_CNS_DATASET}")
            return []
    
        print(f"[INFO] Found {len(neurons_df)} neuron(s) for instance query '{inst}' (regex={use_regex})")
        print(neurons_df[["bodyId", "instance", "type"]].head())
        return neurons_df["bodyId"].tolist()
def find_mn_within_hops_inline(
    source_bodyId: int,
    token: str = "MN",
    max_hops: int = 3,
    min_weight: int = 1,
    match_on: str = "either",
    limit_paths_per_target: int = 3,
    client=None
):
    if client is None:
        client = get_male_cns_client()

    rx = _as_contains_regex(token)
    # Escape backslashes and quotes for Cypher string literal
    rx_cypher = rx.replace("\\", "\\\\").replace('"', '\\"')

    if match_on == "type":
        target_match = f'dst.type =~ "{rx_cypher}"'
    elif match_on == "instance":
        target_match = f'dst.instance =~ "{rx_cypher}"'
    else:
        target_match = f'(dst.type =~ "{rx_cypher}" OR dst.instance =~ "{rx_cypher}")'

    cypher = f"""
    MATCH (src:Neuron {{bodyId: {int(source_bodyId)}}})
    MATCH p=(src)-[r:ConnectsTo*1..{int(max_hops)}]->(dst:Neuron)
    WHERE ALL(rel IN relationships(p) WHERE rel.weight >= {int(min_weight)})
      AND {target_match}
      AND dst.bodyId <> {int(source_bodyId)}
    WITH dst, p, length(p) AS hops
    ORDER BY hops ASC
    WITH dst,
         min(hops) AS minHops,
         collect([n IN nodes(p) | n.bodyId])[0..{int(limit_paths_per_target)}] AS exampleBodyIdPaths
    RETURN
        dst.bodyId AS bodyId,
        dst.type AS type,
        dst.instance AS instance,
        minHops AS hops,
        exampleBodyIdPaths AS example_paths
    ORDER BY hops ASC, bodyId ASC
    """
    return client.fetch_custom(cypher, format="pandas")
import re
def _as_contains_regex(s: str) -> str:
    """
    Convert a plain token like 'MN' into a safe 'contains' regex: '.*MN.*'
    If it already looks like a regex, return as-is.
    """
    s = (s or "").strip()
    if not s:
        return s

    # Heuristic: if user already typed regex-like chars, respect it.
    # (Also covers things like 'MN.*', '.*MN.*', '^MN', etc.)
    regexy = any(ch in s for ch in r".^$*+?{}[]\|()")
    if regexy:
        return s

    # Treat as substring match
    return f".*{re.escape(s)}.*"
def _summarize_paths_df(df, max_paths_to_print=10):
    """
    Pretty-print a few paths from a fetch_paths/fetch_shortest_paths DataFrame.
    """
    if df.empty:
        print("  (no paths found)")
        return

    # df columns: path, bodyId, type, weight  (per neuprint docs)
    print(f"[INFO] Total rows: {len(df)}; unique paths: {df['path'].nunique()}")
    print(f"[INFO] Showing up to {max_paths_to_print} path(s):")

    for path_id, g in df.groupby("path"):
        if max_paths_to_print <= 0:
            print("  ... (more paths omitted)")
            break
        max_paths_to_print -= 1

        g = g.sort_index()
        chain = []
        for i, row in g.iterrows():
            label = f"{row['type']} ({row['bodyId']})"
            if row["weight"] and int(row["weight"]) > 0:
                chain.append(f"--{int(row['weight'])}--> {label}")
            else:
                # First node in the path has weight 0
                chain.append(label)
        # Make sure the first element is just the label (no leading arrow)
        if chain:
            if chain[0].startswith("--"):
                chain[0] = chain[0].split("--", 1)[-1].lstrip(">-")
        print(f"  Path {path_id}: " + " ".join(chain))
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None  # fallback if tqdm isn't installed
def _make_pair_tasks(upstream_ids, downstream_ids):
    tasks = []
    for up_id in upstream_ids:
        for down_id in downstream_ids:
            if up_id == down_id:
                continue
            tasks.append((up_id, down_id))
    return tasks
def run_pair_path_queries_parallel(
    upstream_ids,
    downstream_ids,
    use_all_paths: bool,
    min_weight: int,
    path_kwargs: dict = None,
    max_workers: int = 12,
):
    """
    Parallel version of the nested loops.
    Returns combined DataFrame (or empty list if nothing).
    """
    tasks = _make_pair_tasks(upstream_ids, downstream_ids)
    if not tasks:
        return []

    total = len(tasks)
    all_results = []

    bar = None
    if tqdm is not None:
        bar = tqdm(total=total, desc="Path queries", unit="pair")

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = []
        for up_id, down_id in tasks:
            futs.append(
                ex.submit(
                    _query_one_pair,
                    up_id, down_id,
                    use_all_paths,
                    min_weight,
                    path_kwargs or {}
                )
            )

        for fut in as_completed(futs):
            try:
                df = fut.result()
                if df is not None and not df.empty:
                    all_results.append(df)
            except Exception as e:
                # Keep going; don’t kill the whole run
                print(f"[WARN] Pair query failed: {e}")
            finally:
                if bar is not None:
                    bar.update(1)

    if bar is not None:
        bar.close()

    return all_results
def _nt_class_from_value(v: str) -> str:
    """
    Map neuPrint neurotransmitter labels to 'exc', 'inh', or 'other/unknown'.
    Adjust this mapping if glutamate should be handled differently.
    """
    if v is None:
        return "unknown"
    s = str(v).strip().lower()
    if not s or s in {"none", "nan", "null"}:
        return "unknown"

    # common labels across datasets
    # inhibitory
    if "gaba" in s or "gly" in s:
        return "inh"

    # excitatory-ish (Drosophila: ACh often excit; Glu sometimes excit depending on system)
    if "ach" in s or "acetylcholine" in s:
        return "exc"
    if "glu" in s or "glutamate" in s:
        return "inh"

    # modulators / unclear
    if "dop" in s or "oct" in s or "oa" in s or "5ht" in s or "seroton" in s or "hist" in s:
        return "other"

    return "other"
def fetch_nt_for_bodyids(bodyIds, client=None):
    """
    Robustly fetch a transmitter label for each bodyId.
    Different datasets may store this under different property names.
    Several common fields are coalesced to the first non-null value.
    Returns a dict: {bodyId:int -> nt_value:str or None}
    """
    if client is None:
        client = get_male_cns_client()

    bodyIds = [int(x) for x in bodyIds]
    if not bodyIds:
        return {}

    id_list = "[" + ",".join(map(str, bodyIds)) + "]"

    # Try common property names; coalesce picks the first available.
    # (If none exist in the dataset, nt will be null.)
    cypher = f"""
    MATCH (n:Neuron)
    WHERE n.bodyId IN {id_list}
    RETURN
      n.bodyId AS bodyId,
      coalesce(
        n.consensusNt,
        n.predictedNt,
        n.predicted_nt,
        n.nt,
        n.neurotransmitter,
        n.transmitter
      ) AS nt
    """
    df = client.fetch_custom(cypher, format="pandas")
    out = {}
    if df is None or df.empty:
        return {int(b): None for b in bodyIds}

    for _, r in df.iterrows():
        out[int(r["bodyId"])] = r["nt"] if ("nt" in r and r["nt"] is not None) else None

    # ensure all requested IDs present
    for b in bodyIds:
        out.setdefault(int(b), None)
    return out
def fetch_nt_for_bodyids_chunked(bodyIds, client=None, chunk_size: int = 5000):
    """
    Chunked wrapper around fetch_nt_for_bodyids() to avoid giant IN[...] queries.
    Returns dict: {bodyId:int -> nt_value:str|None}
    """
    if client is None:
        client = get_male_cns_client()

    bodyIds = [int(x) for x in bodyIds if pd.notna(x)]
    if not bodyIds:
        return {}

    out = {}
    # chunk
    for i in range(0, len(bodyIds), int(chunk_size)):
        chunk = bodyIds[i:i+int(chunk_size)]
        out.update(fetch_nt_for_bodyids(chunk, client=client))
    return out
def ask_polarity_filters():
    """
    Ask for optional neurotransmitter polarity filters.
    Returns:
      start_exclude: set({'exc','inh'}) for upstream neurons
      pretarget_exclude: set({'exc','inh'}) for penultimate (pre-target) neuron
    """
    print("\n" + "=" * 72)
    print("Optional neurotransmitter filters")
    print("-" * 72)

    # 1) upstream/start filter
    start_exclude = set()
    ans = (input("Hide EXCITATORY upstream/start neurons? [y/N]: ").strip().lower() or "n")
    if ans == "y":
        start_exclude.add("exc")
    ans = (input("Hide INHIBITORY upstream/start neurons? [y/N]: ").strip().lower() or "n")
    if ans == "y":
        start_exclude.add("inh")

    # 2) penultimate (just before target) filter
    pretarget_exclude = set()
    ans = (input("Hide paths where the neuron JUST BEFORE the target is EXCITATORY? [y/N]: ").strip().lower() or "n")
    if ans == "y":
        pretarget_exclude.add("exc")
    ans = (input("Hide paths where the neuron JUST BEFORE the target is INHIBITORY? [y/N]: ").strip().lower() or "n")
    if ans == "y":
        pretarget_exclude.add("inh")

    return start_exclude, pretarget_exclude
def filter_upstream_by_nt(upstream_ids, exclude_classes, client=None):
    """
    Remove upstream neurons whose nt class is in exclude_classes.
    """
    if client is None:
        client = get_male_cns_client()

    if not exclude_classes:
        return upstream_ids

    nt_map = fetch_nt_for_bodyids(upstream_ids, client=client)
    keep = []
    dropped = 0
    for bid in upstream_ids:
        cls = _nt_class_from_value(nt_map.get(int(bid)))
        if cls in exclude_classes:
            dropped += 1
        else:
            keep.append(int(bid))

    print(f"[INFO] Upstream filter: kept {len(keep)}/{len(upstream_ids)} (dropped {dropped}) using exclude={sorted(exclude_classes)}")
    return keep
def filter_paths_by_pretarget_nt(paths_df, exclude_classes, client=None):
    """
    Drop entire paths if the penultimate neuron (node immediately before downstream target)
    has nt class in exclude_classes.
    """
    if client is None:
        client = get_male_cns_client()

    if paths_df is None or paths_df.empty or not exclude_classes:
        return paths_df

    # Identify penultimate bodyId for each (upstream,downstream,path)
    # Note: In neuprint shortest path tables, each row is a node along the path;
    # the last row is the downstream node; penultimate is row -2.
    penult_rows = []
    for key, g in paths_df.groupby(["upstream", "downstream", "path"]):
        g2 = g.sort_index()
        if len(g2) < 2:
            continue
        penult = g2.iloc[-2]  # node right before target
        penult_rows.append((key[0], key[1], key[2], int(penult["bodyId"])))

    if not penult_rows:
        return paths_df

    penult_ids = sorted(set(r[3] for r in penult_rows))
    nt_map = fetch_nt_for_bodyids(penult_ids, client=client)

    # Build a set of (up,down,path) to drop
    drop_keys = set()
    for up_id, down_id, path_id, penult_id in penult_rows:
        cls = _nt_class_from_value(nt_map.get(penult_id))
        if cls in exclude_classes:
            drop_keys.add((up_id, down_id, path_id))

    if not drop_keys:
        print(f"[INFO] Pre-target filter: no paths dropped using exclude={sorted(exclude_classes)}")
        return paths_df

    mask = []
    for _, r in paths_df.iterrows():
        mask.append((r["upstream"], r["downstream"], r["path"]) not in drop_keys)

    filtered = paths_df.loc[mask].copy()
    print(f"[INFO] Pre-target filter: dropped {len(drop_keys)} path(s); remaining rows {len(filtered)}/{len(paths_df)} using exclude={sorted(exclude_classes)}")
    return filtered


def run_pathfinding_option_26():
    """
    Run Choice 4.

    Interactively query paths in neuPrint, e.g.
    AN08B098 -> DNp01, possibly via intermediates.

    Adds:
      - parallel pair queries
      - optional exc/inh filtering:
          (1) exclude exc/inh START (upstream) neurons
          (2) exclude paths where the neuron JUST BEFORE the target is exc/inh
      - suppresses huge upstream/downstream ID list printing
    """
    client = get_male_cns_client()

    print("\n" + "=" * 80)
    print(f"OPTION 26: Pathfinding in neuPrint ({MALE_CNS_DATASET})")
    print("=" * 80)
    print("Goal example: Are AN08B098 neurons connected to DNp01 via intermediate neurons")
    print("              (e.g. AN08B098 → X → DNp01)?")
    print()

    # --------------------------
    # Choose neuron sets
    # --------------------------
    upstream_ids = _choose_neuron_set(
        "Select UPSTREAM neuron set (sources)",
        default_type="AN08B098"
    )
    if not upstream_ids:
        print("[ERROR] No upstream neurons selected. Aborting Option 26.")
        return

    downstream_ids = _choose_neuron_set(
        "Select DOWNSTREAM neuron set (targets)",
        default_type="DNp01"
    )
    if not downstream_ids:
        print("[ERROR] No downstream neurons selected. Aborting Option 26.")
        return

    # --------------------------
    # Optional exc/inh filtering
    # --------------------------
    start_exclude, pretarget_exclude = ask_polarity_filters()

    # Filter upstream set BEFORE expensive queries
    upstream_ids = filter_upstream_by_nt(upstream_ids, start_exclude, client=client)
    if not upstream_ids:
        print("[ERROR] Upstream set is empty after filtering. Aborting.")
        return

    # --------------------------
    # Numeric knobs
    # --------------------------
    try:
        min_weight_str = input("Minimum connection weight per hop [default 1]: ").strip()
        min_weight = int(min_weight_str) if min_weight_str else 1
    except ValueError:
        print("[WARN] Invalid min_weight, using 1.")
        min_weight = 1

    mode = input("Use shortest paths only (S) or all paths up to max length (A)? [S/a]: ").strip().lower()
    use_all_paths = (mode == "a")

    path_kwargs = {}
    if use_all_paths:
        max_len_str = input("Max path length in hops (optional; empty for no limit): ").strip()
        if max_len_str:
            try:
                path_kwargs["max_path_length"] = int(max_len_str)
            except ValueError:
                print("[WARN] Invalid max path length; ignoring and using no limit.")

    # --------------------------
    # Run parallel path queries
    # --------------------------
    print("\n[INFO] Running path queries in parallel ...\n")

    all_results = run_pair_path_queries_parallel(
        upstream_ids=upstream_ids,
        downstream_ids=downstream_ids,
        use_all_paths=use_all_paths,
        min_weight=min_weight,
        path_kwargs=path_kwargs,
        max_workers=48,
    )

    if not all_results:
        print("\n[RESULT] No paths found for any upstream/downstream pairs with the given settings.")
        return

    paths_df = pd.concat(all_results, ignore_index=True)
    # --------------------------
    # Add predictedNT to EACH ROW (node-level)
    # --------------------------
    unique_nodes = paths_df["bodyId"].dropna().astype(int).unique().tolist()
    print(f"[INFO] Fetching predictedNT for {len(unique_nodes):,} unique neurons in paths table ...")
    
    nt_map = fetch_nt_for_bodyids_chunked(unique_nodes, client=client, chunk_size=5000)
    
    # node-level NT (for the neuron represented by this row)
    paths_df["predictedNT"] = paths_df["bodyId"].apply(lambda x: nt_map.get(int(x)) if pd.notna(x) else None)
    paths_df["predictedNT_class"] = paths_df["predictedNT"].apply(_nt_class_from_value)

    # Optional: also include upstream/downstream NT on every row
    unique_endpoints = pd.unique(pd.concat([paths_df["upstream"], paths_df["downstream"]], ignore_index=True)).tolist()
    unique_endpoints = [int(x) for x in unique_endpoints if pd.notna(x)]
    
    ep_nt_map = fetch_nt_for_bodyids_chunked(unique_endpoints, client=client, chunk_size=5000)
    
    paths_df["upstream_predictedNT"] = paths_df["upstream"].apply(lambda x: ep_nt_map.get(int(x)) if pd.notna(x) else None)
    paths_df["downstream_predictedNT"] = paths_df["downstream"].apply(lambda x: ep_nt_map.get(int(x)) if pd.notna(x) else None)
    
    paths_df["upstream_predictedNT_class"] = paths_df["upstream_predictedNT"].apply(_nt_class_from_value)
    paths_df["downstream_predictedNT_class"] = paths_df["downstream_predictedNT"].apply(_nt_class_from_value)


    # Apply "pre-target exc/inh" filter after path table construction.
    paths_df = filter_paths_by_pretarget_nt(paths_df, pretarget_exclude, client=client)
    if paths_df is None or paths_df.empty:
        print("\n[RESULT] All paths were filtered out by the pre-target exc/inh filter.")
        return

    print("\n[RESULT] Combined path table shape:", paths_df.shape)
    print()

    # --------------------------
    # Intermediates summary
    # --------------------------
    intermediates = []
    for (up_id, down_id, path_id), g in paths_df.groupby(["upstream", "downstream", "path"]):
        g_sorted = g.sort_index()
        if len(g_sorted) > 2:
            mids = g_sorted.iloc[1:-1]  # drop first/last
            intermediates.append(mids)

    if intermediates:
        inter_df = pd.concat(intermediates, ignore_index=True)
        print("[INFO] Unique intermediate neuron types (excluding endpoints):")
        print(inter_df[["type", "bodyId"]].drop_duplicates().head(30))
    else:
        print("[INFO] No intermediate neurons found (paths are direct or trivial).")

    # --------------------------
    # Example paths + 2-hop summary
    # --------------------------
    print("\n[INFO] Example paths:")
    _summarize_paths_df(paths_df)

    print("\n[INFO] Detailed 2-hop (three-neuron) paths summary:")
    _summarize_two_hop_paths(paths_df)

    # --------------------------
    # Save CSV
    # --------------------------
    save = input("\nSave combined paths to CSV? [y/N]: ").strip().lower()
    if save == "y":
        dataset_slug = MALE_CNS_DATASET.replace(":", "_")
        default_name = f"outputs/paths_{upstream_ids[0]}_to_{downstream_ids[0]}_{dataset_slug}.csv"
        out_name = input(f"CSV filename [{default_name}]: ").strip() or default_name
        out_path = phase1_output_path(out_name)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        paths_df.to_csv(out_path, index=False)
        print(f"[OK] Saved {len(paths_df)} rows to '{out_path}'.")
    else:
        print("[INFO] Not saving CSV.")

    # --------------------------
    # OPTIONAL: MN-like targets within hops of the FIRST upstream neuron
    # --------------------------
    print("\n[INFO] Scanning MN-like targets within 1..3 hops of the first upstream neuron ...")
    src = upstream_ids[0]

    mn_df = find_mn_within_hops_inline(
        source_bodyId=src,
        token="MN",
        max_hops=3,
        min_weight=min_weight,
        match_on="either",
        limit_paths_per_target=3,
        client=client
    )

    if mn_df is None or mn_df.empty:
        print("[RESULT] No MN-like targets found within 1..3 hops.")
    else:
        print("[RESULT] MN-like targets reachable within 1..3 hops:")
        print(mn_df.head(30))


# =============================================================================
# Choice 5 glia-volume export
# =============================================================================

def export_glia_volume_manc_v121(out_base: str = "outputs/manc_v1.2.1_glia_volume"):
    """
    Run Choice 5: export a glia-focused volume table from manc:v1.2.1.

    Writes:
      - <out_base>.csv
      - <out_base>.parquet

    Uses strict class='glia' first, then automatically falls back to a broader
    glia match if strict returns zero rows.
    """
    import pandas as pd
    import numpy as np
    import os
    from pathlib import Path
    from neuprint.skeleton import heal_skeleton, upsample_skeleton

    target_dataset = "manc:v1.2.1"
    target_server = "https://neuprint.janelia.org"

    global manc_v121_client
    try:
        client_mc = manc_v121_client
    except NameError:
        token = None
        for cname in ("client", "navis_client"):
            cobj = globals().get(cname)
            if cobj is None:
                continue
            token = getattr(cobj, "token", None) or getattr(cobj, "_token", None)
            if token:
                break
        if not token:
            token = get_neuprint_token(required=False)

        manc_v121_client = neu.Client(
            target_server,
            dataset=target_dataset,
            token=token,
        )
        client_mc = manc_v121_client

    q_strict = """
    MATCH (n:Neuron)
    WHERE toLower(coalesce(n.class, '')) = 'glia'
    RETURN
      n.bodyId      AS bodyId,
      n.type        AS type,
      n.instance    AS instance,
      n.class       AS class_,
      n.superclass  AS superclass,
      n.status      AS status,
      n.pre         AS pre,
      n.post        AS post,
      n.size        AS size
    ORDER BY size DESC
    """

    q_fallback = """
    MATCH (n:Neuron)
    WHERE toLower(coalesce(n.class, '')) = 'glia'
       OR toLower(coalesce(n.superclass, '')) = 'glia'
       OR toLower(coalesce(n.type, '')) CONTAINS 'glia'
       OR toLower(coalesce(n.instance, '')) CONTAINS 'glia'
    RETURN
      n.bodyId      AS bodyId,
      n.type        AS type,
      n.instance    AS instance,
      n.class       AS class_,
      n.superclass  AS superclass,
      n.status      AS status,
      n.pre         AS pre,
      n.post        AS post,
      n.size        AS size
    ORDER BY size DESC
    """

    df = client_mc.fetch_custom(q_strict)
    mode_used = "strict class='glia'"
    if df is None or df.empty:
        print("[WARN] Strict class='glia' returned 0 rows; trying broader glia match...")
        df = client_mc.fetch_custom(q_fallback)
        mode_used = "fallback(class/superclass/type/instance glia)"

    if df is None:
        df = pd.DataFrame(columns=["bodyId", "type", "instance", "class_", "superclass", "status", "pre", "post", "size"])

    # keep native neuPrint size and add bbox-based geometry metrics from skeleton XYZ
    if "size" in df.columns and "n_size" not in df.columns:
        df = df.rename(columns={"size": "n_size"})

    bbox_x, bbox_y, bbox_z, bbox_diag, bbox_vol, bbox_err = [], [], [], [], [], []
    upsample_nm = 2000.0
    n_rows = len(df)
    if n_rows:
        print(f"[INFO] Computing bbox metrics for {n_rows} glia neuron(s) from skeleton XYZ...")

    for i, bid in enumerate(df.get("bodyId", pd.Series(dtype='Int64')).tolist(), start=1):
        try:
            skel = client_mc.fetch_skeleton(int(bid), heal=False, format="pandas")
            skel = heal_skeleton(skel, max_distance=np.inf, root_parent=-1)
            skel = upsample_skeleton(skel, max_segment_length=upsample_nm)
            xyz_nm = skel[["x", "y", "z"]].astype(float).to_numpy()
            if xyz_nm.size == 0:
                raise RuntimeError("empty xyz")
            lo = np.min(xyz_nm, axis=0)
            hi = np.max(xyz_nm, axis=0)
            span_um = (hi - lo) / 1000.0
            x_um, y_um, z_um = float(span_um[0]), float(span_um[1]), float(span_um[2])
            bbox_x.append(x_um)
            bbox_y.append(y_um)
            bbox_z.append(z_um)
            bbox_diag.append(float(np.linalg.norm(span_um)))
            bbox_vol.append(float(np.prod(span_um)))
            bbox_err.append("")
        except Exception as e:
            bbox_x.append(np.nan)
            bbox_y.append(np.nan)
            bbox_z.append(np.nan)
            bbox_diag.append(np.nan)
            bbox_vol.append(np.nan)
            bbox_err.append(str(e))

        if i % 25 == 0 or i == n_rows:
            print(f"[INFO] bbox progress: {i}/{n_rows}")

    df["bbox_x_um"] = bbox_x
    df["bbox_y_um"] = bbox_y
    df["bbox_z_um"] = bbox_z
    df["bbox_diag_um"] = bbox_diag
    df["bbox_volume_um3"] = bbox_vol
    df["bbox_error"] = bbox_err
    df["size_metric"] = "bbox_volume_um3"
    df["size_value"] = df["bbox_volume_um3"]

    out_path = phase1_output_path(out_base)
    if out_path.suffix.lower() in {'.csv', '.parquet'}:
        out_path = out_path.with_suffix('')

    csv_out = out_path.with_suffix('.csv')
    pq_out  = out_path.with_suffix('.parquet')

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_out, index=False)
    df.to_parquet(pq_out, index=False)

    print(f"[INFO] Dataset: {target_dataset}")
    print(f"[INFO] Filter mode: {mode_used}")
    print(f"[INFO] Rows exported: {len(df)}")
    print(f"[save] CSV     -> {csv_out}")
    print(f"[save] Parquet -> {pq_out}")

    return df


# =============================================================================
# Choice 6 dataset-label coverage
# =============================================================================

def report_dataset_label_coverage(client):
    """
    Run Choice 6: count labeled vs unlabeled Neuron nodes in the active dataset.

    Definition used:
      - labeled   = non-empty `type` OR non-empty `instance`
      - unlabeled = both `type` and `instance` empty/missing
    """
    dataset_name = getattr(client, "dataset", None) or os.environ.get("NEUPRINT_DATASET", "(unknown)")

    q_counts = """
    MATCH (n:Neuron)
    WHERE n.bodyId IS NOT NULL
    WITH trim(coalesce(n.type, '')) AS t,
         trim(coalesce(n.instance, '')) AS i
    RETURN count(*) AS total,
           sum(CASE WHEN t <> '' OR i <> '' THEN 1 ELSE 0 END) AS labeled,
           sum(CASE WHEN t = '' AND i = '' THEN 1 ELSE 0 END) AS unlabeled
    """

    counts_df = client.fetch_custom(q_counts)
    if counts_df is None or counts_df.empty:
        print("[INFO] No Neuron rows returned for this dataset.")
        return {"dataset": dataset_name, "total": 0, "labeled": 0, "unlabeled": 0}

    total = int(counts_df.iloc[0].get("total", 0) or 0)
    labeled = int(counts_df.iloc[0].get("labeled", 0) or 0)
    unlabeled = int(counts_df.iloc[0].get("unlabeled", 0) or 0)

    pct = (lambda n: (100.0 * n / total) if total else 0.0)

    print("\n===== Dataset Label Coverage =====")
    print(f"Dataset: {dataset_name}")
    print("Definition: labeled = has `type` or `instance`; unlabeled = missing both")
    print(f"Total neurons : {total:,}")
    print(f"Labeled       : {labeled:,} ({pct(labeled):.2f}%)")
    print(f"Unlabeled     : {unlabeled:,} ({pct(unlabeled):.2f}%)")

    show_samples = (input("Show up to 20 STRICT unlabeled bodyIds? [y/N]: ").strip().lower() or "n")
    if show_samples == "y" and unlabeled > 0:
        ids = fetch_unlabeled_body_ids(client, strict=True)[:20]
        if ids:
            print(f"Sample STRICT unlabeled bodyIds ({len(ids)}): {ids}")
        else:
            print("[INFO] No strict unlabeled sample IDs found with the current filters.")

    return {
        "dataset": dataset_name,
        "total": total,
        "labeled": labeled,
        "unlabeled": unlabeled,
    }


# =============================================================================
# Choice 7 proximity scan
# =============================================================================

def find_neurons_near_reference_skeletons(client):
    """
    Run Choice 7.

    From a master metadata CSV, find neurons whose exported SWC is within a
    given Euclidean distance (um) of reference skeletons. The result CSV is
    written beside the master metadata CSV.
    """
    from pathlib import Path
    from scipy.spatial import cKDTree

    default_candidates = [
        Path(UNLABELED_EXPORT_ROOT) / "unlabeled_strict_master_metadata.csv",
        Path(UNLABELED_EXPORT_ROOT) / "unlabeled_unstrict_master_metadata.csv",
    ]
    default_master = next((p for p in default_candidates if p.exists()), default_candidates[0])

    master_in = input(f"Master CSV path [{default_master}]: ").strip()
    master_csv = phase1_output_path(master_in) if master_in else default_master
    if not master_csv.exists():
        print(f"[ERROR] Master CSV not found: {master_csv}")
        return pd.DataFrame()

    refs_in = input("Reference bodyIds (comma-separated) [10000,10002]: ").strip() or "10000,10002"
    try:
        ref_ids = [int(x) for x in refs_in.replace(" ", "").split(",") if x]
    except Exception:
        print(f"[ERROR] Invalid reference ID list: {refs_in}")
        return pd.DataFrame()
    if not ref_ids:
        print("[ERROR] No reference bodyIds provided.")
        return pd.DataFrame()

    radius_in = input("Distance threshold in um [5.0]: ").strip() or "5.0"
    try:
        radius_um = float(radius_in)
    except Exception:
        print(f"[ERROR] Invalid radius: {radius_in}")
        return pd.DataFrame()

    mode_in = (input("Require proximity to BOTH refs or EITHER ref? [either/both] [either]: ").strip().lower() or "either")
    require_both = (mode_in == "both")

    print(f"[INFO] Loading master CSV: {master_csv}")
    df = pd.read_csv(master_csv)
    if df.empty or "bodyId" not in df.columns:
        print("[ERROR] Master CSV is empty or missing bodyId column.")
        return pd.DataFrame()

    def _load_points_um_from_swc(path_val):
        if path_val is None or (isinstance(path_val, float) and np.isnan(path_val)):
            return None
        sp = Path(str(path_val)).expanduser()
        if not sp.exists():
            return None
        try:
            _, recs = _parse_swc(str(sp))
            if not recs:
                return None
            pts = np.array([[r[2], r[3], r[4]] for r in recs], dtype=float)
            if pts.size == 0:
                return None
            return pts
        except Exception:
            return None

    def _fetch_reference_points_um(ref_bid):
        try:
            skel = client.fetch_skeleton(int(ref_bid), heal=False, format="pandas")
            if skel is None or len(skel) == 0:
                return None

            # Keep consistent with Option-20 processing
            skel = heal_skeleton(skel, max_distance=np.inf, root_parent=-1)
            skel = upsample_skeleton(skel, max_segment_length=UPSAMPLE_NM)

            pts_nm = skel[["x", "y", "z"]].astype(float).to_numpy()
            if pts_nm.size == 0:
                return None
            return pts_nm / 1000.0  # um
        except Exception as e:
            print(f"[WARN] reference {ref_bid} fetch failed: {e}")
            return None

    ref_trees = {}
    for ref_id in ref_ids:
        pts = _fetch_reference_points_um(ref_id)
        if pts is None:
            print(f"[WARN] No reference skeleton points for {ref_id}; skipping this ref")
            continue
        ref_trees[ref_id] = cKDTree(pts)

    if not ref_trees:
        print("[ERROR] Could not build any reference trees.")
        return pd.DataFrame()

    ref_ids = sorted(ref_trees.keys())
    print(f"[INFO] Reference IDs used: {ref_ids}")
    print(f"[INFO] Distance threshold: {radius_um:.3f} um | mode={'both' if require_both else 'either'}")

    if _tqdm_opt20 is not None:
        pbar = _tqdm_opt20(total=len(df), desc="Proximity scan", unit="neuron", position=0, leave=True)
    else:
        pbar = None

    dist_cols = {rid: f"dist_to_{rid}_um" for rid in ref_ids}
    all_dist = {rid: [] for rid in ref_ids}
    near_any = []
    near_all = []
    missing_swc = []

    try:
        for i, row in enumerate(df.itertuples(index=False), start=1):
            rowd = row._asdict()
            swc_path = rowd.get("healed_final_swc", None)
            pts = _load_points_um_from_swc(swc_path)

            if pts is None:
                missing_swc.append(True)
                for rid in ref_ids:
                    all_dist[rid].append(np.nan)
                near_any.append(False)
                near_all.append(False)
            else:
                missing_swc.append(False)
                per_ref_near = []
                for rid in ref_ids:
                    dists, _ = ref_trees[rid].query(pts, k=1)
                    min_d = float(np.min(dists)) if len(dists) else np.nan
                    all_dist[rid].append(min_d)
                    per_ref_near.append(np.isfinite(min_d) and (min_d <= radius_um))

                near_any.append(any(per_ref_near))
                near_all.append(all(per_ref_near))

            if pbar is not None:
                pbar.update(1)
            elif i % 200 == 0 or i == len(df):
                print(f"[INFO] Proximity progress: {i}/{len(df)}")
    finally:
        if pbar is not None:
            pbar.close()

    out = df.copy()
    for rid in ref_ids:
        out[dist_cols[rid]] = all_dist[rid]
        out[f"near_{rid}"] = out[dist_cols[rid]] <= radius_um
    out["near_either_ref"] = near_any
    out["near_all_refs"] = near_all
    out["missing_swc"] = missing_swc
    out["proximity_radius_um"] = float(radius_um)
    out["proximity_ref_ids"] = ",".join(map(str, ref_ids))

    hit_mask = out["near_all_refs"] if require_both else out["near_either_ref"]
    hits = out[hit_mask].copy()

    stem = master_csv.stem
    mode_label = "both" if require_both else "either"
    refs_label = "_".join(map(str, ref_ids))
    out_csv = master_csv.with_name(f"{stem}__near_{refs_label}__r{radius_um:g}um__{mode_label}.csv")
    hits.to_csv(out_csv, index=False)

    print(f"[INFO] Total candidates: {len(out):,}")
    print(f"[INFO] Missing SWC path/points: {int(np.sum(out['missing_swc'])):,}")
    print(f"[INFO] Hits ({mode_label}): {len(hits):,}")
    print(f"[save] Proximity CSV -> {out_csv}")

    return hits
