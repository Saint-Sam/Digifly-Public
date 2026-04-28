from __future__ import annotations

import argparse
import builtins
import copy
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd


TOOLS_DIR = Path(__file__).resolve().parent
APP_ROOT = TOOLS_DIR.parent
PHASE2_ROOT = APP_ROOT.parent.parent

DEFAULT_GLIA_NOTEBOOK = Path(
    os.environ.get(
        'DIGIFLY_GLIA_SIMULATION_NOTEBOOK',
        APP_ROOT / 'notebooks' / 'glia_simulation.ipynb',
    )
)
DEFAULT_PHASE2_NEURON_ROOT = Path(os.environ.get('DIGIFLY_PHASE2_NEURON_ROOT', PHASE2_ROOT))
DEFAULT_PHASE2_ARBOR_ROOT = Path(
    os.environ.get('DIGIFLY_PHASE2_ARBOR_ROOT', PHASE2_ROOT / 'backends' / 'arbor_phase2')
)
DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get(
        'DIGIFLY_GLIA_COMPARE_OUTPUT_ROOT',
        APP_ROOT / 'notebooks' / 'debug' / 'outputs' / 'glia_neuron_vs_arbor_compare',
    )
)


def _ts_utc() -> str:
    return pd.Timestamp.utcnow().strftime('%Y%m%dT%H%M%SZ')


def _exec_notebook_code(src: str, ns: dict, nb_path: Path, code_cell_idx: int) -> None:
    filename = f"{nb_path}::code_cell_{code_cell_idx}"
    compiled = compile(src, filename, 'exec')
    exec(compiled, ns)


def _rewrite_source_for_backend(src: str, backend: str) -> str:
    if backend.lower() != 'arbor':
        return src
    # Minimal source rewrite for glia_simulation notebook loader cell(s) that hard-import NEURON build modules.
    return src.replace('digifly.phase2.neuron_build.', 'digifly.phase2.arbor_build.')


def load_glia_namespace(
    nb_path: str | Path,
    *,
    backend: str,
    phase2_root: str | Path,
    stop_after_run_helpers: bool = True,
    skip_selector_launcher_cell: bool = True,
) -> dict:
    nb_path = Path(nb_path).expanduser().resolve()
    phase2_root = Path(phase2_root).expanduser().resolve()
    nb = json.loads(nb_path.read_text(encoding='utf-8'))
    cells = nb.get('cells', [])
    code_cells = [c for c in cells if c.get('cell_type') == 'code']

    # Isolate module resolution per backend load.
    os.environ['DIGIFLY_PHASE2_ROOT'] = str(phase2_root)

    # Purge previously imported digifly modules to avoid cross-backend contamination in-process.
    for name in list(sys.modules.keys()):
        if name == 'digifly' or name.startswith('digifly.'):
            sys.modules.pop(name, None)

    ns = {
        '__name__': '__main__',
        '__file__': str(nb_path),
        '__builtins__': builtins.__dict__,
    }

    truncated_run_helper_cell = False
    for code_idx, cell in enumerate(code_cells):
        src = ''.join(cell.get('source', []))
        if not src.strip():
            continue
        if skip_selector_launcher_cell and ('RUN_SELECTOR_FLOW = False' in src and 'SELECTOR_TARGET_NEURON_IDS' in src):
            continue

        src = _rewrite_source_for_backend(src, backend)

        if 'def _run_one_scenario_from_overrides' in src:
            marker = 'RUN_OUTPUTS_BY_SCENARIO = {}'
            if marker not in src:
                raise RuntimeError('Could not find run helper truncation marker; glia_simulation cell format changed.')
            helper_src = src.split(marker, 1)[0]
            _exec_notebook_code(helper_src, ns, nb_path, code_idx)
            truncated_run_helper_cell = True
            if stop_after_run_helpers:
                break
            continue

        if 'pd.read_csv(Path(out_dir) / "records.csv")' in src:
            continue

        _exec_notebook_code(src, ns, nb_path, code_idx)

    required = [
        '_selector_json_for_scenario',
        'refresh_glia_spec_from_selector',
        '_run_one_scenario_from_overrides',
        'USER_OVERRIDES',
        'RUN_ID',
        'VIP_ROOT',
        'GLIA_RUNS_ROOT',
    ]
    missing = [k for k in required if k not in ns]
    if missing:
        raise RuntimeError(f'glia_simulation loader missing expected symbols for {backend}: {missing}')
    if not truncated_run_helper_cell:
        raise RuntimeError('Did not find _run_one_scenario_from_overrides in glia_simulation.ipynb.')
    return ns


def _resolve_selector_json(ns: dict, explicit: str | None, scenario: str) -> str:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f'--selector-json not found: {p}')
        return str(p)

    cand = ns.get('GLIA_SELECTOR_OUTPUT_JSON')
    if cand:
        p = Path(str(cand)).expanduser().resolve()
        if p.exists():
            return str(p)

    fn = ns.get('_selector_json_for_scenario')
    if callable(fn):
        try:
            p = Path(str(fn(str(scenario)))).expanduser().resolve()
            if p.exists():
                return str(p)
            raise FileNotFoundError(
                f'Selector JSON does not exist: {p}. Run the selector flow in glia_simulation.ipynb first or pass --selector-json.'
            )
        except Exception as e:
            raise RuntimeError(f'Could not resolve selector JSON path from glia notebook namespace: {e}')

    raise RuntimeError('No selector JSON path available in glia namespace and --selector-json was not provided.')


def _normalize_spikes_df(p: Path) -> pd.DataFrame:
    p1 = p / 'spike_times.csv'
    p2 = p / 'spikes.csv'
    if p1.exists():
        df = pd.read_csv(p1)
        if {'neuron_id', 'spike_time_ms'}.issubset(df.columns):
            out = df[['neuron_id', 'spike_time_ms']].copy()
            out['neuron_id'] = pd.to_numeric(out['neuron_id'], errors='coerce').astype('Int64')
            out['spike_time_ms'] = pd.to_numeric(out['spike_time_ms'], errors='coerce')
            return out.dropna().astype({'neuron_id': int, 'spike_time_ms': float})
    if p2.exists():
        df = pd.read_csv(p2)
        if {'nid', 't_ms'}.issubset(df.columns):
            out = df.rename(columns={'nid': 'neuron_id', 't_ms': 'spike_time_ms'})[['neuron_id', 'spike_time_ms']].copy()
            out['neuron_id'] = pd.to_numeric(out['neuron_id'], errors='coerce').astype('Int64')
            out['spike_time_ms'] = pd.to_numeric(out['spike_time_ms'], errors='coerce')
            return out.dropna().astype({'neuron_id': int, 'spike_time_ms': float})
    return pd.DataFrame(columns=['neuron_id', 'spike_time_ms'])


def _load_records_df(p: Path) -> pd.DataFrame:
    f = p / 'records.csv'
    if not f.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(f)
    except Exception:
        return pd.DataFrame()


def _greedy_match_count(a: np.ndarray, b: np.ndarray, tol_ms: float) -> int:
    i = j = m = 0
    while i < len(a) and j < len(b):
        da = float(a[i]); db = float(b[j])
        d = db - da
        if abs(d) <= tol_ms:
            m += 1
            i += 1
            j += 1
        elif da < db:
            i += 1
        else:
            j += 1
    return m


def compare_spikes(neuron_out: Path, arbor_out: Path, tol_ms: float = 0.25) -> tuple[pd.DataFrame, pd.DataFrame]:
    ndf = _normalize_spikes_df(neuron_out)
    adf = _normalize_spikes_df(arbor_out)

    n_groups = {int(n): np.sort(g['spike_time_ms'].to_numpy(float)) for n, g in ndf.groupby('neuron_id')} if not ndf.empty else {}
    a_groups = {int(n): np.sort(g['spike_time_ms'].to_numpy(float)) for n, g in adf.groupby('neuron_id')} if not adf.empty else {}
    all_ids = sorted(set(n_groups) | set(a_groups))

    rows = []
    total_matches = 0
    total_n = 0
    total_a = 0
    for nid in all_ids:
        ns = n_groups.get(nid, np.array([], dtype=float))
        ars = a_groups.get(nid, np.array([], dtype=float))
        m = _greedy_match_count(ns, ars, tol_ms=tol_ms)
        total_matches += int(m)
        total_n += int(len(ns))
        total_a += int(len(ars))
        rows.append({
            'neuron_id': int(nid),
            'neuron_spike_count': int(len(ns)),
            'arbor_spike_count': int(len(ars)),
            'count_delta_arbor_minus_neuron': int(len(ars) - len(ns)),
            'neuron_first_spike_ms': (float(ns[0]) if len(ns) else np.nan),
            'arbor_first_spike_ms': (float(ars[0]) if len(ars) else np.nan),
            'first_spike_delta_ms': (float(ars[0] - ns[0]) if len(ns) and len(ars) else np.nan),
            'matched_spikes_tol_ms': int(m),
            'match_recall_vs_neuron': (float(m) / float(len(ns)) if len(ns) else np.nan),
            'match_precision_vs_arbor': (float(m) / float(len(ars)) if len(ars) else np.nan),
        })

    recall = (float(total_matches) / float(total_n)) if total_n else np.nan
    precision = (float(total_matches) / float(total_a)) if total_a else np.nan
    if np.isfinite(recall) and np.isfinite(precision) and (recall + precision) > 0:
        f1 = 2.0 * recall * precision / (recall + precision)
    else:
        f1 = np.nan

    summary = pd.DataFrame([{
        'tol_ms': float(tol_ms),
        'neuron_total_spikes': int(total_n),
        'arbor_total_spikes': int(total_a),
        'total_spike_delta_arbor_minus_neuron': int(total_a - total_n),
        'matched_spikes': int(total_matches),
        'recall_vs_neuron': recall,
        'precision_vs_arbor': precision,
        'f1': f1,
        'neurons_with_spikes_neuron': int(ndf['neuron_id'].nunique()) if not ndf.empty else 0,
        'neurons_with_spikes_arbor': int(adf['neuron_id'].nunique()) if not adf.empty else 0,
    }])
    return summary, pd.DataFrame(rows)


def _corr_safe(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return np.nan
    if np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return np.nan
    try:
        return float(np.corrcoef(a, b)[0, 1])
    except Exception:
        return np.nan


def compare_records(neuron_out: Path, arbor_out: Path, t_round_decimals: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    ndf = _load_records_df(neuron_out)
    adf = _load_records_df(arbor_out)
    if ndf.empty or adf.empty or 't_ms' not in ndf.columns or 't_ms' not in adf.columns:
        summary = pd.DataFrame([{
            'status': 'missing_records',
            'neuron_records_exists': bool(not ndf.empty),
            'arbor_records_exists': bool(not adf.empty),
            'common_trace_columns': 0,
        }])
        return summary, pd.DataFrame()

    n = ndf.copy()
    a = adf.copy()
    n['__t_key__'] = pd.to_numeric(n['t_ms'], errors='coerce').round(t_round_decimals)
    a['__t_key__'] = pd.to_numeric(a['t_ms'], errors='coerce').round(t_round_decimals)
    n = n.dropna(subset=['__t_key__']).groupby('__t_key__', as_index=False).mean(numeric_only=True)
    a = a.dropna(subset=['__t_key__']).groupby('__t_key__', as_index=False).mean(numeric_only=True)

    merge = n.merge(a, on='__t_key__', suffixes=('_neuron', '_arbor'))
    if merge.empty:
        summary = pd.DataFrame([{
            'status': 'no_time_overlap',
            'neuron_samples': int(len(n)),
            'arbor_samples': int(len(a)),
            'overlap_samples': 0,
            'common_trace_columns': 0,
        }])
        return summary, pd.DataFrame()

    neuron_cols = [c for c in ndf.columns if c != 't_ms']
    arbor_cols = [c for c in adf.columns if c != 't_ms']
    common = sorted(set(neuron_cols) & set(arbor_cols))

    rows = []
    for col in common:
        cn = f'{col}_neuron'
        ca = f'{col}_arbor'
        if cn not in merge.columns or ca not in merge.columns:
            continue
        x = pd.to_numeric(merge[cn], errors='coerce').to_numpy(float)
        y = pd.to_numeric(merge[ca], errors='coerce').to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        if not m.any():
            continue
        xx = x[m]
        yy = y[m]
        d = yy - xx
        rows.append({
            'signal': str(col),
            'n_overlap': int(len(xx)),
            'mae': float(np.mean(np.abs(d))),
            'rmse': float(np.sqrt(np.mean(d * d))),
            'max_abs_err': float(np.max(np.abs(d))),
            'mean_delta_arbor_minus_neuron': float(np.mean(d)),
            'corr': _corr_safe(xx, yy),
            'neuron_min': float(np.min(xx)),
            'neuron_max': float(np.max(xx)),
            'arbor_min': float(np.min(yy)),
            'arbor_max': float(np.max(yy)),
        })

    detail = pd.DataFrame(rows)
    summary = pd.DataFrame([{
        'status': 'ok',
        'neuron_samples': int(len(n)),
        'arbor_samples': int(len(a)),
        'overlap_samples': int(len(merge)),
        'common_trace_columns': int(len(detail)),
        'mean_mae': (float(detail['mae'].mean()) if not detail.empty else np.nan),
        'mean_rmse': (float(detail['rmse'].mean()) if not detail.empty else np.nan),
        'median_rmse': (float(detail['rmse'].median()) if not detail.empty else np.nan),
        'mean_corr': (float(detail['corr'].mean()) if ('corr' in detail.columns and not detail.empty) else np.nan),
    }])
    return summary, detail


def _numeric_row_diff(neuron_row: dict, arbor_row: dict) -> pd.DataFrame:
    keys = sorted(set(neuron_row) & set(arbor_row))
    rows = []
    for k in keys:
        nv = neuron_row.get(k)
        av = arbor_row.get(k)
        # Bool-valued metrics are categorical here; don't subtract them.
        if isinstance(nv, (bool, np.bool_)) or isinstance(av, (bool, np.bool_)):
            continue
        n_num = pd.to_numeric(pd.Series([nv]), errors='coerce').iloc[0]
        a_num = pd.to_numeric(pd.Series([av]), errors='coerce').iloc[0]
        if pd.notna(n_num) or pd.notna(a_num):
            rows.append({'metric': k, 'neuron': n_num, 'arbor': a_num, 'delta_arbor_minus_neuron': a_num - n_num})
    return pd.DataFrame(rows)


def _run_subprocess(payload: dict) -> dict:
    worker_python = str(payload.get('worker_python') or sys.executable)
    cmd = [worker_python, str(Path(__file__).resolve()), '--worker-payload-json', json.dumps(payload)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    stdout = proc.stdout or ''
    stderr = proc.stderr or ''
    result: dict[str, Any]
    try:
        # worker prints exactly one JSON object
        result = json.loads(stdout.strip().splitlines()[-1]) if stdout.strip() else {}
    except Exception:
        result = {
            'ok': False,
            'error': 'worker_output_parse_failed',
            'stdout': stdout,
            'stderr': stderr,
        }
    result.setdefault('ok', False)
    result['returncode'] = int(proc.returncode)
    if stdout.strip():
        result['stdout_tail'] = stdout[-8000:]
    if stderr.strip():
        result['stderr_tail'] = stderr[-8000:]
    return result


def _worker_main(payload: dict) -> int:
    backend = str(payload['backend']).strip().lower()
    phase2_root = Path(payload['phase2_root']).expanduser().resolve()
    glia_notebook = Path(payload['glia_notebook']).expanduser().resolve()
    run_output_root = Path(payload['run_output_root']).expanduser().resolve()
    run_output_root.mkdir(parents=True, exist_ok=True)

    # Headless safety flags; harmless if not used by local NEURON build.
    os.environ.setdefault('MPLBACKEND', 'Agg')
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    os.environ.setdefault('NEURON_MODULE_OPTIONS', '-nogui')

    try:
        t0 = time.perf_counter()
        ns = load_glia_namespace(glia_notebook, backend=backend, phase2_root=phase2_root)
        selector_json = _resolve_selector_json(ns, payload.get('selector_json'), scenario=str(payload.get('scenario_selector_key', 'single')))

        glia_state = str(payload.get('glia_state', 'inherit')).strip().lower()
        if glia_state not in {'inherit', 'on', 'off'}:
            raise ValueError(f'Unsupported glia_state={glia_state}')
        if glia_state == 'on':
            ns['GLIA_LOSS_ENABLED'] = True
            ns['RUN_STATE_TAG'] = 'glia_on'
        elif glia_state == 'off':
            ns['GLIA_LOSS_ENABLED'] = False
            ns['RUN_STATE_TAG'] = 'glia_off'

        if payload.get('force_override_enabled') is not None:
            ns['GLIA_FORCE_KO_OVERRIDE_ENABLED'] = bool(payload['force_override_enabled'])
        if payload.get('ko_mM') is not None:
            ns['GLIA_FORCE_KO_mM'] = float(payload['ko_mM'])

        user_overrides = copy.deepcopy(ns['USER_OVERRIDES'])
        if backend == 'arbor':
            arbor_cfg = dict(user_overrides.get('arbor') or {})
            if payload.get('arbor_force_native') is not None:
                arbor_cfg['run_native'] = bool(payload['arbor_force_native'])
            if payload.get('arbor_native_strict') is not None:
                arbor_cfg['native_strict'] = bool(payload['arbor_native_strict'])
            if payload.get('arbor_rect_gap_policy') is not None:
                arbor_cfg['native_rectifying_gap_policy'] = str(payload['arbor_rect_gap_policy'])
            if payload.get('arbor_gap_exclude_nids') is not None:
                arbor_cfg['native_gap_exclude_nids'] = payload.get('arbor_gap_exclude_nids')
            if arbor_cfg:
                user_overrides['arbor'] = arbor_cfg
        user_overrides['run_id'] = str(payload['run_id'])
        user_overrides['runs_root'] = str(run_output_root)
        # Keep notebook edge cache/selector behavior; override only backend-specific path via env/loader.

        cfg, out_dir, row = ns['_run_one_scenario_from_overrides'](
            user_overrides,
            scenario_label=str(payload.get('scenario_label', backend)),
            selector_json=str(selector_json),
        )
        row = dict(row)
        row['backend_label'] = backend
        row['phase2_root'] = str(phase2_root)
        row['selector_json'] = str(selector_json)
        row['worker_wall_s'] = float(time.perf_counter() - t0)

        print(json.dumps({
            'ok': True,
            'backend': backend,
            'row': row,
            'cfg': {
                'run_id': str(cfg.get('run_id')),
                'runs_root': str(cfg.get('runs_root')),
                'backend': cfg.get('backend'),
            },
            'out_dir': str(out_dir),
        }, default=str))
        return 0
    except Exception as e:
        print(json.dumps({
            'ok': False,
            'backend': backend,
            'error': repr(e),
            'traceback': traceback.format_exc(),
        }, default=str))
        return 1


def _write_outputs(out_dir: Path, payload: dict, neuron_res: dict | None, arbor_res: dict | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'compare_payload.json').write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')

    rows = []
    for res in (neuron_res, arbor_res):
        if not res:
            continue
        if res.get('ok') and isinstance(res.get('row'), dict):
            rows.append(dict(res['row']))
        else:
            rows.append({
                'backend_label': res.get('backend'),
                'error': res.get('error'),
                'returncode': res.get('returncode'),
            })
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / 'backend_runs.csv', index=False)

    # Save raw worker responses for debugging.
    with (out_dir / 'worker_results.json').open('w', encoding='utf-8') as f:
        json.dump({'neuron': neuron_res, 'arbor': arbor_res}, f, indent=2, default=str)

    if not (neuron_res and arbor_res and neuron_res.get('ok') and arbor_res.get('ok')):
        return

    neuron_row = dict(neuron_res['row'])
    arbor_row = dict(arbor_res['row'])
    neuron_out = Path(neuron_res['out_dir']).expanduser().resolve()
    arbor_out = Path(arbor_res['out_dir']).expanduser().resolve()

    timing = pd.DataFrame([{
        'neuron_runtime_s': pd.to_numeric(pd.Series([neuron_row.get('runtime_s')]), errors='coerce').iloc[0],
        'arbor_runtime_s': pd.to_numeric(pd.Series([arbor_row.get('runtime_s')]), errors='coerce').iloc[0],
        'neuron_worker_wall_s': pd.to_numeric(pd.Series([neuron_row.get('worker_wall_s')]), errors='coerce').iloc[0],
        'arbor_worker_wall_s': pd.to_numeric(pd.Series([arbor_row.get('worker_wall_s')]), errors='coerce').iloc[0],
    }])
    if pd.notna(timing.loc[0, 'neuron_runtime_s']) and float(timing.loc[0, 'neuron_runtime_s']) > 0:
        if pd.notna(timing.loc[0, 'arbor_runtime_s']) and float(timing.loc[0, 'arbor_runtime_s']) > 0:
            timing.loc[0, 'neuron_runtime_div_arbor_runtime'] = float(timing.loc[0, 'neuron_runtime_s']) / float(timing.loc[0, 'arbor_runtime_s'])
            timing.loc[0, 'arbor_runtime_div_neuron_runtime'] = float(timing.loc[0, 'arbor_runtime_s']) / float(timing.loc[0, 'neuron_runtime_s'])
        else:
            timing.loc[0, 'neuron_runtime_div_arbor_runtime'] = np.nan
            timing.loc[0, 'arbor_runtime_div_neuron_runtime'] = np.nan
    timing.to_csv(out_dir / 'timing_comparison.csv', index=False)

    row_diff = _numeric_row_diff(neuron_row, arbor_row)
    if not row_diff.empty:
        row_diff.to_csv(out_dir / 'run_metric_deltas.csv', index=False)

    spike_summary, spike_detail = compare_spikes(neuron_out, arbor_out, tol_ms=float(payload.get('spike_match_tol_ms', 0.25)))
    spike_summary.to_csv(out_dir / 'spike_comparison_summary.csv', index=False)
    spike_detail.to_csv(out_dir / 'spike_comparison_per_neuron.csv', index=False)

    rec_summary, rec_detail = compare_records(neuron_out, arbor_out)
    rec_summary.to_csv(out_dir / 'trace_comparison_summary.csv', index=False)
    rec_detail.to_csv(out_dir / 'trace_comparison_per_signal.csv', index=False)

    summary_json = {
        'neuron_out_dir': str(neuron_out),
        'arbor_out_dir': str(arbor_out),
        'timing': (timing.iloc[0].to_dict() if not timing.empty else {}),
        'spike_summary': (spike_summary.iloc[0].to_dict() if not spike_summary.empty else {}),
        'trace_summary': (rec_summary.iloc[0].to_dict() if not rec_summary.empty else {}),
    }
    (out_dir / 'comparison_summary.json').write_text(json.dumps(summary_json, indent=2, default=str), encoding='utf-8')


def main() -> int:
    ap = argparse.ArgumentParser(description='Run glia_simulation scenario against NEURON and Arbor Phase 2 backends and compare timing/results.')
    ap.add_argument('--worker-payload-json', help=argparse.SUPPRESS)

    ap.add_argument('--glia-notebook', default=str(DEFAULT_GLIA_NOTEBOOK))
    ap.add_argument('--phase2-neuron-root', default=str(DEFAULT_PHASE2_NEURON_ROOT))
    ap.add_argument('--phase2-arbor-root', default=str(DEFAULT_PHASE2_ARBOR_ROOT))
    ap.add_argument('--neuron-worker-python', default=os.environ.get('DIGIFLY_NEURON_WORKER_PYTHON'))
    ap.add_argument('--arbor-worker-python', default=os.environ.get('DIGIFLY_ARBOR_WORKER_PYTHON'))
    ap.add_argument('--output-root', default=str(DEFAULT_OUTPUT_ROOT))
    ap.add_argument('--selector-json', default=None, help='Explicit selector JSON. If omitted, inferred from glia_simulation notebook namespace.')
    ap.add_argument('--scenario-selector-key', default='single', help='Key passed to _selector_json_for_scenario(...) when inferring selector JSON (default: single).')
    ap.add_argument('--scenario-label', default='single_compare', help='Scenario label passed to _run_one_scenario_from_overrides (suffixes backend names).')
    ap.add_argument('--glia-state', choices=['inherit', 'on', 'off'], default='inherit')
    ap.add_argument('--force-override-enabled', choices=['inherit', 'true', 'false'], default='inherit')
    ap.add_argument('--ko-mm', type=float, default=None, help='Override GLIA_FORCE_KO_mM for both backends.')
    ap.add_argument('--parallel-launch', action='store_true', help='Launch NEURON and Arbor workers concurrently (faster, but less fair for runtime comparisons).')
    ap.add_argument('--arbor-force-native', choices=['inherit', 'true', 'false'], default='inherit')
    ap.add_argument('--arbor-native-strict', choices=['inherit', 'true', 'false'], default='inherit')
    ap.add_argument('--arbor-rect-gap-policy', choices=['inherit', 'symmetric_ohmic', 'ignore', 'fallback'], default='inherit')
    ap.add_argument('--arbor-gap-exclude-nids', default='inherit', help="Comma-separated neuron IDs with native gap incompatibilities; rectifying->ohmic contacts for these IDs are demoted to a surrogate-ohmic correction by default.")
    ap.add_argument('--skip-neuron', action='store_true')
    ap.add_argument('--skip-arbor', action='store_true')
    ap.add_argument('--spike-match-tol-ms', type=float, default=0.25)
    ap.add_argument('--dry-run', action='store_true', help='Load notebook helpers and print resolved paths/config only; do not run simulations.')

    args = ap.parse_args()

    if args.worker_payload_json:
        payload = json.loads(args.worker_payload_json)
        return _worker_main(payload)

    glia_notebook = Path(args.glia_notebook).expanduser().resolve()
    if not glia_notebook.exists():
        raise FileNotFoundError(f'glia notebook not found: {glia_notebook}')

    output_root = Path(args.output_root).expanduser().resolve()
    run_group = f'glia_neuron_vs_arbor_{_ts_utc()}'
    out_dir = output_root / run_group
    out_dir.mkdir(parents=True, exist_ok=True)

    force_override_enabled = None
    if args.force_override_enabled != 'inherit':
        force_override_enabled = (args.force_override_enabled == 'true')
    arbor_force_native = None
    if args.arbor_force_native != 'inherit':
        arbor_force_native = (args.arbor_force_native == 'true')
    arbor_native_strict = None
    if args.arbor_native_strict != 'inherit':
        arbor_native_strict = (args.arbor_native_strict == 'true')
    arbor_rect_gap_policy = None if args.arbor_rect_gap_policy == 'inherit' else str(args.arbor_rect_gap_policy)
    arbor_gap_exclude_nids = None if str(args.arbor_gap_exclude_nids).strip().lower() == 'inherit' else str(args.arbor_gap_exclude_nids).strip()

    common_payload = {
        'glia_notebook': str(glia_notebook),
        'selector_json': args.selector_json,
        'scenario_selector_key': str(args.scenario_selector_key),
        'scenario_label': str(args.scenario_label),
        'glia_state': str(args.glia_state),
        'force_override_enabled': force_override_enabled,
        'ko_mM': args.ko_mm,
        'spike_match_tol_ms': float(args.spike_match_tol_ms),
        'arbor_force_native': arbor_force_native,
        'arbor_native_strict': arbor_native_strict,
        'arbor_rect_gap_policy': arbor_rect_gap_policy,
        'arbor_gap_exclude_nids': arbor_gap_exclude_nids,
    }

    if args.dry_run:
        # Validate both namespaces/selector resolution without running simulations.
        report = {}
        for backend, root in [('neuron', args.phase2_neuron_root), ('arbor', args.phase2_arbor_root)]:
            if (backend == 'neuron' and args.skip_neuron) or (backend == 'arbor' and args.skip_arbor):
                continue
            ns = load_glia_namespace(glia_notebook, backend=backend, phase2_root=root)
            selector_json = _resolve_selector_json(ns, args.selector_json, scenario=args.scenario_selector_key)
            report[backend] = {
                'phase2_root': str(Path(root).expanduser().resolve()),
                'run_id_base': str(ns.get('RUN_ID')),
                'selector_json': str(selector_json),
                'glia_loss_enabled_default': bool(ns.get('GLIA_LOSS_ENABLED', False)),
                'run_state_tag_default': str(ns.get('RUN_STATE_TAG', '')),
            }
        (out_dir / 'dry_run_report.json').write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
        print(json.dumps({'ok': True, 'dry_run': True, 'out_dir': str(out_dir), 'report': report}, indent=2, default=str))
        return 0

    workers: list[tuple[str, dict]] = []
    if not args.skip_neuron:
        workers.append(('neuron', {
            **common_payload,
            'backend': 'neuron',
            'worker_python': (str(args.neuron_worker_python) if args.neuron_worker_python else None),
            'phase2_root': str(Path(args.phase2_neuron_root).expanduser().resolve()),
            'run_output_root': str((out_dir / 'runs_neuron').resolve()),
            'run_id': f'{run_group}_neuron',
        }))
    if not args.skip_arbor:
        workers.append(('arbor', {
            **common_payload,
            'backend': 'arbor',
            'worker_python': (str(args.arbor_worker_python) if args.arbor_worker_python else None),
            'phase2_root': str(Path(args.phase2_arbor_root).expanduser().resolve()),
            'run_output_root': str((out_dir / 'runs_arbor').resolve()),
            'run_id': f'{run_group}_arbor',
        }))
    if not workers:
        raise ValueError('Both backends are skipped. Remove --skip-neuron and/or --skip-arbor.')

    results: dict[str, dict] = {}

    t_master = time.perf_counter()
    if args.parallel_launch and len(workers) > 1:
        procs = []
        for backend, payload in workers:
            worker_python = str(payload.get('worker_python') or sys.executable)
            cmd = [worker_python, str(Path(__file__).resolve()), '--worker-payload-json', json.dumps(payload)]
            procs.append((backend, payload, subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)))
        for backend, _payload, proc in procs:
            stdout, stderr = proc.communicate()
            try:
                res = json.loads((stdout or '').strip().splitlines()[-1]) if (stdout or '').strip() else {}
            except Exception:
                res = {'ok': False, 'backend': backend, 'error': 'worker_output_parse_failed', 'stdout': stdout, 'stderr': stderr}
            res.setdefault('ok', False)
            res['returncode'] = int(proc.returncode)
            if stderr.strip():
                res['stderr_tail'] = stderr[-8000:]
            results[backend] = res
    else:
        for backend, payload in workers:
            results[backend] = _run_subprocess(payload)
    master_wall_s = float(time.perf_counter() - t_master)

    compare_payload = {
        'run_group': run_group,
        'master_wall_s': master_wall_s,
        'parallel_launch': bool(args.parallel_launch),
        'spike_match_tol_ms': float(args.spike_match_tol_ms),
        'args': vars(args),
    }
    _write_outputs(out_dir, compare_payload, results.get('neuron'), results.get('arbor'))

    ok_neuron = bool(results.get('neuron', {}).get('ok')) if 'neuron' in results else None
    ok_arbor = bool(results.get('arbor', {}).get('ok')) if 'arbor' in results else None
    print(f'[compare] output_dir: {out_dir}')
    print(f'[compare] master_wall_s: {master_wall_s:.3f}')
    print(f'[compare] neuron_ok={ok_neuron} arbor_ok={ok_arbor}')
    if (out_dir / 'comparison_summary.json').exists():
        print(f'[compare] summary: {out_dir / "comparison_summary.json"}')
        print(f'[compare] timing : {out_dir / "timing_comparison.csv"}')
        print(f'[compare] traces : {out_dir / "trace_comparison_summary.csv"}')
        print(f'[compare] spikes : {out_dir / "spike_comparison_summary.csv"}')

    # Non-zero if any requested backend failed.
    failed = [b for b in ('neuron', 'arbor') if (b in results and not bool(results[b].get('ok')))]
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
