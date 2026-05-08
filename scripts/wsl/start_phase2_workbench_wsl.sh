#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_DIR="${DIGIFLY_WSL_VENV:-${REPO_ROOT}/.venv-wsl}"
PHASE2_ROOT="${REPO_ROOT}/Phase 2"
SWC_DIR_DEFAULT="${REPO_ROOT}/Phase 1/manc_v1.2.1/export_swc"
PORT="${DIGIFLY_JUPYTER_PORT:-8888}"
WORKBENCH_PATH="/lab/tree/Phase%202/notebooks/Digifly_Phase2_Workbench.ipynb"
WORKBENCH_URL="http://localhost:${PORT}${WORKBENCH_PATH}"

open_browser() {
  local url="$1"
  if command -v wslview >/dev/null 2>&1; then
    wslview "${url}" >/dev/null 2>&1 || true
  elif command -v explorer.exe >/dev/null 2>&1; then
    explorer.exe "${url}" >/dev/null 2>&1 || true
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "${url}" >/dev/null 2>&1 || true
  fi
}

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "[digifly-wsl] First run detected. Installing the WSL runtime..."
  "${SCRIPT_DIR}/setup_phase2_wsl.sh"
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

export PYTHONPATH="${PHASE2_ROOT}:${REPO_ROOT}/Phase 1:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export DIGIFLY_WORKSPACE="${REPO_ROOT}"
export DIGIFLY_PHASE2_ROOT="${PHASE2_ROOT}"
export DIGIFLY_SWC_DIR="${DIGIFLY_SWC_DIR:-${SWC_DIR_DEFAULT}}"
export DIGIFLY_GAP_MECH_DIR="${DIGIFLY_GAP_MECH_DIR:-${PHASE2_ROOT}/data}"
export NEURON_MODULE_OPTIONS="${NEURON_MODULE_OPTIONS:--nogui}"

if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  cat <<'MSG'
[digifly-wsl] DISPLAY/WAYLAND_DISPLAY is not set.
[digifly-wsl] NEURON simulations can still run, but the PyVista mutation app needs WSLg or an X server.
MSG
else
  echo "[digifly-wsl] Desktop display detected; the workbench can launch the PyVista mutation app."
fi

echo "[digifly-wsl] Repo: ${REPO_ROOT}"
echo "[digifly-wsl] Phase 2: ${DIGIFLY_PHASE2_ROOT}"
echo "[digifly-wsl] SWC root: ${DIGIFLY_SWC_DIR}"
echo "[digifly-wsl] Opening: ${WORKBENCH_URL}"

if [[ "${DIGIFLY_OPEN_BROWSER:-1}" != "0" ]]; then
  (sleep 3; open_browser "${WORKBENCH_URL}") &
fi

exec python -m jupyter lab \
  --ip=127.0.0.1 \
  --port="${PORT}" \
  --no-browser \
  --ServerApp.token= \
  --ServerApp.password= \
  --ServerApp.root_dir="${REPO_ROOT}" \
  --ServerApp.default_url="${WORKBENCH_PATH}"
