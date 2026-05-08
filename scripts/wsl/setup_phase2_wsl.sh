#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_DIR="${DIGIFLY_WSL_VENV:-${REPO_ROOT}/.venv-wsl}"
REQ_FILE="${SCRIPT_DIR}/requirements-phase2-wsl.txt"
PHASE2_ROOT="${REPO_ROOT}/Phase 2"
MECH_DIR="${PHASE2_ROOT}/data"

APT_PACKAGES=(
  build-essential
  ca-certificates
  cmake
  curl
  git
  libdbus-1-3
  libegl1
  libfontconfig1
  libgl1
  libglib2.0-0
  libglx-mesa0
  libgomp1
  libopenmpi-dev
  libsm6
  libx11-6
  libxcursor1
  libxext6
  libxi6
  libxinerama1
  libxrandr2
  libxrender1
  make
  openmpi-bin
  pkg-config
  python3
  python3-dev
  python3-venv
)

is_wsl() {
  grep -qi microsoft /proc/version 2>/dev/null || grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null
}

install_apt_packages() {
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "[digifly-wsl] apt-get not found; skipping Ubuntu package install."
    return
  fi

  echo "[digifly-wsl] Installing Ubuntu packages needed by NEURON, MPI, and PyVista..."
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends "${APT_PACKAGES[@]}"
}

create_venv() {
  if [[ ! -d "${VENV_DIR}" ]]; then
    echo "[digifly-wsl] Creating Python virtual environment: ${VENV_DIR}"
    python3 -m venv "${VENV_DIR}"
  fi

  # shellcheck source=/dev/null
  source "${VENV_DIR}/bin/activate"
  python -m pip install --upgrade pip wheel "setuptools<81"
  python -m pip install -r "${REQ_FILE}"
}

compile_mechanisms() {
  # shellcheck source=/dev/null
  source "${VENV_DIR}/bin/activate"

  echo "[digifly-wsl] Compiling Phase 2 NEURON mechanisms..."
  (
    cd "${MECH_DIR}"
    nrnivmodl Gap.mod RectGap.mod HeteroRectGap.mod
  )

  DIGIFLY_GAP_MECH_DIR="${MECH_DIR}" python - <<'PY'
import os
from pathlib import Path
from neuron import h, load_mechanisms

mech_dir = Path(os.environ["DIGIFLY_GAP_MECH_DIR"]).resolve()
load_mechanisms(str(mech_dir))
missing = [name for name in ("Gap", "RectGap", "HeteroRectGap") if not hasattr(h, name)]
if missing:
    raise SystemExit(f"Missing compiled NEURON mechanisms: {missing}")
print("[digifly-wsl] NEURON mechanisms are compiled and loadable.")
PY
}

if ! is_wsl; then
  echo "[digifly-wsl] This does not look like WSL. Continuing anyway."
fi

install_apt_packages
create_venv
compile_mechanisms

echo "[digifly-wsl] Setup complete."
