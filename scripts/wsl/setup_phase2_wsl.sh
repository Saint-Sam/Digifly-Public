#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Allow users to run setup directly from Windows Git Bash too.
# shellcheck source=scripts/wsl/launch_helpers.sh
source "${SCRIPT_DIR}/launch_helpers.sh"
ensure_wsl_when_started_from_windows_bash "${REPO_ROOT}" "scripts/wsl/setup_phase2_wsl.sh" "$@"
maybe_print_launch_check "${REPO_ROOT}" "$@"

VENV_DIR="${DIGIFLY_WSL_VENV:-${REPO_ROOT}/.venv-wsl}"
REQ_FILE="${SCRIPT_DIR}/requirements-phase2-wsl.txt"
PHASE2_ROOT="${REPO_ROOT}/Phase 2"
MECH_DIR="${PHASE2_ROOT}/data"
PYTHON_ONLY=0

for arg in "$@"; do
  case "${arg}" in
    --python-only)
      PYTHON_ONLY=1
      ;;
    --check-launch)
      ;;
    *)
      echo "[digifly-wsl] Unknown setup argument: ${arg}" >&2
      exit 2
      ;;
  esac
done

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

venv_is_usable() {
  [[ -x "${VENV_DIR}/bin/python" && -f "${VENV_DIR}/bin/activate" ]]
}

move_broken_venv_aside() {
  if [[ ! -e "${VENV_DIR}" || -L "${VENV_DIR}" || venv_is_usable ]]; then
    return
  fi

  local backup_dir
  backup_dir="${VENV_DIR}.broken.$(date +%Y%m%d-%H%M%S)"
  echo "[digifly-wsl] Found an incomplete Python environment: ${VENV_DIR}"
  echo "[digifly-wsl] Moving it aside so setup can rebuild cleanly: ${backup_dir}"
  mv "${VENV_DIR}" "${backup_dir}"
}

install_apt_packages() {
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "[digifly-wsl] apt-get not found; skipping Ubuntu package install."
    return
  fi

  echo "[digifly-wsl] Installing Ubuntu packages needed by NEURON, MPI, and PyVista..."
  if ! sudo -n true >/dev/null 2>&1; then
    local sudo_user distro_name
    sudo_user="$(whoami)"
    distro_name="${WSL_DISTRO_NAME:-<your-distro-name>}"
    cat <<MSG
[digifly-wsl] sudo may ask for the Linux password for '${sudo_user}'.
[digifly-wsl] This is the password created when this WSL distro was first set up, not your Windows password.
[digifly-wsl] If you do not know it, open PowerShell and reset it with:
[digifly-wsl]   wsl -d ${distro_name} -u root passwd ${sudo_user}
MSG
  fi
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends "${APT_PACKAGES[@]}"
}

create_venv() {
  if ! command -v python3 >/dev/null 2>&1; then
    cat >&2 <<'MSG'
[digifly-wsl] python3 is not installed in this Linux environment.
[digifly-wsl] On Ubuntu/WSL, install it with:
[digifly-wsl]   sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-dev
MSG
    exit 1
  fi

  move_broken_venv_aside

  if ! venv_is_usable; then
    echo "[digifly-wsl] Creating Python virtual environment: ${VENV_DIR}"
    python3 -m venv "${VENV_DIR}"
  fi

  if ! venv_is_usable; then
    cat >&2 <<MSG
[digifly-wsl] Python virtual environment was not created correctly:
[digifly-wsl]   ${VENV_DIR}
[digifly-wsl] Make sure python3-venv installed successfully, then run the launcher again.
MSG
    exit 1
  fi

  # shellcheck source=/dev/null
  source "${VENV_DIR}/bin/activate"
  python -m pip install --upgrade pip wheel "setuptools<81"
  python -m pip install --upgrade -r "${REQ_FILE}"
}

verify_jupyter_stack() {
  if ! venv_is_usable; then
    echo "[digifly-wsl] Python virtual environment is missing or incomplete: ${VENV_DIR}" >&2
    return 1
  fi

  # shellcheck source=/dev/null
  source "${VENV_DIR}/bin/activate"

  python - <<'PY'
from importlib.metadata import PackageNotFoundError, version

required = {
    "ipywidgets": "8.",
    "jupyterlab_widgets": "3.",
    "jupyterlab": "4.",
    "notebook": "7.",
    "ipykernel": "6.",
}

for package, prefix in required.items():
    try:
        installed = version(package)
    except PackageNotFoundError as exc:
        raise SystemExit(f"{package} is not installed") from exc
    if not installed.startswith(prefix):
        raise SystemExit(f"{package} {installed} is installed, expected {prefix}x")

print("[digifly-wsl] JupyterLab, kernel, and widget stack are compatible.")
PY
}

compile_mechanisms() {
  if ! venv_is_usable; then
    echo "[digifly-wsl] Python virtual environment is missing or incomplete: ${VENV_DIR}" >&2
    return 1
  fi

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

if [[ "${PYTHON_ONLY}" != "1" ]]; then
  install_apt_packages
fi
create_venv
verify_jupyter_stack

if [[ "${PYTHON_ONLY}" != "1" ]]; then
  compile_mechanisms
else
  echo "[digifly-wsl] Python dependencies updated; skipping apt packages and mechanism compilation."
fi

echo "[digifly-wsl] Setup complete."
