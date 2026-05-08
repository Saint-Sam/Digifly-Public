#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# If this .sh is started from Git Bash/MSYS/Cygwin on Windows, hand off to the
# user's default WSL distro before doing Linux setup work.
# shellcheck source=scripts/wsl/launch_helpers.sh
source "${SCRIPT_DIR}/scripts/wsl/launch_helpers.sh"
ensure_wsl_when_started_from_windows_bash "${SCRIPT_DIR}" "./Start_Digifly_Phase2_WSL.sh" "$@"
maybe_print_launch_check "${SCRIPT_DIR}" "$@"

exec "${SCRIPT_DIR}/scripts/wsl/start_phase2_workbench_wsl.sh" "$@"
