#!/usr/bin/env bash

is_wsl() {
  grep -qi microsoft /proc/version 2>/dev/null || grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null
}

is_windows_bash() {
  case "$(uname -s 2>/dev/null || true)" in
    MINGW*|MSYS*|CYGWIN*) return 0 ;;
    *) return 1 ;;
  esac
}

find_wsl_exe() {
  if [[ -n "${DIGIFLY_WSL_EXE:-}" ]]; then
    printf '%s\n' "${DIGIFLY_WSL_EXE}"
    return 0
  fi

  command -v wsl.exe 2>/dev/null || command -v wsl 2>/dev/null
}

to_windows_path() {
  local path="$1"

  if command -v cygpath >/dev/null 2>&1; then
    cygpath -am "${path}"
    return
  fi

  case "${path}" in
    /[a-zA-Z]/*)
      local drive="${path:1:1}"
      local rest="${path:2}"
      drive="$(printf '%s' "${drive}" | tr '[:lower:]' '[:upper:]')"
      printf '%s:%s\n' "${drive}" "${rest}"
      ;;
    *)
      return 1
      ;;
  esac
}

run_wsl() {
  local wsl_exe="$1"
  shift

  if [[ -n "${DIGIFLY_WSL_DISTRO:-}" ]]; then
    "${wsl_exe}" -d "${DIGIFLY_WSL_DISTRO}" "$@"
  else
    "${wsl_exe}" "$@"
  fi
}

relaunch_in_wsl() {
  local repo_root="$1"
  local script_from_repo="$2"
  shift 2

  local wsl_exe
  wsl_exe="$(find_wsl_exe || true)"
  if [[ -z "${wsl_exe}" ]]; then
    cat >&2 <<'MSG'
[digifly-wsl] This launcher is running from a Windows Bash shell, but wsl.exe was not found.
[digifly-wsl] Install WSL with Ubuntu, then run this script again.
MSG
    exit 1
  fi

  local repo_root_win
  if ! repo_root_win="$(to_windows_path "${repo_root}")"; then
    cat >&2 <<MSG
[digifly-wsl] Could not convert this path for WSL:
[digifly-wsl]   ${repo_root}
[digifly-wsl] Move the repo under a normal Windows drive path, then run the launcher again.
MSG
    exit 1
  fi

  local repo_root_wsl
  if ! repo_root_wsl="$(run_wsl "${wsl_exe}" wslpath -a -u "${repo_root_win}" | tr -d '\015')"; then
    cat >&2 <<MSG
[digifly-wsl] WSL could not access this Windows path:
[digifly-wsl]   ${repo_root_win}
[digifly-wsl] Check that WSL is installed and the repo is on a mounted Windows drive.
MSG
    exit 1
  fi

  if [[ -z "${repo_root_wsl}" ]]; then
    echo "[digifly-wsl] WSL returned an empty repo path." >&2
    exit 1
  fi

  local cmd
  printf -v cmd 'cd %q && exec bash %q' "${repo_root_wsl}" "${script_from_repo}"
  local arg
  for arg in "$@"; do
    printf -v cmd '%s %q' "${cmd}" "${arg}"
  done

  if [[ -n "${DIGIFLY_WSL_DISTRO:-}" ]]; then
    echo "[digifly-wsl] Relaunching inside WSL distro ${DIGIFLY_WSL_DISTRO}: ${repo_root_wsl}"
    "${wsl_exe}" -d "${DIGIFLY_WSL_DISTRO}" bash -lc "${cmd}"
  else
    echo "[digifly-wsl] Relaunching inside WSL default distro: ${repo_root_wsl}"
    "${wsl_exe}" bash -lc "${cmd}"
  fi
  local status=$?

  if [[ ${status} -ne 0 && -t 0 && -z "${CI:-}" && "${DIGIFLY_NO_PAUSE:-0}" != "1" ]]; then
    echo
    read -r -p "[digifly-wsl] Startup failed. Press Enter to close this window..."
  fi

  exit "${status}"
}

ensure_wsl_when_started_from_windows_bash() {
  local repo_root="$1"
  local script_from_repo="$2"
  shift 2

  if ! is_wsl && is_windows_bash; then
    relaunch_in_wsl "${repo_root}" "${script_from_repo}" "$@"
  fi
}

maybe_print_launch_check() {
  local repo_root="$1"
  shift

  if [[ "${1:-}" == "--check-launch" ]]; then
    echo "[digifly-wsl] Launch check OK."
    echo "[digifly-wsl] Repo: ${repo_root}"
    echo "[digifly-wsl] System: $(uname -srm)"
    if command -v python3 >/dev/null 2>&1; then
      echo "[digifly-wsl] python3: $(command -v python3)"
      python3 --version
    else
      echo "[digifly-wsl] python3: not found"
    fi
    exit 0
  fi
}
