"""neuPrint token helpers for Phase 1.

The project prefers the `NEUPRINT_TOKEN` environment variable when present.
For notebook users, it also supports a local gitignored file:
`Phase 1/Neuprint Token.txt`.
"""

from __future__ import annotations

import getpass
import os
from pathlib import Path


TOKEN_FILE_NAME = "Neuprint Token.txt"
PLACEHOLDER_TOKENS = {
    "",
    "your-token-here",
    "paste-your-token-here",
    "paste token here",
    "paste your neuprint token here",
}


def phase1_root() -> Path:
    cwd = Path.cwd().resolve()
    candidates = [
        cwd,
        cwd / "Phase 1",
        Path(__file__).resolve().parents[1],
    ]
    for candidate in candidates:
        if (candidate / "digifly_phase1").exists() or (candidate / "Phase 1.ipynb").exists():
            return candidate
    return Path(__file__).resolve().parents[1]


def token_file_path() -> Path:
    return phase1_root() / TOKEN_FILE_NAME


def token_looks_plausible(token: str | None) -> bool:
    """Return True for values that look like real neuPrint auth tokens."""
    if token is None:
        return False

    text = str(token).strip()
    if text.lower() in PLACEHOLDER_TOKENS:
        return False

    # neuPrint tokens are normally JWTs. Keep a length fallback so this helper
    # does not become brittle if token formatting changes later.
    if text.startswith("eyJ") and text.count(".") == 2:
        return True
    return len(text) >= 80


def get_neuprint_token(*, required: bool = True) -> str | None:
    token = os.environ.get("NEUPRINT_TOKEN", "").strip()
    if token and token_looks_plausible(token):
        return token

    path = token_file_path()
    if path.exists():
        token = path.read_text(encoding="utf-8").strip()
        if token and token_looks_plausible(token):
            return token

    if required:
        if token:
            raise RuntimeError(
                "A neuPrint token was found, but it does not look valid. "
                f"Replace the contents of {path} with your full neuPrint token, "
                "or set NEUPRINT_TOKEN."
            )
        raise RuntimeError(
            "No neuPrint token found. Set NEUPRINT_TOKEN or save it to "
            f"{path}."
        )
    return None


def save_neuprint_token(token: str | None = None, *, overwrite: bool = True) -> Path:
    path = token_file_path()
    if token is None:
        token = getpass.getpass("neuPrint token: ").strip()
    else:
        token = str(token).strip()

    if not token:
        raise ValueError("Token was empty; nothing was saved.")
    if not token_looks_plausible(token):
        raise ValueError(
            "That does not look like a full neuPrint token. Copy the complete "
            "token from neuPrint and try again."
        )
    if path.exists() and path.read_text(encoding="utf-8").strip() and not overwrite:
        raise FileExistsError(f"Token file already exists: {path}")

    path.write_text(token + "\n", encoding="utf-8")
    return path


def ensure_neuprint_token(*, prompt_if_missing: bool = True) -> str:
    token = get_neuprint_token(required=False)
    if token:
        print(f"neuPrint token found via NEUPRINT_TOKEN or {token_file_path().name}.")
        return token

    if not prompt_if_missing:
        raise RuntimeError(
            "No neuPrint token found. Set NEUPRINT_TOKEN or save it with "
            "save_neuprint_token()."
        )

    save_neuprint_token()
    print(f"Saved neuPrint token to {token_file_path()}.")
    return get_neuprint_token(required=True)
