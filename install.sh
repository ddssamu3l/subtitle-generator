#!/usr/bin/env bash
#
# One-command setup for macOS and Linux.
#
# The awkward part of installing a Python GUI tool is that many Python builds
# ship without tkinter — Homebrew's Python and several distro packages both do
# this — and the failure only shows up at launch, as an opaque ImportError. We
# sidestep it by using a uv-managed Python, which always bundles tkinter.

set -euo pipefail

BOLD=$(tput bold 2>/dev/null || printf '')
DIM=$(tput dim 2>/dev/null || printf '')
RED=$(tput setaf 1 2>/dev/null || printf '')
GREEN=$(tput setaf 2 2>/dev/null || printf '')
YELLOW=$(tput setaf 3 2>/dev/null || printf '')
RESET=$(tput sgr0 2>/dev/null || printf '')

step()  { printf "\n%s==>%s %s%s\n" "$BOLD" "$RESET" "$1" "$RESET"; }
ok()    { printf "  %s✓%s %s\n" "$GREEN" "$RESET" "$1"; }
warn()  { printf "  %s!%s %s\n" "$YELLOW" "$RESET" "$1"; }
fail()  { printf "  %s✗%s %s\n" "$RED" "$RESET" "$1" >&2; }

cd "$(dirname "$0")"

PYTHON_VERSION="3.12"

# --- 1. uv ------------------------------------------------------------------
step "Checking for uv"
if command -v uv >/dev/null 2>&1; then
  ok "uv is already installed"
else
  warn "uv not found — installing it (this only touches ~/.local/bin)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # The installer adds it to PATH for new shells; make it usable right now.
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  if command -v uv >/dev/null 2>&1; then
    ok "uv installed"
  else
    fail "uv installed but is not on PATH. Open a new terminal and re-run this script."
    exit 1
  fi
fi

# --- 2. environment ---------------------------------------------------------
step "Creating an isolated environment (Python $PYTHON_VERSION)"
uv venv --python "$PYTHON_VERSION" .venv
ok "Environment ready at .venv"

step "Installing the package"
uv pip install --python .venv -e . --quiet
ok "Installed"

# Confirm the GUI toolkit is genuinely present rather than assuming it.
if .venv/bin/python -c "import tkinter" 2>/dev/null; then
  ok "GUI support confirmed"
else
  warn "This Python has no tkinter, so the graphical picker will not open."
  warn "The command line still works: subgen video.mp4 --lang en"
fi

# --- 3. ffmpeg --------------------------------------------------------------
step "Checking for ffmpeg"
if command -v ffmpeg >/dev/null 2>&1; then
  ok "ffmpeg found at $(command -v ffmpeg)"
else
  fail "ffmpeg is required and was not found."
  if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "     Install it with:  brew install ffmpeg"
  elif command -v apt >/dev/null 2>&1; then
    echo "     Install it with:  sudo apt install ffmpeg"
  elif command -v dnf >/dev/null 2>&1; then
    echo "     Install it with:  sudo dnf install ffmpeg"
  elif command -v pacman >/dev/null 2>&1; then
    echo "     Install it with:  sudo pacman -S ffmpeg"
  else
    echo "     See https://ffmpeg.org/download.html"
  fi
  echo "     Then re-run this script."
  exit 1
fi

# --- 4. models --------------------------------------------------------------
step "Downloading models for offline use"
echo "  ${DIM}This needs internet once. Afterwards the tool runs with no network.${RESET}"
.venv/bin/subgen setup

# --- 5. optional extras -----------------------------------------------------
step "Optional extras"

if [[ "$OSTYPE" == "darwin"* && "$(uname -m)" == "arm64" ]]; then
  echo "  You're on Apple Silicon. GPU transcription is much faster:"
  echo "      ${BOLD}uv pip install --python .venv -e '.[mlx]'${RESET}"
fi

if command -v ollama >/dev/null 2>&1; then
  if .venv/bin/subgen --list-models >/dev/null 2>&1; then
    ok "Ollama is ready for translation"
  else
    warn "Ollama is installed but has no chat models. Pull one to enable translation:"
    echo "      ${BOLD}ollama pull qwen3:8b${RESET}   ${DIM}(any chat model works)${RESET}"
  fi
else
  warn "Ollama is not installed — subtitles will stay in the spoken language."
  echo "     Translation is optional. To enable it, install Ollama yourself:"
  echo "      ${BOLD}https://ollama.com/download${RESET}"
  echo "     ${DIM}We don't install it for you: it registers a background service.${RESET}"
fi

# --- done -------------------------------------------------------------------
printf "\n%sDone.%s Start it with:\n\n" "$BOLD$GREEN" "$RESET"
printf "    %ssource .venv/bin/activate%s\n" "$BOLD" "$RESET"
printf "    %ssubgen%s\n\n" "$BOLD" "$RESET"
printf "Or without activating:  %s./.venv/bin/subgen%s\n\n" "$BOLD" "$RESET"
