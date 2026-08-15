# One-command setup for Windows.
#
# See install.sh for the reasoning: we use a uv-managed Python because many
# Python builds ship without tkinter, and that failure only surfaces at launch
# as an opaque ImportError.

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$PythonVersion = "3.12"

function Step($message) { Write-Host "`n==> $message" -ForegroundColor White }
function Ok($message)   { Write-Host "  [ok] $message" -ForegroundColor Green }
function Warn($message) { Write-Host "  [!] $message" -ForegroundColor Yellow }
function Fail($message) { Write-Host "  [x] $message" -ForegroundColor Red }

# --- 1. uv ------------------------------------------------------------------
Step "Checking for uv"
if (Get-Command uv -ErrorAction SilentlyContinue) {
    Ok "uv is already installed"
} else {
    Warn "uv not found - installing it"
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        Ok "uv installed"
    } else {
        Fail "uv installed but is not on PATH. Open a new terminal and re-run this script."
        exit 1
    }
}

# --- 2. environment ---------------------------------------------------------
Step "Creating an isolated environment (Python $PythonVersion)"
uv venv --python $PythonVersion .venv
Ok "Environment ready at .venv"

Step "Installing the package"
uv pip install --python .venv -e . --quiet
Ok "Installed"

if (& .\.venv\Scripts\python.exe -c "import tkinter" 2>$null) {
    Ok "GUI support confirmed"
} else {
    Warn "This Python has no tkinter, so the graphical picker will not open."
    Warn "The command line still works: subgen video.mp4 --lang en"
}

# --- 3. ffmpeg --------------------------------------------------------------
Step "Checking for ffmpeg"
if (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
    Ok "ffmpeg found"
} else {
    Fail "ffmpeg is required and was not found."
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host "     Install it with:  winget install Gyan.FFmpeg"
    } else {
        Write-Host "     See https://ffmpeg.org/download.html"
    }
    Write-Host "     Then re-run this script."
    exit 1
}

# --- 4. models --------------------------------------------------------------
Step "Downloading models for offline use"
Write-Host "  This needs internet once. Afterwards the tool runs with no network."
& .\.venv\Scripts\subgen.exe setup

# --- 5. optional extras -----------------------------------------------------
Step "Optional extras"
if (Get-Command ollama -ErrorAction SilentlyContinue) {
    & .\.venv\Scripts\subgen.exe --list-models > $null 2>&1
    if ($LASTEXITCODE -eq 0) {
        Ok "Ollama is ready for translation"
    } else {
        Warn "Ollama is installed but has no chat models. Pull one to enable translation:"
        Write-Host "      ollama pull qwen3:8b   (any chat model works)"
    }
} else {
    Warn "Ollama is not installed - subtitles will stay in the spoken language."
    Write-Host "     Translation is optional. To enable it, install Ollama yourself:"
    Write-Host "      https://ollama.com/download"
    Write-Host "     We don't install it for you: it registers a background service."
}

Write-Host "`nDone. Start it with:`n" -ForegroundColor Green
Write-Host "    .\.venv\Scripts\Activate.ps1"
Write-Host "    subgen`n"
Write-Host "Or without activating:  .\.venv\Scripts\subgen.exe`n"
