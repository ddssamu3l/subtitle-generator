"""Thin Ollama client: find the daemon, list what the user has, run a chat.

Design note — there is intentionally no list of "approved" model names here.
Model names churn constantly and anything hardcoded today is wrong in a year,
so we show the user whatever they actually have installed and let them choose.
The only filtering we do is structural (does this model do text completion at
all?), which stays true across model generations.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field

import httpx

DEFAULT_HOST = "http://127.0.0.1:11434"

# Long, because a cold model load can take a while on the first call — Ollama
# has to page several GB off disk before it emits a single token.
LOAD_TIMEOUT = 600.0


def resolve_host(explicit: str | None = None) -> str:
    """Where to reach Ollama, honouring the same OLLAMA_HOST the CLI uses."""
    host = explicit or os.environ.get("OLLAMA_HOST") or DEFAULT_HOST
    if not host.startswith(("http://", "https://")):
        host = f"http://{host}"
    return host.rstrip("/")


@dataclass(frozen=True)
class ModelInfo:
    """One installed model, as reported by /api/tags."""

    name: str
    size_bytes: int = 0
    parameter_size: str = ""
    family: str = ""
    quantization: str = ""
    capabilities: tuple[str, ...] = field(default_factory=tuple)

    @property
    def size_label(self) -> str:
        if self.size_bytes <= 0:
            return ""
        gb = self.size_bytes / 1_000_000_000
        return f"{gb:.1f} GB" if gb >= 1 else f"{self.size_bytes / 1_000_000:.0f} MB"

    @property
    def params_label(self) -> str:
        """Parameter count, falling back to the tag when metadata is empty.

        Ollama reports an empty `parameter_size` for some models (notably
        imported safetensors builds like `gemma4:31b-mlx`), so we recover it
        from the tag rather than showing the user a blank column.
        """
        if self.parameter_size:
            return self.parameter_size
        import re

        match = re.search(r"[:\-](\d+(?:\.\d+)?)b\b", self.name, re.IGNORECASE)
        return f"{match.group(1)}B" if match else ""

    def describe(self) -> str:
        """One-line label for the picker, e.g. 'gemma4:latest — 8.0B, 9.6 GB'."""
        bits = [b for b in (self.params_label, self.size_label) if b]
        return f"{self.name} — {', '.join(bits)}" if bits else self.name

    @property
    def can_generate_text(self) -> bool:
        """Exclude models that structurally cannot hold a conversation.

        Embedding and reranking models are listed by /api/tags exactly like
        chat models but will never produce a translation. Ollama tells us
        directly via `capabilities`; when that field is absent (older daemons)
        we fall back to name patterns for the same *roles*, not for specific
        model families — role names outlive model names.
        """
        if self.capabilities:
            return "completion" in self.capabilities
        lowered = self.name.lower()
        role_markers = ("embed", "bge", "minilm", "rerank", "e5-", "gte-")
        return not any(marker in lowered for marker in role_markers)


def _client(host: str, timeout: float) -> httpx.Client:
    return httpx.Client(base_url=host, timeout=timeout)


def is_installed() -> bool:
    """Whether the Ollama CLI exists on PATH, regardless of daemon state."""
    return shutil.which("ollama") is not None


def is_running(host: str | None = None) -> bool:
    """Whether the daemon answers. Cheap enough to call from the GUI."""
    try:
        with _client(resolve_host(host), 2.0) as client:
            return client.get("/api/tags").status_code == 200
    except (httpx.HTTPError, OSError):
        return False


def try_start_daemon() -> bool:
    """Start `ollama serve` in the background if the binary exists.

    Installing Ollama is the user's decision, but launching an already-installed
    daemon is not a meaningful imposition — it is what `ollama run` would do
    anyway. Returns True if the daemon is answering afterwards.
    """
    if not is_installed() or is_running():
        return is_running()
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except (OSError, subprocess.SubprocessError):
        return False

    # Poll rather than sleeping a fixed amount; startup is usually <1s.
    import time

    for _ in range(20):
        time.sleep(0.25)
        if is_running():
            return True
    return False


def list_models(host: str | None = None) -> list[ModelInfo]:
    """Installed models that can generate text, newest-pulled first.

    Ordering is by recency only. We are not ranking quality — the user picks.
    """
    try:
        with _client(resolve_host(host), 10.0) as client:
            response = client.get("/api/tags")
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, OSError, ValueError):
        return []

    models: list[tuple[str, ModelInfo]] = []
    for entry in payload.get("models") or []:
        name = entry.get("name") or entry.get("model")
        if not name:
            continue
        details = entry.get("details") or {}
        info = ModelInfo(
            name=name,
            size_bytes=int(entry.get("size") or 0),
            parameter_size=details.get("parameter_size") or "",
            family=details.get("family") or "",
            quantization=details.get("quantization_level") or "",
            capabilities=tuple(entry.get("capabilities") or ()),
        )
        if info.can_generate_text:
            models.append((entry.get("modified_at") or "", info))

    models.sort(key=lambda pair: pair[0], reverse=True)
    return [info for _, info in models]


class OllamaError(RuntimeError):
    """Raised when a generation call fails in a way worth showing the user."""


def chat(
    model: str,
    messages: list[dict[str, str]],
    *,
    host: str | None = None,
    temperature: float = 0.2,
    num_ctx: int | None = None,
    timeout: float = LOAD_TIMEOUT,
) -> str:
    """Run a non-streaming chat completion and return the assistant text.

    `think` is disabled explicitly: several current models emit chain-of-thought
    by default, which would end up inside the subtitles. Daemons that don't
    understand the option ignore it, so this is safe to always send.
    """
    options: dict[str, object] = {"temperature": temperature}
    if num_ctx:
        options["num_ctx"] = num_ctx

    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": options,
    }

    try:
        with _client(resolve_host(host), timeout) as client:
            response = client.post("/api/chat", json=body)
            if response.status_code == 404:
                raise OllamaError(
                    f"Model '{model}' is not installed. Run: ollama pull {model}"
                )
            response.raise_for_status()
            payload = response.json()
    except httpx.TimeoutException as exc:
        raise OllamaError(
            f"Ollama timed out after {timeout:.0f}s loading or running '{model}'."
        ) from exc
    except httpx.HTTPError as exc:
        raise OllamaError(f"Could not reach Ollama at {resolve_host(host)}: {exc}") from exc
    except ValueError as exc:
        raise OllamaError("Ollama returned a malformed response.") from exc

    content = (payload.get("message") or {}).get("content", "")
    if not content.strip():
        raise OllamaError(f"Model '{model}' returned an empty response.")
    return content


def install_hint() -> str:
    """Platform-appropriate instructions for installing Ollama.

    We never install it automatically: Ollama registers a background service
    and costs about a gigabyte, which is not something a subtitle tool should
    do to someone's machine without being asked.
    """
    if sys.platform == "darwin":
        primary = "brew install ollama" if shutil.which("brew") else None
        lines = ["Install Ollama from https://ollama.com/download"]
        if primary:
            lines.append(f"or, since you have Homebrew:  {primary}")
        return "\n".join(lines)
    if sys.platform.startswith("linux"):
        return "Install Ollama:  curl -fsSL https://ollama.com/install.sh | sh"
    return "Install Ollama from https://ollama.com/download"


def pull_hint(suggestions: list[str] | None = None) -> str:
    """Template commands for getting a first model.

    These are examples to copy, not requirements — any chat-capable model the
    user pulls will show up in the picker. Sizes are approximate and the names
    will age; that is fine, because nothing in the code depends on them.
    """
    examples = suggestions or [
        ("ollama pull qwen3:8b", "~5 GB, strong at Chinese/English"),
        ("ollama pull gemma3:12b", "~8 GB, strong all-round multilingual"),
        ("ollama pull llama3.1:8b", "~5 GB, widely available"),
    ]
    width = max(len(cmd) for cmd, _ in examples)
    return "\n".join(f"  {cmd.ljust(width)}   # {note}" for cmd, note in examples)
