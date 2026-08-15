"""Tkinter front end: pick videos, choose a language each, watch it run.

Tkinter is used rather than a nicer toolkit for one reason — it ships with
Python. Adding PyQt or wxPython would mean a large binary dependency and a
platform-specific install story, which works against the goal of the repo being
clone-and-run for someone who is not a developer.

Threading rule: Tk is not thread-safe, so the pipeline runs on a worker thread
and communicates only by putting messages on a queue that the UI drains from
`after()` on the main thread. No widget is ever touched off the main thread.
"""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from . import config, diarize, formats, languages, media, models, ollama, pipeline, transcribe
from .languages import Language

PAD = 10


def pick_files(parent: tk.Misc | None = None) -> list[Path]:
    """Open the native file chooser, filtered to formats we can process."""
    selection = filedialog.askopenfilenames(
        parent=parent,
        title="Choose video files to subtitle",
        filetypes=formats.tk_filetypes(),
    )
    # Filter again: the "All files" entry lets people pick anything, and the
    # error is far friendlier here than three minutes into an ffmpeg failure.
    return [Path(item) for item in selection if formats.is_supported(item)]


class App:
    def __init__(self) -> None:
        self.settings = config.load()
        self.root = tk.Tk()
        self.root.title("Subtitle Generator")
        self.root.minsize(760, 520)

        self.videos: list[Path] = []
        self.language_vars: dict[Path, tk.StringVar] = {}
        # Built on the main thread in _start and read by the worker, because
        # Tk variables must not be touched off the main thread.
        self._options_snapshot: dict[Path, pipeline.Options] = {}

        self.queue: queue.Queue = queue.Queue()
        self.cancel_event = threading.Event()
        self.worker: threading.Thread | None = None

        self.setup_frame = ttk.Frame(self.root, padding=PAD)
        self.progress_frame = ttk.Frame(self.root, padding=PAD)

        self._build_setup()

    # --- setup screen -------------------------------------------------------

    def _build_setup(self) -> None:
        frame = self.setup_frame
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        header = ttk.Frame(frame)
        header.grid(row=0, column=0, sticky="ew", pady=(0, PAD))
        header.columnconfigure(0, weight=1)

        ttk.Label(
            header, text="Videos", font=("", 13, "bold")
        ).grid(row=0, column=0, sticky="w")
        ttk.Button(header, text="Choose videos…", command=self._add_videos).grid(
            row=0, column=1, padx=(PAD, 0)
        )
        ttk.Button(header, text="Remove all", command=self._clear_videos).grid(
            row=0, column=2, padx=(6, 0)
        )

        # Scrollable list of videos, one row per file.
        container = ttk.Frame(frame, relief="solid", borderwidth=1)
        container.grid(row=1, column=0, sticky="nsew")
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(container, highlightthickness=0, height=200)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        self.rows = ttk.Frame(self.canvas)

        self.rows.bind(
            "<Configure>",
            lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas_window = self.canvas.create_window((0, 0), window=self.rows, anchor="nw")
        self.canvas.bind(
            "<Configure>",
            lambda event: self.canvas.itemconfigure(self.canvas_window, width=event.width),
        )
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        self._build_options(frame)
        self._build_warnings(frame)

        actions = ttk.Frame(frame)
        actions.grid(row=4, column=0, sticky="ew", pady=(PAD, 0))
        actions.columnconfigure(0, weight=1)

        self.start_button = ttk.Button(
            actions, text="Generate subtitles", command=self._start
        )
        self.start_button.grid(row=0, column=1, sticky="e")
        ttk.Button(actions, text="Quit", command=self.root.destroy).grid(
            row=0, column=0, sticky="w"
        )

        self._refresh_rows()

    def _build_options(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Options", padding=PAD)
        box.grid(row=2, column=0, sticky="ew", pady=(PAD, 0))
        box.columnconfigure(1, weight=1)
        box.columnconfigure(3, weight=1)

        # Row 0 — translation model.
        ttk.Label(box, text="Translation model:").grid(row=0, column=0, sticky="w")
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(
            box, textvariable=self.model_var, state="readonly", width=34
        )
        self.model_combo.grid(row=0, column=1, sticky="ew", padx=(6, PAD))

        ttk.Button(box, text="Refresh", width=8, command=self._refresh_models).grid(
            row=0, column=2, sticky="w"
        )

        # Row 1 — transcription quality.
        ttk.Label(box, text="Transcription quality:").grid(
            row=1, column=0, sticky="w", pady=(6, 0)
        )
        self.whisper_var = tk.StringVar(value=self.settings["whisper_model"])
        whisper_combo = ttk.Combobox(
            box,
            textvariable=self.whisper_var,
            values=list(transcribe.SUGGESTED_MODELS),
            width=34,
        )
        whisper_combo.grid(row=1, column=1, sticky="ew", padx=(6, PAD), pady=(6, 0))

        ttk.Label(
            box, text=transcribe.active_backend(), foreground="#666"
        ).grid(row=1, column=2, columnspan=2, sticky="w", pady=(6, 0))

        # Row 2 — checkboxes.
        self.speakers_var = tk.BooleanVar(value=bool(self.settings["identify_speakers"]))
        self.speakers_check = ttk.Checkbutton(
            box,
            text="Stack lines when a second speaker interrupts",
            variable=self.speakers_var,
        )
        self.speakers_check.grid(row=2, column=0, columnspan=3, sticky="w", pady=(8, 0))

        self.sidecar_var = tk.BooleanVar(value=bool(self.settings["keep_sidecar_files"]))
        ttk.Checkbutton(
            box,
            text="Also save .ass and .srt files next to the video",
            variable=self.sidecar_var,
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(2, 0))

        self._refresh_models()

    def _build_warnings(self, parent: ttk.Frame) -> None:
        self.warning_box = ttk.Frame(parent)
        self.warning_box.grid(row=3, column=0, sticky="ew", pady=(PAD, 0))
        self.warning_box.columnconfigure(0, weight=1)
        self._refresh_warnings()

    # --- dynamic content ----------------------------------------------------

    def _refresh_models(self) -> None:
        """Populate the model dropdown from whatever Ollama actually has.

        No model name is baked in — the list is exactly what is installed, so
        this keeps working as models come and go.
        """
        if not ollama.is_running():
            ollama.try_start_daemon()

        self.available_models = ollama.list_models(self.settings.get("ollama_host"))
        labels = [model.describe() for model in self.available_models]

        self.model_combo["values"] = labels or ["(no models found)"]

        remembered = self.settings.get("ollama_model")
        chosen = next(
            (m.describe() for m in self.available_models if m.name == remembered),
            labels[0] if labels else "(no models found)",
        )
        self.model_var.set(chosen)
        self.model_combo.configure(state="readonly" if labels else "disabled")

        if hasattr(self, "warning_box"):
            self._refresh_warnings()

    def selected_model(self) -> str | None:
        label = self.model_var.get()
        for model in getattr(self, "available_models", []):
            if model.describe() == label:
                return model.name
        return None

    def _refresh_warnings(self) -> None:
        for child in self.warning_box.winfo_children():
            child.destroy()

        row = 0

        if not media.ffmpeg_path():
            row = self._warning(
                row,
                "ffmpeg is required and was not found.",
                media.install_hint(),
                action=("Install it for me", self._install_ffmpeg)
                if media.install_command()
                else None,
            )

        if not ollama.is_installed():
            row = self._warning(
                row,
                "Ollama is not installed — subtitles will stay in the spoken language.",
                ollama.install_hint()
                + "\n\nInstalling Ollama registers a background service, so we "
                "won't do it for you. The tool still works without it.",
                action=("Open download page", lambda: webbrowser.open("https://ollama.com/download")),
            )
        elif not getattr(self, "available_models", []):
            row = self._warning(
                row,
                "Ollama has no chat models installed yet.",
                "Pull one to enable translation — any chat model works:\n\n"
                + ollama.pull_hint()
                + "\n\nThen press Refresh.",
                action=("Copy first command", self._copy_pull_command),
            )

        if not diarize.is_available() and self.speakers_var.get():
            row = self._warning(
                row,
                "Speaker models are not downloaded (~35 MB).",
                "Line stacking needs them. Run this once while online:\n"
                "  subgen setup",
                action=("Download now", self._download_speaker_models),
            )

    def _warning(self, row: int, title: str, detail: str, action=None) -> int:
        frame = ttk.Frame(self.warning_box, relief="solid", borderwidth=1, padding=8)
        frame.grid(row=row, column=0, sticky="ew", pady=(0, 6))
        frame.columnconfigure(0, weight=1)

        ttk.Label(frame, text=title, font=("", 11, "bold"), foreground="#8a5a00").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(frame, text=detail, justify="left", foreground="#444").grid(
            row=1, column=0, sticky="w", pady=(2, 0)
        )
        if action:
            label, command = action
            ttk.Button(frame, text=label, command=command).grid(
                row=0, column=1, rowspan=2, padx=(PAD, 0)
            )
        return row + 1

    def _copy_pull_command(self) -> None:
        command = ollama.pull_hint().splitlines()[0].strip().split("   #")[0].strip()
        self.root.clipboard_clear()
        self.root.clipboard_append(command)
        messagebox.showinfo(
            "Copied",
            f"Copied to your clipboard:\n\n{command}\n\n"
            "Paste it into a terminal, then press Refresh.",
            parent=self.root,
        )

    def _install_ffmpeg(self) -> None:
        command = media.install_command()
        if not command:
            return
        if not messagebox.askyesno(
            "Install ffmpeg",
            "This will run:\n\n  " + " ".join(command) + "\n\nProceed?",
            parent=self.root,
        ):
            return
        try:
            subprocess.run(command, check=True)
        except (OSError, subprocess.SubprocessError) as exc:
            messagebox.showerror("Install failed", str(exc), parent=self.root)
        self._refresh_warnings()

    def _download_speaker_models(self) -> None:
        window = tk.Toplevel(self.root)
        window.title("Downloading speaker models")
        window.transient(self.root)
        label = ttk.Label(window, text="Starting…", padding=PAD)
        label.pack()
        bar = ttk.Progressbar(window, length=320, mode="determinate", maximum=100)
        bar.pack(padx=PAD, pady=(0, PAD))

        outcome: dict[str, str] = {}

        def work() -> None:
            models.allow_downloads()
            try:
                models.fetch_diarization(
                    progress=lambda fraction, text: self.queue.put(
                        ("download", fraction, text)
                    )
                )
                outcome["ok"] = "done"
            except models.DownloadError as exc:
                outcome["error"] = str(exc)
            finally:
                models.enforce_offline()
                self.queue.put(("download-done", 1.0, ""))

        def poll() -> None:
            try:
                while True:
                    kind, fraction, text = self.queue.get_nowait()
                    if kind == "download":
                        bar["value"] = fraction * 100
                        label.configure(text=text)
                    elif kind == "download-done":
                        window.destroy()
                        if "error" in outcome:
                            messagebox.showerror(
                                "Download failed", outcome["error"], parent=self.root
                            )
                        self._refresh_warnings()
                        return
            except queue.Empty:
                pass
            window.after(100, poll)

        threading.Thread(target=work, daemon=True).start()
        window.after(100, poll)

    def _add_videos(self) -> None:
        chosen = pick_files(self.root)
        added = [path for path in chosen if path not in self.videos]
        self.videos.extend(added)
        self._refresh_rows()

    def _clear_videos(self) -> None:
        self.videos.clear()
        self.language_vars.clear()
        self._refresh_rows()

    def _refresh_rows(self) -> None:
        for child in self.rows.winfo_children():
            child.destroy()

        if not self.videos:
            # Empty state carries its own call to action, so the first thing to
            # do is obvious without hunting for the button in the header.
            empty = ttk.Frame(self.rows, padding=28)
            empty.pack(expand=True)

            ttk.Label(
                empty,
                text="No videos chosen yet",
                font=("", 12),
                foreground="#666",
            ).pack()
            ttk.Button(
                empty, text="Choose videos…", command=self._add_videos
            ).pack(pady=(12, 10))
            ttk.Label(
                empty,
                text=f"Supported formats: {formats.describe()}",
                justify="center",
                foreground="#999",
                font=("", 9),
                wraplength=460,
            ).pack()

            self.start_button.state(["disabled"])
            return

        self.start_button.state(["!disabled"])
        self.rows.columnconfigure(0, weight=1)

        default_language = self.settings["target_language"]

        for index, path in enumerate(self.videos):
            row = ttk.Frame(self.rows, padding=(8, 6))
            row.grid(row=index, column=0, sticky="ew")
            row.columnconfigure(0, weight=1)

            ttk.Label(row, text=path.name).grid(row=0, column=0, sticky="w")
            ttk.Label(
                row, text=str(path.parent), foreground="#888", font=("", 9)
            ).grid(row=1, column=0, sticky="w")

            variable = self.language_vars.get(path) or tk.StringVar(value=default_language)
            self.language_vars[path] = variable

            ttk.Label(row, text="Subtitle language:").grid(row=0, column=1, rowspan=2, padx=(PAD, 4))
            combo = ttk.Combobox(
                row,
                textvariable=variable,
                values=languages.label_choices(),
                state="readonly",
                width=26,
            )
            combo.grid(row=0, column=2, rowspan=2)

            ttk.Button(
                row, text="✕", width=3,
                command=lambda p=path: self._remove(p),
            ).grid(row=0, column=3, rowspan=2, padx=(6, 0))

            if index == 0 and len(self.videos) > 1:
                ttk.Button(
                    row, text="Apply to all", width=11,
                    command=lambda v=variable: self._apply_to_all(v.get()),
                ).grid(row=0, column=4, rowspan=2, padx=(6, 0))

    def _apply_to_all(self, value: str) -> None:
        for variable in self.language_vars.values():
            variable.set(value)

    def _remove(self, path: Path) -> None:
        self.videos = [item for item in self.videos if item != path]
        self.language_vars.pop(path, None)
        self._refresh_rows()

    # --- running ------------------------------------------------------------

    def _build_options_for(self, path: Path) -> pipeline.Options:
        target = languages.resolve(self.language_vars[path].get())
        return pipeline.Options(
            target_language=target,
            whisper_model=self.whisper_var.get().strip() or transcribe.DEFAULT_MODEL,
            ollama_model=self.selected_model(),
            ollama_host=self.settings.get("ollama_host"),
            identify_speakers=self.speakers_var.get(),
            keep_sidecar_files=self.sidecar_var.get(),
        )

    def _start(self) -> None:
        if not self.videos:
            return

        first_options = self._build_options_for(self.videos[0])
        problems = pipeline.preflight(first_options)

        # A missing translation model is a downgrade, not a blocker — offer to
        # continue with untranslated subtitles rather than refusing to run.
        blocking = [p for p in problems if "translation model" not in p]
        if blocking:
            messagebox.showerror(
                "Cannot start yet", "\n\n".join(blocking), parent=self.root
            )
            return

        if len(problems) > len(blocking):
            wants_translation = any(
                languages.resolve(var.get()) for var in self.language_vars.values()
            )
            if wants_translation and not messagebox.askyesno(
                "No translation model",
                "No Ollama model is available, so subtitles will stay in the "
                "language that was spoken.\n\nContinue anyway?",
                parent=self.root,
            ):
                return

        self._remember_choices()
        self._options_snapshot = {path: self._build_options_for(path) for path in self.videos}
        self._show_progress()

        self.cancel_event.clear()
        self.worker = threading.Thread(target=self._run_all, daemon=True)
        self.worker.start()
        self.root.after(100, self._drain_queue)

    def _remember_choices(self) -> None:
        model = self.selected_model()
        self.settings.update(
            {
                "ollama_model": model,
                "whisper_model": self.whisper_var.get().strip(),
                "identify_speakers": self.speakers_var.get(),
                "keep_sidecar_files": self.sidecar_var.get(),
                "target_language": self.language_vars[self.videos[0]].get(),
            }
        )
        config.save(self.settings)

    def _show_progress(self) -> None:
        self.setup_frame.pack_forget()
        self.progress_frame.pack(fill="both", expand=True)
        self.progress_frame.columnconfigure(0, weight=1)
        self.progress_frame.rowconfigure(3, weight=1)

        self.file_label = ttk.Label(self.progress_frame, text="Starting…", font=("", 12, "bold"))
        self.file_label.grid(row=0, column=0, sticky="w")

        self.stage_label = ttk.Label(self.progress_frame, text="", foreground="#555")
        self.stage_label.grid(row=1, column=0, sticky="w", pady=(2, 6))

        self.bar = ttk.Progressbar(self.progress_frame, mode="determinate", maximum=100)
        self.bar.grid(row=2, column=0, sticky="ew")

        self.log = tk.Text(self.progress_frame, height=12, wrap="word", state="disabled")
        self.log.grid(row=3, column=0, sticky="nsew", pady=(PAD, 0))

        buttons = ttk.Frame(self.progress_frame)
        buttons.grid(row=4, column=0, sticky="ew", pady=(PAD, 0))
        buttons.columnconfigure(0, weight=1)

        self.cancel_button = ttk.Button(buttons, text="Cancel", command=self._cancel)
        self.cancel_button.grid(row=0, column=0, sticky="w")

        self.done_button = ttk.Button(buttons, text="Close", command=self.root.destroy)
        self.done_button.grid(row=0, column=1, sticky="e")
        self.done_button.state(["disabled"])

    def _cancel(self) -> None:
        self.cancel_event.set()
        self.cancel_button.state(["disabled"])
        self._append_log("Cancelling after the current step…")

    def _run_all(self) -> None:
        total = len(self.videos)
        for index, path in enumerate(self.videos, start=1):
            if self.cancel_event.is_set():
                break

            self.queue.put(("file", index / total, f"[{index}/{total}] {path.name}"))
            options = self._options_snapshot[path]

            result = pipeline.process(
                path,
                options,
                report=lambda fraction, text: self.queue.put(("progress", fraction, text)),
                cancel=self.cancel_event,
            )
            self.queue.put(("result", 1.0, _describe(result)))

        self.queue.put(("all-done", 1.0, ""))

    def _drain_queue(self) -> None:
        try:
            while True:
                kind, fraction, text = self.queue.get_nowait()
                if kind == "file":
                    self.file_label.configure(text=text)
                    self.bar["value"] = 0
                elif kind == "progress":
                    self.bar["value"] = fraction * 100
                    self.stage_label.configure(text=text)
                elif kind == "result":
                    self._append_log(text)
                elif kind == "all-done":
                    self.stage_label.configure(text="Finished.")
                    self.bar["value"] = 100
                    self.cancel_button.state(["disabled"])
                    self.done_button.state(["!disabled"])
                    return
        except queue.Empty:
            pass
        self.root.after(100, self._drain_queue)

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def run(self) -> None:
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._bring_to_front()
        self.root.mainloop()

    def _bring_to_front(self) -> None:
        """Make sure the window appears in front of the terminal that launched it.

        A Tk window started from a terminal on macOS routinely opens behind it,
        which looks like the command silently did nothing. Raising it briefly
        with topmost, then releasing, puts it in front without pinning it above
        everything else for the rest of the session.
        """
        try:
            self.root.update_idletasks()
            self.root.lift()
            self.root.attributes("-topmost", True)
            self.root.after(200, lambda: self.root.attributes("-topmost", False))
            self.root.focus_force()
        except tk.TclError:
            # Cosmetic only — never let window management stop the app running.
            pass

    def _on_close(self) -> None:
        self.cancel_event.set()
        self.root.destroy()


def _describe(result: pipeline.Result) -> str:
    if result.error:
        return f"✗ {result.source.name}: {result.error}"

    bits = [f"✓ {result.source.name} → {result.output.name if result.output else '?'}"]
    detail = [f"{result.cue_count} lines"]
    if result.speaker_count:
        detail.append(f"{result.speaker_count} speakers")
    if result.translated:
        detail.append("translated")
    if result.detected_language:
        detail.append(f"heard {result.detected_language}")
    bits.append("   " + ", ".join(detail))
    return "\n".join(bits)


def run() -> int:
    """Entry point used by `subgen` with no arguments."""
    try:
        app = App()
    except tk.TclError as exc:
        print(
            "Could not open a window. If you are on a headless machine, use the "
            f"command line instead:  subgen --help\n\nDetail: {exc}",
            file=sys.stderr,
        )
        return 1

    app.run()
    return 0
