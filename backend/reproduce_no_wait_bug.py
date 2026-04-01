"""
Reproduce the Gemini "no-wait" bug: the model calls `continue_on` without
waiting for the user to answer "Are you ready?".

Usage:
    uv run python reproduce_no_wait_bug.py

Env vars (via .env):
    GOOGLE_CLOUD_PROJECT    Vertex AI project
    GOOGLE_CLOUD_LOCATION   Vertex AI region (e.g. us-central1)
    BUG_RUNS                Number of repetitions (default: 10)
    BUG_TIMEOUT_SEC         Seconds to wait before declaring PASS (default: 30)
"""

import asyncio
import json
import multiprocessing
import os
import time
from multiprocessing import Queue
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from audio_utils import AudioConverter  # noqa: E402
from gemini_config import MODEL
from session import pregenerate_audio, run_single


def _worker(run_id: int, audio_replies: list[bytes], timeout_sec: float, queue: Queue) -> None:
    """Entry point for each worker process. Runs one session and puts events into the queue."""

    def on_event(event: dict) -> None:
        queue.put({"run_id": run_id, **event})

    t0 = time.monotonic()
    try:
        result = asyncio.run(run_single(audio_replies, timeout_sec, on_event=on_event))
    except Exception as exc:
        result = {"bug": False, "transcript": str(exc)}
    elapsed = time.monotonic() - t0

    on_event({
        "type": "run_end",
        "bug": bool(result.get("bug")),
        "timeout": bool(result.get("timeout")),
        "elapsed": round(elapsed, 2),
        "summary": str(result.get("summary", "")),
        "transcript": str(result.get("transcript", "")),
    })


def emit(event: dict) -> None:
    print(json.dumps(event), flush=True)


def main() -> None:
    n_runs = int(os.environ.get("BUG_RUNS", "10"))
    timeout_sec = float(os.environ.get("BUG_TIMEOUT_SEC", "120"))

    converter = AudioConverter()
    audio_replies = pregenerate_audio(converter, on_event=emit)

    emit({"type": "ready", "n_runs": n_runs, "timeout_sec": timeout_sec, "model": MODEL})

    ctx = multiprocessing.get_context("spawn")
    queue: Queue = ctx.Queue()

    processes = []
    for run_id in range(1, n_runs + 1):
        emit({"type": "run_start", "run_id": run_id})
        p = ctx.Process(target=_worker, args=(run_id, audio_replies, timeout_sec, queue))
        p.start()
        processes.append(p)

    finished = 0
    bug_count = 0
    timeout_count = 0
    while finished < n_runs:
        event = queue.get()
        emit(event)
        if event.get("type") == "run_end":
            finished += 1
            if event.get("bug"):
                bug_count += 1
            elif event.get("timeout"):
                timeout_count += 1

    for p in processes:
        p.join()

    pass_count = n_runs - bug_count - timeout_count
    emit({
        "type": "summary",
        "n_runs": n_runs,
        "bug_count": bug_count,
        "pass_count": pass_count,
        "timeout_count": timeout_count,
    })


if __name__ == "__main__":
    main()
