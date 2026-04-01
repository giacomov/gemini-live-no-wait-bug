"""Session driver: manages a single Gemini Live conversation run."""

import asyncio
from collections.abc import Callable

from google.genai.live import AsyncSession

from audio_utils import AudioConverter
from gemini_config import CONFIG, MODEL, make_client

# Scripted user replies sent after each model turn, in order.
# The last reply is the closing one; after it continue_on is expected (PASS).
REPLY_TEXTS: list[str] = ["yes"]


class BugFound(Exception):
    def __init__(self, summary: str, transcript: str) -> None:
        self.summary = summary
        self.transcript = transcript


def pregenerate_audio(
    converter: AudioConverter,
    on_event: Callable[[dict], None] | None = None,
) -> list[bytes]:
    """Convert all scripted replies to PCM16 audio at 16kHz once at startup."""
    if on_event:
        on_event({"type": "pregenerating", "text": "Pre-generating audio for user replies (first run may take a few minutes to download TTS model weights)..."})
    else:
        print("Pre-generating audio for user replies...", flush=True)

    audio_clips = []
    for text in REPLY_TEXTS:
        audio_clips.append(converter.text_to_audio_bytes(text))
        if on_event:
            on_event({"type": "pregenerating", "text": f"generated: {text!r}"})
        else:
            print(f"  generated: {text!r}", flush=True)

    if not on_event:
        print(flush=True)

    return audio_clips


async def run_session(
    session: AsyncSession,
    audio_replies: list[bytes],
    timeout_sec: float,
    converter: AudioConverter,
    on_event: Callable[[dict], None] | None = None,
) -> None:
    """Drive the conversation with two concurrent tasks:

    - _receive: continuously processes turns from Gemini, detects premature continue_on
    - _send: waits for each turn_complete signal, then sends the next user audio reply

    Concurrency is essential: if the model fires continue_on in a new turn immediately
    after asking a question (the bug), _receive catches it even while _send is still
    waiting or uploading audio.
    """
    transcript_parts: list[str] = []
    pending_replies = list(zip(REPLY_TEXTS, audio_replies))

    model_idle = asyncio.Event()
    user_has_replied = asyncio.Event()

    turn_number = 0

    async def _receive() -> None:
        nonlocal turn_number

        while True:
            turn_number += 1
            async for response in session.receive():

                model_idle.clear()

                if response.tool_call:
                    for fn_call in response.tool_call.function_calls or []:
                        premature = not user_has_replied.is_set()
                        if on_event:
                            on_event({"type": "tool_call", "name": fn_call.name, "premature": premature, "turn": turn_number})
                        else:
                            print(f"  [turn {turn_number}] [tool_call] {fn_call.name} (premature={premature})", flush=True)
                        if fn_call.name == "continue_on":
                            if premature:
                                args = fn_call.args or {}
                                summary = str(args.get("conversation_summary", ""))
                                raise BugFound(summary, " ".join(transcript_parts).strip())
                            else:
                                return  # correctly waited for all replies — PASS

                if response.server_content:
                    sc = response.server_content
                    if sc.output_transcription and sc.output_transcription.text:
                        transcript_parts.append(sc.output_transcription.text)
                        if on_event:
                            on_event({"type": "transcript", "text": sc.output_transcription.text, "turn": turn_number})
                        else:
                            print(f"  [turn {turn_number}] [transcript] {sc.output_transcription.text!r}", flush=True)

            # Generated question, now waiting for the answer
            model_idle.set()
            if on_event:
                on_event({"type": "model_turn_end", "turn": turn_number})

    async def _send() -> None:

        while pending_replies:

            await model_idle.wait()

            text, audio = pending_replies.pop(0)
            completed = await converter.send_audio(session, audio, model_idle)

            if not completed:
                return

            if on_event:
                on_event({"type": "user_reply", "text": text})
            else:
                print(f"  [user] {text!r}", flush=True)

            user_has_replied.set()
            model_idle.clear()  # reset for next round

    async with asyncio.timeout(timeout_sec):
        async with asyncio.TaskGroup() as tg:
            tg.create_task(_receive())
            tg.create_task(_send())


async def run_single(
    audio_replies: list[bytes],
    timeout_sec: float,
    on_event: Callable[[dict], None] | None = None,
) -> dict[str, str | bool]:
    client = make_client()

    async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:
        await session.send_realtime_input(text="Hello again!")

        converter = AudioConverter()
        try:
            await run_session(session, audio_replies, timeout_sec, converter, on_event=on_event)
        except asyncio.TimeoutError:
            return {"bug": False, "timeout": True, "transcript": ""}
        except BaseExceptionGroup as eg:
            bug_excs = [e for e in eg.exceptions if isinstance(e, BugFound)]
            if bug_excs:
                exc = bug_excs[0]
                return {"bug": True, "summary": exc.summary, "transcript": exc.transcript}
            raise

    return {"bug": False, "transcript": ""}
