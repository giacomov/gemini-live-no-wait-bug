# gemini-live-bug

Reproducer for a Gemini Live API bug where the instructions say to ask a question, and the model asks it but then does not wait for an answer. In this example, it immediately call a tool to close the conversation (`continue_on`) without waiting for the user to reply. In other words, the model asks "Are you ready to continue?" then immediately fires the tool call before the user has a chance to answer.

This bug is reported in several places, like here: https://github.com/google-gemini/live-api-web-console/issues/139.

<img width="1283" height="495" alt="Screenshot 2026-03-31 at 9 17 56 PM" src="https://github.com/user-attachments/assets/5b154ef7-c48b-4fd5-bdac-cce1335baf7e" />

## What it does

Each run opens a Gemini Live session, waits for the model to finish its opening turn, sends a scripted audio reply ("yes"), and checks whether `continue_on` was called before or after the reply was sent. A **BUG** result means the tool fired prematurely; a **PASS** means it correctly waited.

Multiple runs are executed in parallel as separate processes to increase the chance of triggering the race. A terminal UI shows each run's live transcript and final status.

## Setup

**Prerequisites:** Python 3.13+, Node.js, a Google Cloud project with Vertex AI enabled and the Gemini Live API accessible.

```bash
# Python dependencies
cd backend && uv sync && cd ..

# Node dependencies (for the terminal UI)
cd frontend && npm install && cd ..

# Configure credentials
cp env.example .env
# Edit .env and set GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION
gcloud auth application-default login
```

## Usage

**Terminal UI** (recommended — shows all runs live):

```bash
cd frontend && npm start
```

**Headless** (plain JSON lines to stdout):

```bash
cd backend && uv run python reproduce_no_wait_bug.py
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_CLOUD_PROJECT` | — | Vertex AI project ID |
| `GOOGLE_CLOUD_LOCATION` | `us-central1` | Vertex AI region |
| `BUG_RUNS` | `10` | Number of parallel sessions to run |
| `BUG_TIMEOUT_SEC` | `120` | Seconds before declaring a run a PASS by timeout |
| `POCKET_TTS_VOICE` | `alba` | Voice used for audio replies |

## Project layout

```
.
├── .env                         # credentials (not committed)
├── env.example
├── backend/
│   ├── pyproject.toml
│   ├── gemini_config.py         # model name, system prompt, tool declaration, client factory
│   ├── audio_utils.py           # TTS conversion and audio streaming to Gemini
│   ├── session.py               # single-session driver: concurrent send/receive, bug detection
│   └── reproduce_no_wait_bug.py # multiprocess harness; emits JSON events to stdout
└── frontend/
    ├── package.json
    └── ui.tsx                   # Ink terminal UI consuming the JSON event stream
```
