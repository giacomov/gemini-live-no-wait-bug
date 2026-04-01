import { spawn } from "node:child_process";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import React, { useEffect, useState } from "react";
import { render, Box, Text } from "ink";

const __dirname = dirname(fileURLToPath(import.meta.url));

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface RunState {
  status: "running" | "bug" | "pass" | "timeout";
  lines: string[];       // committed conversation lines (scrolling, capped at MAX_LINES)
  pendingModel: string;  // live model turn — shown while streaming, committed on model_turn_end
  pendingModelTurn: number; // turn number for the currently streaming model text
  pendingUser: string;   // user reply buffered silently — committed on model_turn_end, not shown live
  elapsed?: number;
  summary?: string;
}

type PythonEvent =
  | { type: "pregenerating"; text: string }
  | { type: "ready"; n_runs: number; timeout_sec: number; model: string }
  | { type: "run_start"; run_id: number }
  | { type: "transcript"; run_id: number; text: string; turn: number }
  | { type: "model_turn_end"; run_id: number; turn: number }
  | { type: "user_reply"; run_id: number; text: string }
  | { type: "tool_call"; run_id: number; name: string; premature: boolean; turn: number }
  | { type: "run_end"; run_id: number; bug: boolean; timeout: boolean; elapsed: number; summary: string }
  | { type: "summary"; n_runs: number; bug_count: number; pass_count: number; timeout_count: number };

interface AppState {
  phase: "pregenerating" | "ready" | "done";
  pregenLines: string[];
  model: string;
  nRuns: number;
  timeoutSec: number;
  runs: Record<number, RunState>;
  bugCount: number;
  passCount: number;
  timeoutCount: number;
}

const MAX_LINES = 6;
const COLS = 5;
const SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function pushLine(state: RunState, line: string): RunState {
  return { ...state, lines: [...state.lines, line].slice(-MAX_LINES) };
}

/**
 * Commit pendingModel then pendingUser to lines in order.
 * Called on model_turn_end — the model has definitively stopped streaming.
 */
function flushPending(state: RunState): RunState {
  let s = state;
  if (s.pendingModel) {
    s = pushLine({ ...s, pendingModel: "" }, `model[${s.pendingModelTurn}]> ${s.pendingModel}`);
  }
  if (s.pendingUser) {
    s = pushLine({ ...s, pendingUser: "" }, `user>  ${s.pendingUser}`);
  }
  return s;
}

// ---------------------------------------------------------------------------
// Spinner component
// ---------------------------------------------------------------------------

function SpinnerDots() {
  const [frame, setFrame] = useState(0);

  useEffect(() => {
    const id = setInterval(() => {
      setFrame((f) => (f + 1) % SPINNER_FRAMES.length);
    }, 80);
    return () => clearInterval(id);
  }, []);

  return <Text color="cyan">{SPINNER_FRAMES[frame]}</Text>;
}

// ---------------------------------------------------------------------------
// RunBox component
// ---------------------------------------------------------------------------

interface RunBoxProps {
  runId: number;
  state: RunState;
}

function RunBox({ runId, state }: RunBoxProps) {
  const borderColor =
    state.status === "running" ? "yellow"
    : state.status === "bug" ? "red"
    : state.status === "timeout" ? "gray"
    : "green";

  const statusLabel =
    state.status === "running" ? "running"
    : state.status === "bug" ? `BUG (${state.elapsed?.toFixed(1)}s)`
    : state.status === "timeout" ? `TIMEOUT (${state.elapsed?.toFixed(1)}s)`
    : `PASS (${state.elapsed?.toFixed(1)}s)`;

  // Display: committed lines + live model turn (if streaming).
  // pendingUser is intentionally NOT shown — it appears only after model_turn_end commits it.
  const displayLines = [
    ...state.lines,
    ...(state.pendingModel ? [`model[${state.pendingModelTurn}]> ${state.pendingModel}`] : []),
  ].slice(-MAX_LINES);

  return (
    // width="20%" constrains horizontal size; no fixed height — Ink flex-row stretches
    // all boxes in the same row to the same height automatically.
    <Box borderStyle="round" borderColor={borderColor} flexDirection="column" width="20%" paddingX={1}>
      <Box>
        <Text bold color={borderColor}>Run {runId}</Text>
        <Text color={borderColor}> [{statusLabel}]</Text>
      </Box>
      {displayLines.length === 0 ? (
        // Each logical line is wrapped in its own <Box> so it is block-level.
        // Text wraps within the box width; the next <Box> always starts below the wrapped content.
        <Box><Text dimColor>waiting...</Text></Box>
      ) : (
        displayLines.map((line, i) => (
          <Box key={i}>
            <Text dimColor={line.startsWith("model[")}>{line}</Text>
          </Box>
        ))
      )}
    </Box>
  );
}

// ---------------------------------------------------------------------------
// Event reducer
// ---------------------------------------------------------------------------

function applyEvent(prev: AppState, event: PythonEvent): AppState {
  switch (event.type) {
    case "pregenerating":
      return { ...prev, pregenLines: [...prev.pregenLines, event.text] };

    case "ready":
      return {
        ...prev,
        phase: "ready",
        model: event.model,
        nRuns: event.n_runs,
        timeoutSec: event.timeout_sec,
      };

    case "run_start": {
      const run: RunState = { status: "running", lines: [], pendingModel: "", pendingModelTurn: 0, pendingUser: "" };
      return { ...prev, runs: { ...prev.runs, [event.run_id]: run } };
    }

    case "transcript": {
      const run = prev.runs[event.run_id];
      if (!run) return prev;
      // Concatenate directly — Gemini includes its own word spacing in each chunk
      return {
        ...prev,
        runs: { ...prev.runs, [event.run_id]: { ...run, pendingModel: run.pendingModel + event.text, pendingModelTurn: event.turn } },
      };
    }

    case "model_turn_end": {
      const run = prev.runs[event.run_id];
      if (!run) return prev;
      // All transcript for this turn is in — commit model text then user reply in order
      return {
        ...prev,
        runs: { ...prev.runs, [event.run_id]: flushPending(run) },
      };
    }

    case "user_reply": {
      const run = prev.runs[event.run_id];
      if (!run) return prev;
      if (run.pendingModel) {
        // Model is still streaming — buffer silently until model_turn_end guarantees ordering
        return {
          ...prev,
          runs: { ...prev.runs, [event.run_id]: { ...run, pendingUser: event.text } },
        };
      } else {
        // model_turn_end already fired (rare race) — commit directly
        return {
          ...prev,
          runs: { ...prev.runs, [event.run_id]: pushLine(run, `user>  ${event.text}`) },
        };
      }
    }

    case "tool_call": {
      const run = prev.runs[event.run_id];
      if (!run) return prev;
      let s = flushPending(run);
      if (event.premature) {
        s = pushLine(s, `!! ${event.name}[${event.turn}]`);
      } else {
        s = pushLine(s, `✓ ${event.name}[${event.turn}]`);
      }
      return { ...prev, runs: { ...prev.runs, [event.run_id]: s } };
    }

    case "run_end": {
      const run = prev.runs[event.run_id];
      if (!run) return prev;
      const flushed = flushPending(run);
      const status = event.bug ? "bug" : event.timeout ? "timeout" : "pass";
      return {
        ...prev,
        bugCount: prev.bugCount + (event.bug ? 1 : 0),
        timeoutCount: prev.timeoutCount + (event.timeout && !event.bug ? 1 : 0),
        passCount: prev.passCount + (!event.bug && !event.timeout ? 1 : 0),
        runs: {
          ...prev.runs,
          [event.run_id]: { ...flushed, status, elapsed: event.elapsed, summary: event.summary },
        },
      };
    }

    case "summary":
      return prev; // already tracked via run_end events
  }
}

// ---------------------------------------------------------------------------
// App component
// ---------------------------------------------------------------------------

function App() {
  const [state, setState] = useState<AppState>({
    phase: "pregenerating",
    pregenLines: [],
    model: "",
    nRuns: 0,
    timeoutSec: 0,
    runs: {},
    bugCount: 0,
    passCount: 0,
    timeoutCount: 0,
  });

  useEffect(() => {
    const py = spawn("uv", ["run", "python", "reproduce_no_wait_bug.py"], {
      cwd: resolve(__dirname, "../backend"),
      env: process.env,
    });

    const rl = createInterface({ input: py.stdout });

    rl.on("line", (raw: string) => {
      let event: PythonEvent;
      try {
        event = JSON.parse(raw) as PythonEvent;
      } catch {
        return;
      }
      setState((prev) => applyEvent(prev, event));
    });

    py.on("close", () => {
      setState((prev) => ({ ...prev, phase: "done" }));
    });

    return () => {
      py.kill();
    };
  }, []);

  if (state.phase === "pregenerating") {
    return (
      <Box flexDirection="column" padding={1}>
        <Box gap={1}>
          <SpinnerDots />
          <Text>Generating audio...</Text>
        </Box>
        {state.pregenLines.map((l, i) => (
          <Text key={i} dimColor>{l}</Text>
        ))}
      </Box>
    );
  }

  const runIds = Object.keys(state.runs)
    .map(Number)
    .sort((a, b) => a - b);

  const rows: number[][] = [];
  for (let i = 0; i < runIds.length; i += COLS) {
    rows.push(runIds.slice(i, i + COLS));
  }

  return (
    <Box flexDirection="column">
      <Box paddingX={1} marginBottom={1} gap={2}>
        <Text><Text bold>Model:</Text> {state.model}</Text>
        <Text><Text bold>Runs:</Text> {state.nRuns}</Text>
        <Text><Text bold>Timeout:</Text> {state.timeoutSec}s</Text>
        <Text bold color="red">BUG: {state.bugCount}</Text>
        <Text bold color="green">PASS: {state.passCount}</Text>
        <Text bold color="gray">TIMEOUT: {state.timeoutCount}</Text>
      </Box>

      {rows.map((row, rowIdx) => (
        <Box key={rowIdx} flexDirection="row">
          {row.map((id) => (
            <RunBox key={id} runId={id} state={state.runs[id]!} />
          ))}
        </Box>
      ))}

      {state.phase === "done" && runIds.length > 0 && (
        <Box paddingX={1} marginTop={1}>
          <Text bold>Done. </Text>
          <Text>BUG: {state.bugCount}{"  "}PASS: {state.passCount}{"  "}TIMEOUT: {state.timeoutCount}{"  "}(of {state.nRuns})</Text>
        </Box>
      )}
    </Box>
  );
}

// ---------------------------------------------------------------------------
// Entry
// ---------------------------------------------------------------------------

render(<App />);
