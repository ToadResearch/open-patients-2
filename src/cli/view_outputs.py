#!/usr/bin/env python3
"""Simple local web viewer for Open-Patients JSONL enrichment outputs."""

from __future__ import annotations

import argparse
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from ..utils.utils import print_header


def _is_data_shard_file(name: str) -> bool:
    return name.startswith("data_shard") and name.endswith(".jsonl")


def _is_within(base: Path, value: Path) -> bool:
    try:
        value.relative_to(base)
        return True
    except Exception:
        return False


def _safe_list_dirs(parent: Path) -> list[Path]:
    try:
        return sorted([p for p in parent.iterdir() if p.is_dir()], key=lambda p: p.name)
    except Exception:
        return []


def _has_metadata(run_dir: Path) -> bool:
    try:
        return any(run_dir.glob("run_metadata*.json"))
    except Exception:
        return False


def _iter_shard_files_direct(run_dir: Path):
    for base in (run_dir, run_dir / "shards"):
        if not base.is_dir():
            continue
        try:
            for p in base.glob("data_shard*.jsonl"):
                if p.is_file() and _is_data_shard_file(p.name):
                    yield p
        except Exception:
            continue


def _count_shards_direct(run_dir: Path) -> tuple[int, float]:
    count = 0
    latest = 0.0
    for p in _iter_shard_files_direct(run_dir):
        count += 1
        try:
            latest = max(latest, float(p.stat().st_mtime))
        except Exception:
            continue
    if count == 0:
        try:
            latest = float(run_dir.stat().st_mtime)
        except Exception:
            latest = 0.0
    return count, latest


def _discover_models(outputs_root: Path, preferred_model: str | None = None) -> dict[str, Any]:
    """
    Discover models as immediate subdirectories under outputs_root.

    Each model contains zero or more runs (run_* subfolders, or any folder with metadata/shards).
    """
    models_payload: list[dict[str, Any]] = []
    all_runs: list[dict[str, Any]] = []

    for model_dir in _safe_list_dirs(outputs_root):
        if model_dir.name.startswith("."):
            continue
        model_name = model_dir.name

        run_dirs: list[Path] = []
        for d in _safe_list_dirs(model_dir):
            if d.name.startswith(".") or d.name == "shards":
                continue
            shard_count, _ = _count_shards_direct(d)
            if d.name.startswith("run_") or shard_count > 0 or _has_metadata(d):
                run_dirs.append(d)

        root_shards, root_latest = _count_shards_direct(model_dir)
        root_has_meta = _has_metadata(model_dir)

        runs_payload: list[dict[str, Any]] = []
        if run_dirs:
            if root_shards > 0 or root_has_meta:
                runs_payload.append(
                    {
                        "model": model_name,
                        "name": "default",
                        "run_dir": model_dir.relative_to(outputs_root).as_posix(),
                        "last_modified": root_latest,
                        "files": root_shards,
                        "model_dir": model_name,
                    }
                )
            for rd in run_dirs:
                shards, latest = _count_shards_direct(rd)
                runs_payload.append(
                    {
                        "model": model_name,
                        "name": rd.name,
                        "run_dir": rd.relative_to(outputs_root).as_posix(),
                        "last_modified": latest,
                        "files": shards,
                        "model_dir": model_name,
                    }
                )
        else:
            # No run subfolders; treat the model directory itself as the run.
            runs_payload.append(
                {
                    "model": model_name,
                    "name": "default",
                    "run_dir": model_dir.relative_to(outputs_root).as_posix(),
                    "last_modified": root_latest,
                    "files": root_shards,
                    "model_dir": model_name,
                }
            )

        runs_payload.sort(key=lambda r: float(r.get("last_modified") or 0.0), reverse=True)
        models_payload.append({"name": model_name, "runs": runs_payload})
        for r in runs_payload:
            all_runs.append(r)

    # Default selection: latest run with shards, optionally constrained to preferred_model.
    default_model: str | None = None
    default_run_dir = ""

    def _pick_best(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not runs:
            return None
        with_files = [r for r in runs if int(r.get("files") or 0) > 0]
        pool = with_files or runs
        return max(pool, key=lambda r: float(r.get("last_modified") or 0.0))

    if preferred_model is not None:
        model_entry = next((m for m in models_payload if m["name"] == preferred_model), None)
        if model_entry:
            best = _pick_best(model_entry.get("runs") or [])
            if best:
                default_model = str(best["model"])
                default_run_dir = str(best["run_dir"])
    else:
        best = _pick_best(all_runs)
        if best:
            default_model = str(best["model"])
            default_run_dir = str(best["run_dir"])

    if default_model is None and models_payload:
        default_model = str(models_payload[0]["name"])
        default_run_dir = str((models_payload[0]["runs"][0]["run_dir"] if models_payload[0]["runs"] else ""))

    return {
        "root": str(outputs_root),
        "models": models_payload,
        "default_model": default_model,
        "default_run_dir": default_run_dir,
    }


def _find_jsonl_files(run_dir: Path) -> list[dict[str, object]]:
    files: list[dict[str, object]] = []
    for path in sorted(_iter_shard_files_direct(run_dir), key=lambda p: p.name):
        try:
            rel = path.relative_to(run_dir)
        except Exception:
            rel = path
        try:
            size = path.stat().st_size
        except Exception:
            size = 0
        files.append({"path": str(rel), "name": path.name, "size": size})
    return files


def _safe_int(value: str | None, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    try:
        parsed = int(value or "")
    except Exception:
        return default
    if minimum is not None:
        parsed = max(parsed, minimum)
    if maximum is not None:
        parsed = min(parsed, maximum)
    return parsed


def _load_records(path: Path, offset: int, limit: int) -> tuple[list[dict[str, object]], int, bool]:
    records: list[dict[str, object]] = []
    read_limit = max(limit, 0) + 1
    next_offset = offset
    has_more = False

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f):
            if line_no < offset:
                continue
            if len(records) >= read_limit:
                has_more = True
                break
            text = line.strip()
            if not text:
                continue
            entry: dict[str, object]
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    entry = parsed
                else:
                    entry = {"_raw_record": parsed}
            except Exception as exc:
                entry = {
                    "_parse_error": str(exc),
                    "_raw_line": text[:20_000],
                }
            entry.setdefault("patient_note", "")
            records.append({"line_no": line_no, "data": entry})

    if has_more:
        records = records[:limit]

    if records:
        next_offset = int(records[-1]["line_no"]) + 1  # type: ignore[index]
    else:
        next_offset = offset

    return records, next_offset, has_more


def _make_handler(outputs_root: Path, discovery: dict[str, Any]):
    outputs_root = outputs_root.resolve()
    default_run_dir = str(discovery.get("default_run_dir") or "")

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:  # pragma: no cover - UX
            pass

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, status: int, html: str) -> None:
            body = html.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _bad_request(self, detail: str) -> None:
            self._send_json(400, {"error": detail})

        def _not_found(self, detail: str) -> None:
            self._send_json(404, {"error": detail})

        def _resolve_run_dir(self, raw: str | None) -> Path:
            if raw is None:
                raw = default_run_dir
            rel = Path(unquote(raw or ""))
            if rel.is_absolute():
                raise ValueError("invalid run_dir path")
            run_dir = (outputs_root / rel).resolve()
            if not _is_within(outputs_root, run_dir):
                raise ValueError("invalid run_dir path")
            if not run_dir.is_dir():
                raise FileNotFoundError(str(rel))
            return run_dir

        def _resolve_file(self) -> Path:
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query)
            raw = qs.get("file", [""])[0]
            if not raw:
                raise ValueError("missing file")
            path = Path(unquote(raw))
            if path.is_absolute():
                raise ValueError("invalid file path")
            resolved = (outputs_root / path).resolve()
            if not _is_within(outputs_root, resolved):
                raise ValueError("invalid file path")
            if not resolved.is_file():
                raise FileNotFoundError(str(raw))
            return resolved

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            route = parsed.path
            qs = parse_qs(parsed.query)

            if route in {"/", "/index.html"}:
                self._send_html(200, VIEW_HTML)
                return

            if route == "/api/models":
                self._send_json(200, discovery)
                return

            if route == "/api/files":
                try:
                    run_dir = self._resolve_run_dir(qs.get("run_dir", [None])[0])
                except ValueError as exc:
                    self._bad_request(str(exc))
                    return
                except FileNotFoundError as exc:
                    self._not_found(str(exc))
                    return
                self._send_json(
                    200,
                    {
                        "files": _find_jsonl_files(run_dir),
                        "run_dir": (
                            "" if run_dir == outputs_root else run_dir.relative_to(outputs_root).as_posix()
                        ),
                    },
                )
                return

            if route == "/api/records":
                try:
                    path = self._resolve_file()
                except ValueError as exc:
                    self._bad_request(str(exc))
                    return
                except FileNotFoundError as exc:
                    self._not_found(str(exc))
                    return

                offset = _safe_int(qs.get("offset", [None])[0], 0, minimum=0)
                limit = _safe_int(qs.get("limit", [None])[0], 40, minimum=1, maximum=200)
                records, next_offset, has_more = _load_records(path, offset=offset, limit=limit)
                self._send_json(
                    200,
                    {
                        "file": str(path.relative_to(outputs_root)),
                        "offset": offset,
                        "next_offset": next_offset,
                        "limit": limit,
                        "has_more": has_more,
                        "records": records,
                    },
                )
                return

            self.send_error(404, "Not Found")

    return _Handler


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="View Open-Patients JSONL output files in browser.")
    ap.add_argument(
        "--run_dir",
        default="outputs",
        help="Directory containing Open-Patients output data (default: outputs)",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="Open the latest run for this model (optional)",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument(
        "--open-browser",
        action="store_true",
        default=False,
        help="Open the viewer automatically in your browser",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    print_header("Open-Patients JSONL Viewer")

    outputs_root = Path(args.run_dir)
    if not outputs_root.is_absolute():
        outputs_root = Path.cwd() / outputs_root
    if not outputs_root.exists():
        raise SystemExit(f"Output directory not found: {outputs_root}")
    if not outputs_root.is_dir():
        raise SystemExit(f"Output path is not a directory: {outputs_root}")

    # If the given directory itself looks like a run dir (has shards directly or under ./shards),
    # treat it as a single-model view rather than an outputs/ root.
    direct_shards, direct_latest = _count_shards_direct(outputs_root)
    if direct_shards > 0 or _has_metadata(outputs_root):
        model_name = outputs_root.parent.name if outputs_root.name == "shards" else outputs_root.name
        discovery = {
            "root": str(outputs_root),
            "models": [
                {
                    "name": model_name,
                    "runs": [
                        {
                            "model": model_name,
                            "name": "default",
                            "run_dir": "",
                            "last_modified": direct_latest,
                            "files": direct_shards,
                            "model_dir": "",
                        }
                    ],
                }
            ],
            "default_model": model_name,
            "default_run_dir": "",
        }
    else:
        discovery = _discover_models(outputs_root, preferred_model=args.model)

    available_models = [m.get("name") for m in discovery.get("models") or []]
    if args.model is not None and args.model not in available_models:
        raise SystemExit(
            f"Model not found: {args.model}. Available: {', '.join([m for m in available_models if m])}"
        )

    handler = _make_handler(outputs_root, discovery)
    server = ThreadingHTTPServer((args.host, args.port), handler)

    url = f"http://{args.host}:{args.port}"
    print(f"Viewer running: {url}")
    print(f"Output root: {outputs_root}")
    print(f"Default model: {discovery.get('default_model')}")
    print("Press Ctrl+C to stop.")
    if discovery.get("default_run_dir"):
        print(f"Default run_dir: {discovery.get('root')}/{discovery.get('default_run_dir')}")

    if args.open_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()


VIEW_HTML = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Open-Patients Output Viewer</title>
    <style>
      :root {
        --bg: #f5f7fb;
        --panel: #ffffff;
        --ink: #111827;
        --muted: #4b5563;
        --line: #d6dce4;
        --ok: #0f766e;
        --bad: #b91c1c;
        --accent: #1d4ed8;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        font-family: "Trebuchet MS", "Arial", sans-serif;
        background: linear-gradient(120deg, #f6f8fc 0%, #eef2ff 50%, #f8fafc 100%);
        color: var(--ink);
      }
      .container { max-width: 1200px; margin: 0 auto; padding: 24px; }
      h1 {
        margin: 0 0 8px 0;
        font-size: 1.8rem;
      }
      .help {
        margin: 0 0 16px;
        color: var(--muted);
      }
      .panel {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 14px;
      }
      .controls {
        display: flex;
        gap: 10px;
        align-items: center;
        flex-wrap: wrap;
      }
      .controls > div {
        display: flex;
        gap: 6px;
        align-items: center;
      }
      select, button {
        border: 1px solid var(--line);
        padding: 8px 10px;
        border-radius: 8px;
        background: #fff;
        color: var(--ink);
      }
      button {
        cursor: pointer;
        background: var(--accent);
        color: #fff;
        border-color: var(--accent);
      }
      button[disabled] { opacity: 0.55; cursor: not-allowed; }
      .meta { margin: 12px 0; color: var(--muted); }
      .records { display: grid; gap: 10px; margin-top: 12px; }
      .card {
        border: 1px solid var(--line);
        background: #fff;
        border-radius: 10px;
        padding: 10px;
      }
      .row {
        display: flex; justify-content: space-between; gap: 8px;
      }
      .note {
        margin: 8px 0;
        white-space: pre-wrap;
        background: #f8fafc;
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 8px;
        color: #111827;
      }
      .json {
        margin: 0;
        white-space: pre-wrap;
        background: #0f172a;
        color: #e2e8f0;
        border-radius: 8px;
        padding: 10px;
        max-height: 620px;
        min-height: 340px;
        overflow: auto;
      }
      .split {
        display: grid;
        grid-template-columns: 1.25fr 0.75fr;
        gap: 10px;
        margin-top: 10px;
      }
      @media (max-width: 900px) {
        .split { grid-template-columns: 1fr; }
      }
      .paneTitle {
        font-weight: 700;
        font-size: 0.95rem;
      }
      .noteBox {
        margin-top: 6px;
        white-space: pre-wrap;
        background: #f8fafc;
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 10px;
        color: #111827;
        max-height: 380px;
        overflow: auto;
      }
      .noteHint {
        margin-top: 6px;
        color: var(--muted);
        font-size: 0.9rem;
      }
      mark.hl {
        background: #fde68a;
        border-radius: 2px;
        padding: 0 1px;
      }
      .evList {
        margin-top: 6px;
        display: grid;
        gap: 6px;
        max-height: 380px;
        overflow: auto;
        padding-right: 4px;
      }
      .ev {
        text-align: left;
        border: 1px solid var(--line);
        padding: 8px;
        border-radius: 10px;
        background: #ffffff;
        cursor: pointer;
      }
      .ev:hover { border-color: var(--accent); }
      .ev.active {
        border-color: var(--accent);
        background: #eff6ff;
      }
      .evTop {
        display: flex;
        align-items: baseline;
        gap: 8px;
      }
      .evTag {
        font-size: 0.72rem;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        color: var(--muted);
      }
      .evLabel { font-weight: 700; }
      .evMeta {
        margin-top: 2px;
        color: var(--muted);
        font-size: 0.85rem;
      }
      .evEvidence {
        margin-top: 6px;
        color: #111827;
        background: #f8fafc;
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 6px 8px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono",
          "Courier New", monospace;
        font-size: 0.85rem;
        white-space: pre-wrap;
      }
      details.raw { margin-top: 10px; }
      details.raw > summary { cursor: pointer; color: var(--muted); }
      .bad {
        color: var(--bad);
        font-weight: 600;
      }
      .ok {
        color: var(--ok);
        font-weight: 600;
      }
      .btnrow { margin-top: 10px; }
      .subtle { font-size: 0.9rem; color: var(--muted); }
    </style>
  </head>
  <body>
    <div class="container">
      <h1>Open-Patients Output Viewer</h1>
      <p class="help">Select model and run, then browse records with patient note and parsed JSON.</p>
      <div class="panel">
        <div class="controls">
          <div>
            <label for="modelSelect">Model</label>
            <select id="modelSelect"></select>
          </div>
          <div>
            <label for="runSelect">Run</label>
            <select id="runSelect"></select>
          </div>
          <div>
            <label for="fileSelect">JSONL file</label>
            <select id="fileSelect"></select>
          </div>
          <button id="reloadBtn">Reload</button>
        </div>
        <div class="meta" id="status">Loading models...</div>
        <div class="records" id="records"></div>
        <div class="btnrow">
          <button id="loadMoreBtn" style="display:none;">Load more</button>
        </div>
        <div class="subtle" id="stateInfo"></div>
      </div>
    </div>

    <script>
      const modelSelect = document.getElementById("modelSelect");
      const runSelect = document.getElementById("runSelect");
      const fileSelect = document.getElementById("fileSelect");
      const reloadBtn = document.getElementById("reloadBtn");
      const status = document.getElementById("status");
      const records = document.getElementById("records");
      const loadMoreBtn = document.getElementById("loadMoreBtn");
      const stateInfo = document.getElementById("stateInfo");

      let nextOffset = 0;
      let currentRun = "";
      let currentFile = "";
      let models = [];

      function escapeHtml(value) {
        return String(value ?? "")
          .replace(/&/g, "&amp;")
          .replace(/</g, "&lt;")
          .replace(/>/g, "&gt;");
      }

      function patientStatus(rec) {
        const ok = rec.extraction_ok;
        return ok
          ? `<span class="ok">extraction_ok: true</span>`
          : `<span class="bad">extraction_ok: false</span>`;
      }

      function truncate(text, n = 140) {
        const s = String(text ?? "");
        if (s.length <= n) return s;
        return s.slice(0, Math.max(0, n - 3)) + "...";
      }

      function collapseWhitespace(text) {
        return String(text ?? "").replace(/\\s+/g, " ").trim();
      }

      function normalizeWithMap(text) {
        let norm = "";
        const map = [];
        let lastSpace = false;
        for (let i = 0; i < text.length; i++) {
          const ch = text[i];
          if (/\\s/.test(ch)) {
            if (norm.length === 0 || lastSpace) continue;
            norm += " ";
            map.push(i);
            lastSpace = true;
          } else {
            norm += ch;
            map.push(i);
            lastSpace = false;
          }
        }
        return { norm, map };
      }

      function findAllMatches(haystack, needle, cap = 50) {
        const out = [];
        if (!needle) return out;
        let idx = 0;
        while (true) {
          idx = haystack.indexOf(needle, idx);
          if (idx === -1) break;
          out.push([idx, idx + needle.length]);
          idx = idx + Math.max(1, needle.length);
          if (out.length >= cap) break;
        }
        return out;
      }

      function findEvidenceMatches(noteText, evidenceText) {
        const note = String(noteText ?? "");
        const ev = String(evidenceText ?? "").trim();
        if (!ev) return [];

        // Exact
        let matches = findAllMatches(note, ev);
        if (matches.length) return matches;

        // Case-insensitive
        const noteLower = note.toLowerCase();
        const evLower = ev.toLowerCase();
        matches = findAllMatches(noteLower, evLower);
        if (matches.length) return matches;

        // Whitespace-normalized + case-insensitive
        const evNorm = collapseWhitespace(ev);
        if (!evNorm) return [];
        const nm = normalizeWithMap(note);
        const hn = nm.norm.toLowerCase();
        const nd = evNorm.toLowerCase();
        const normMatches = findAllMatches(hn, nd);
        if (!normMatches.length) return [];

        const out = [];
        for (const [s, e] of normMatches) {
          const os = nm.map[s];
          const oe = nm.map[e - 1];
          if (os === undefined || oe === undefined) continue;
          out.push([os, oe + 1]);
          if (out.length >= 50) break;
        }
        return out;
      }

      function mergeRanges(ranges, maxLen) {
        const list = (ranges || [])
          .map((r) => [Number(r[0]), Number(r[1])])
          .filter((r) => Number.isFinite(r[0]) && Number.isFinite(r[1]) && r[1] > r[0])
          .map((r) => [Math.max(0, r[0]), Math.min(maxLen, r[1])])
          .filter((r) => r[1] > r[0]);
        list.sort((a, b) => a[0] - b[0]);
        const out = [];
        for (const [s, e] of list) {
          if (!out.length || s > out[out.length - 1][1]) out.push([s, e]);
          else out[out.length - 1][1] = Math.max(out[out.length - 1][1], e);
        }
        return out;
      }

      function highlightHtml(text, ranges) {
        const s = String(text ?? "");
        if (!ranges || !ranges.length) return escapeHtml(s);
        const merged = mergeRanges(ranges, s.length);
        if (!merged.length) return escapeHtml(s);
        let out = "";
        let last = 0;
        for (const [start, end] of merged) {
          out += escapeHtml(s.slice(last, start));
          out += `<mark class="hl">${escapeHtml(s.slice(start, end))}</mark>`;
          last = end;
        }
        out += escapeHtml(s.slice(last));
        return out;
      }

      function guessLabel(obj, path) {
        if (obj && typeof obj === "object") {
          if (typeof obj.name === "string" && obj.name.trim()) return obj.name;
          if (typeof obj.test === "string" && obj.test.trim()) return obj.test;
          if (typeof obj.condition === "string" && obj.condition.trim()) return obj.condition;
          if (typeof obj.assessment === "string" && obj.assessment.trim()) return obj.assessment;
          if (typeof obj.finding === "string" && obj.finding.trim()) return obj.finding;
        }
        return path.join(".");
      }

      function guessMeta(obj) {
        if (!obj || typeof obj !== "object") return "";
        const parts = [];
        if (typeof obj.status === "string" && obj.status.trim()) parts.push(obj.status);
        if (typeof obj.category === "string" && obj.category.trim()) parts.push(obj.category);
        if (typeof obj.certainty === "string" && obj.certainty.trim()) parts.push(obj.certainty);
        if (typeof obj.temporality === "string" && obj.temporality.trim()) parts.push(obj.temporality);
        return parts.join(" | ");
      }

      function extractEvidenceItems(rec) {
        const items = [];
        const seen = new Set();
        const skip = new Set([
          "patient_note",
          "model_output_raw",
          "source",
          "created_at",
          "id",
          "extraction_ok",
        ]);

        function walk(value, path) {
          if (value == null) return;
          if (Array.isArray(value)) {
            for (let i = 0; i < value.length; i++) {
              walk(value[i], path.concat([String(i)]));
            }
            return;
          }
          if (typeof value === "object") {
            const ev = value.evidence;
            if (typeof ev === "string" && ev.trim()) {
              const top = path[0] || "record";
              const label = guessLabel(value, path);
              const meta = guessMeta(value);
              const key = `${top}\\n${label}\\n${ev}`;
              if (!seen.has(key)) {
                seen.add(key);
                items.push({ top, path: path.join("."), label, meta, evidence: ev });
              }
            }
            for (const [k, v] of Object.entries(value)) {
              if (k === "evidence") continue;
              walk(v, path.concat([k]));
            }
          }
        }

        for (const [k, v] of Object.entries(rec || {})) {
          if (skip.has(k)) continue;
          walk(v, [k]);
        }

        items.sort(
          (a, b) =>
            (b.evidence.length - a.evidence.length) ||
            a.top.localeCompare(b.top) ||
            a.label.localeCompare(b.label)
        );
        return items;
      }

      function renderEvidenceButton(item, idx) {
        const meta = item.meta ? `<div class="evMeta">${escapeHtml(item.meta)}</div>` : "";
        return `
          <button class="ev" type="button" data-ev-idx="${idx}">
            <div class="evTop">
              <span class="evTag">${escapeHtml(item.top)}</span>
              <span class="evLabel">${escapeHtml(item.label)}</span>
            </div>
            ${meta}
            <div class="evEvidence">${escapeHtml(truncate(item.evidence, 180))}</div>
          </button>
        `;
      }

      function wireEvidenceInteractions(card, noteEl, noteText, noteStatusEl, items) {
        if (!noteEl) return;
        const buttons = Array.from(card.querySelectorAll("[data-ev-idx]"));
        if (!buttons.length) {
          if (noteStatusEl) noteStatusEl.textContent = "No evidence fields found in this record.";
          return;
        }
        let pinned = null;

        function setActive(idx) {
          for (const btn of buttons) {
            const i = Number(btn.getAttribute("data-ev-idx"));
            btn.classList.toggle("active", i === idx);
          }
        }

        function clear() {
          noteEl.innerHTML = escapeHtml(noteText);
          if (noteStatusEl) {
            noteStatusEl.textContent =
              "Hover an extracted entry to highlight its evidence in the note. Click to pin.";
          }
          setActive(-1);
        }

        function apply(idx) {
          const it = items[idx];
          const evidence = it ? it.evidence : "";
          const ranges = findEvidenceMatches(noteText, evidence);
          noteEl.innerHTML = highlightHtml(noteText, ranges);
          setActive(idx);
          if (noteStatusEl) {
            if (ranges.length) {
              noteStatusEl.textContent = `Evidence: "${truncate(evidence, 160)}" (${ranges.length} match${
                ranges.length === 1 ? "" : "es"
              })`;
            } else {
              noteStatusEl.textContent = `No match found for evidence: "${truncate(evidence, 160)}"`;
            }
          }
          const first = noteEl.querySelector("mark.hl");
          if (first) {
            try {
              first.scrollIntoView({ block: "center", behavior: "smooth" });
            } catch (e) {}
          }
        }

        clear();
        for (const btn of buttons) {
          const idx = Number(btn.getAttribute("data-ev-idx"));
          btn.addEventListener("mouseenter", () => {
            if (pinned == null) apply(idx);
          });
          btn.addEventListener("mouseleave", () => {
            if (pinned == null) clear();
          });
          btn.addEventListener("click", () => {
            if (pinned === idx) {
              pinned = null;
              clear();
            } else {
              pinned = idx;
              apply(idx);
            }
          });
        }
      }

      function renderRecord(item) {
        const rec = item.data || {};
        const id = rec.id || "(no id)";
        const line = item.line_no;
        const noteText = rec.patient_note || "";
        const evItems = extractEvidenceItems(rec);
        const card = document.createElement("article");
        card.className = "card";
        const evHtml = evItems.length
          ? evItems.map((it, idx) => renderEvidenceButton(it, idx)).join("")
          : `<div class="subtle">No evidence fields found in this record.</div>`;
        card.innerHTML = `
          <div class="row">
            <strong>${escapeHtml(id)} · line ${line}</strong>
            <span>${patientStatus(rec)}</span>
          </div>
          <div class="split">
            <div>
              <div class="paneTitle">Clinical note</div>
              <div class="noteBox" data-note>${escapeHtml(noteText)}</div>
              <div class="noteHint" data-note-status></div>
            </div>
            <div>
              <div class="paneTitle">Extracted entries (evidence)</div>
              <div class="evList" data-ev-list>${evHtml}</div>
              <details class="raw">
                <summary>Raw JSON</summary>
                <pre class="json">${escapeHtml(JSON.stringify(rec, null, 2))}</pre>
              </details>
            </div>
          </div>
        `;
        records.appendChild(card);

        const noteEl = card.querySelector("[data-note]");
        const noteStatusEl = card.querySelector("[data-note-status]");
        wireEvidenceInteractions(card, noteEl, noteText, noteStatusEl, evItems);
      }

      function setStateMessage(text) {
        status.textContent = text;
      }

      async function fetchJson(url) {
        const resp = await fetch(url);
        if (!resp.ok) {
          const err = await resp.json().catch(() => ({}));
          throw new Error(err.error || `HTTP ${resp.status}`);
        }
        return resp.json();
      }

      async function loadModels() {
        try {
          const payload = await fetchJson("/api/models");
          models = payload.models || [];
          const defaultRunDir = payload.default_run_dir || "";
          if (!models.length) {
            setStateMessage("No models found under output root.");
            stateInfo.textContent = "";
            return;
          }
          modelSelect.innerHTML = "";
          for (const m of models) {
            const opt = document.createElement("option");
            opt.value = m.name;
            opt.textContent = m.name;
            modelSelect.appendChild(opt);
          }
          if (payload.default_model && modelSelect.querySelector(`option[value=\"${payload.default_model}\"]`)) {
            modelSelect.value = payload.default_model;
          }
          await loadRuns(defaultRunDir);
        } catch (err) {
          setStateMessage(`Failed to load models: ${err && err.message ? err.message : err}`);
          stateInfo.textContent = "";
        }
      }

      async function loadRuns(explicitRunDir = "") {
        const modelName = modelSelect.value;
        const entry = models.find((m) => m.name === modelName);
        runSelect.innerHTML = "";
        fileSelect.innerHTML = "";
        if (!entry || !entry.runs.length) {
          setStateMessage(`No runs for model ${modelName}`);
          return;
        }
        for (const run of entry.runs) {
          const opt = document.createElement("option");
          opt.value = run.run_dir || "";
          opt.textContent = `${run.name} (${run.files} file${run.files === 1 ? "" : "s"})`;
          runSelect.appendChild(opt);
        }
        if (explicitRunDir) {
          runSelect.value = explicitRunDir;
        }
        if (!runSelect.value) {
          runSelect.value = runSelect.options[0].value;
        }
        await loadFiles();
      }

      async function loadFiles() {
        const modelName = modelSelect.value;
        const runDir = runSelect.value;
        currentRun = runDir;
        stateInfo.textContent = `Model: ${modelName} · Run: ${runDir || "default"}`;

        const payload = await fetchJson(`/api/files?run_dir=${encodeURIComponent(runDir)}`);
        fileSelect.innerHTML = "";
        if (!payload.files.length) {
          setStateMessage(`No JSONL files in run ${runDir || "default"}.`);
          records.innerHTML = "";
          loadMoreBtn.style.display = "none";
          return;
        }
        for (const f of payload.files) {
          const opt = document.createElement("option");
          opt.value = f.path;
          opt.textContent = `${f.path} (${f.size} bytes)`;
          fileSelect.appendChild(opt);
        }
        currentFile = fileSelect.value;
        setStateMessage(`Loaded ${payload.files.length} file(s).`);
        loadRecords(true);
      }

      async function loadRecords(reset = false) {
        if (!currentFile) return;
        if (reset) {
          nextOffset = 0;
          records.innerHTML = "";
        }

        const fullPath = currentRun ? `${currentRun}/${currentFile}` : currentFile;
        const payload = await fetchJson(
          `/api/records?file=${encodeURIComponent(fullPath)}&offset=${nextOffset}&limit=40`
        );
        nextOffset = payload.next_offset || nextOffset;
        for (const item of payload.records) {
          renderRecord(item);
        }
        loadMoreBtn.style.display = payload.has_more ? "inline-block" : "none";
        setStateMessage(`File: ${payload.file} · showing ${records.children.length} record card(s)`);
      }

      modelSelect.addEventListener("change", async () => {
        await loadRuns();
      });

      runSelect.addEventListener("change", async () => {
        await loadFiles();
      });

      fileSelect.addEventListener("change", () => {
        currentFile = fileSelect.value;
        loadRecords(true);
      });

      reloadBtn.addEventListener("click", async () => {
        await loadFiles();
      });

      loadMoreBtn.addEventListener("click", () => {
        loadMoreBtn.disabled = true;
        loadRecords(false).finally(() => {
          loadMoreBtn.disabled = false;
        });
      });

      loadModels();
    </script>
  </body>
</html>
"""


if __name__ == "__main__":
    main()
