#!/usr/bin/env python3
"""
Managed launcher for one or more `vllm serve` processes from run config endpoints.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ..core.config import load_run_config
from ..core.llm_api import EndpointConfig, parse_api_endpoints
from ..utils.utils import colored, print_header


def _split_csv(val: str) -> List[str]:
    return [x.strip() for x in val.split(",") if x.strip()]


def _extract_host_port(base_url: str) -> tuple[str, int | None]:
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme and parsed.hostname:
        return parsed.hostname, parsed.port
    return "127.0.0.1", None


def _health_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/models"


def _wait_for_health(
    url: str, timeout_s: float, poll_s: float, headers: Optional[Dict[str, str]] = None
) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            req = urllib.request.Request(url, headers=headers or {}, method="GET")
            with urllib.request.urlopen(req, timeout=min(poll_s, 5.0)) as resp:
                if 200 <= getattr(resp, "status", 0) < 300:
                    return True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            pass
        time.sleep(max(0.1, poll_s))
    return False


def _build_vllm_cmd(
    endpoint: EndpointConfig,
) -> tuple[List[str], Dict[str, str], Dict[str, str]]:
    serve = dict(endpoint.serve or {})
    if not serve:
        raise ValueError(f"Endpoint '{endpoint.name}' is missing serve config.")

    model = str(serve.pop("model", endpoint.model))
    host_default, port_default = _extract_host_port(endpoint.base_url)
    host = str(serve.pop("host", host_default))
    port_raw = serve.pop("port", port_default)
    if port_raw is None:
        raise ValueError(
            f"Endpoint '{endpoint.name}' must set serve.port "
            "or have a port in endpoint.base_url."
        )
    port = int(port_raw)

    env = os.environ.copy()
    cuda_visible_devices = serve.pop("cuda_visible_devices", None)
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

    extra_args = serve.pop("args", [])
    if not isinstance(extra_args, list):
        raise ValueError(f"Endpoint '{endpoint.name}' serve.args must be a list.")
    extra_args = [str(x) for x in extra_args]

    serve_api_key = serve.pop("api_key", "dummy")

    cmd: List[str] = ["vllm", "serve", model, "--host", host, "--port", str(port)]

    # Any remaining keys in serve are translated to CLI args.
    for key, value in serve.items():
        flag = "--" + str(key).replace("_", "-")
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
            continue
        cmd.extend([flag, str(value)])

    if serve_api_key is not None and "--api-key" not in extra_args:
        cmd.extend(["--api-key", str(serve_api_key)])

    cmd.extend(extra_args)
    health_headers: Dict[str, str] = {}
    if serve_api_key is not None:
        health_headers["Authorization"] = f"Bearer {serve_api_key}"
    return cmd, env, health_headers


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Launch vLLM API servers from run config.")
    ap.add_argument("--config", required=True, help="Run profile YAML (configs/runs/*.yaml)")
    ap.add_argument(
        "--endpoints",
        default=None,
        help="Optional CSV of endpoint names to launch (default: all)",
    )
    ap.add_argument("--logs_dir", default=None, help="Directory for per-endpoint logs")
    ap.add_argument("--dry_run", action="store_true", help="Print commands without launching")
    ap.add_argument(
        "--no_wait_for_health",
        dest="wait_for_health",
        action="store_false",
        help="Skip health checks on /v1/models",
    )
    ap.add_argument(
        "--health_timeout_s",
        type=float,
        default=180.0,
        help="Seconds to wait for endpoint health",
    )
    ap.add_argument(
        "--health_poll_s",
        type=float,
        default=2.0,
        help="Polling interval for endpoint health checks",
    )
    ap.set_defaults(wait_for_health=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    print_header("Open-Patients vLLM Serve")

    cfg = load_run_config(args.config)
    model_cfg = cfg.get("model") or {}
    api_cfg = cfg.get("api") or {}
    endpoints_raw = api_cfg.get("endpoints")
    if not endpoints_raw:
        raise SystemExit("Config is missing api.endpoints.")

    try:
        endpoints = parse_api_endpoints(
            endpoints_raw,
            default_model=model_cfg.get("name"),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    selected_names = set(_split_csv(args.endpoints)) if args.endpoints else None
    if selected_names is not None:
        endpoints = [ep for ep in endpoints if ep.name in selected_names]
        if not endpoints:
            raise SystemExit("No endpoints matched --endpoints filter.")

    log_dir = Path(args.logs_dir) if args.logs_dir else Path("outputs") / "serve_logs" / (
        f"vllm_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    procs: List[subprocess.Popen] = []
    log_handles = []
    endpoint_logs: Dict[str, Path] = {}
    endpoint_health_headers: Dict[str, Dict[str, str]] = {}

    for endpoint in endpoints:
        cmd, env, health_headers = _build_vllm_cmd(endpoint)
        log_path = log_dir / f"{endpoint.name}.log"
        endpoint_logs[endpoint.name] = log_path
        endpoint_health_headers[endpoint.name] = health_headers
        print(colored(f"[launch] {endpoint.name}: {' '.join(cmd)}", "CYAN"))
        if "CUDA_VISIBLE_DEVICES" in env:
            print(
                colored(
                    f"         CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} log={log_path}",
                    "WHITE",
                )
            )
        else:
            print(colored(f"         log={log_path}", "WHITE"))
        if args.dry_run:
            continue
        h = log_path.open("w", encoding="utf-8")
        log_handles.append(h)
        procs.append(subprocess.Popen(cmd, env=env, stdout=h, stderr=subprocess.STDOUT))

    if args.dry_run:
        return

    if args.wait_for_health:
        for endpoint in endpoints:
            url = _health_url(endpoint.base_url)
            print(colored(f"[health] waiting for {endpoint.name}: {url}", "CYAN"))
            ok = _wait_for_health(
                url,
                args.health_timeout_s,
                args.health_poll_s,
                headers=endpoint_health_headers.get(endpoint.name),
            )
            if not ok:
                for p in procs:
                    if p.poll() is None:
                        p.terminate()
                raise SystemExit(f"Health check timeout for endpoint '{endpoint.name}': {url}")
            print(colored(f"[health] ready: {endpoint.name}", "GREEN"))

    stopping = False

    def _shutdown(signum: int, _frame) -> None:
        nonlocal stopping
        if stopping:
            return
        stopping = True
        print(colored(f"[signal] received {signum}; stopping servers...", "YELLOW"))
        for p in procs:
            if p.poll() is None:
                p.terminate()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    exit_code = 0
    try:
        while True:
            all_done = True
            for endpoint, proc in zip(endpoints, procs):
                rc = proc.poll()
                if rc is None:
                    all_done = False
                    continue
                if rc != 0:
                    exit_code = rc
                    print(
                        colored(
                            f"[error] {endpoint.name} exited with rc={rc} log={endpoint_logs[endpoint.name]}",
                            "RED",
                        )
                    )
            if all_done:
                break
            if stopping:
                break
            time.sleep(1.0)
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                p.kill()
        for h in log_handles:
            h.close()

    if exit_code != 0:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
