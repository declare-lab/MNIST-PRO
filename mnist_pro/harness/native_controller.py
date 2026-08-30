#!/usr/bin/env python3
"""Clean-room MNIST-PRO runner over a disposable Antigravity LS.

The runner supports the original app-account transport and a Gemini Developer
API BYOK transport.  Both routes keep Antigravity's planner/MCP agent loop:
each episode starts a fresh language-server process, sends PNG bytes through
``Media.inlineData``, and lets the LS invoke the local ``ag`` MCP bridge.
Labels, manifests, credentials, and scoring remain controller-only.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import http.client
import http.server
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import re
import secrets
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import traceback
import uuid
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from typing import Any

from PIL import Image
from torchvision.datasets import MNIST


BASE = Path("/Users/poriasoujanya/Documents/Codex/2026-08-25/files-pasted-by-the-user-act")
ACTIVEGLIMPSE = Path("/Users/poriasoujanya/Documents/activeglimpse-main")
ENVIRONMENT_PY = ACTIVEGLIMPSE / "src/environment.py"
LEVEL_SUMMARIES = {
    1: ACTIVEGLIMPSE / "main_table_logs/gemini-3.7-flash_img224_box64_step32_maxsteps36_seed42_DefaultVisionAgent_hist-1_evalsets10_20260824_105544/results_summary.json",
    2: ACTIVEGLIMPSE / "main_table_logs/multidigit_2_gemini-3.7-flash_img224_box64_step32_maxsteps78_seed42_MultiDigitDefaultVisionAgent_hist-1_evalsets10_20260824_111931/results_summary.json",
}
PINNED = {
    "environment": "d2973d5bb60ce38501f8b4066cf916510e7c26a3d4d0fa0659d2b817326736c4",
    "summary_1": "48362448e87a819e162bc42ac9104493ad8b3405bf65c1cec52418e88b21c208",
    "summary_2": "d1778e114011974e2416bd45ef2dba15047c8a065596e173b6e3b6daf620c465",
}
MCP_SOURCE = BASE / "work/mnist_pro_three_arm/ag_mcp_server.py"
LANGUAGE_SERVER = Path("/Applications/Antigravity.app/Contents/Resources/bin/language_server")
MODEL_ENUM = "MODEL_PLACEHOLDER_M299"  # Gemini 3.7 Flash (Medium) in this app build.
MODEL_LABEL = "Gemini 3.7 Flash (Medium)"
API_MODEL = "gemini-3.7-flash"
THINKING_LEVEL = "medium"
TRANSPORT_APP_ACCOUNT = "antigravity-app-account-ccpa"
TRANSPORT_GEMINI_API = "antigravity-gemini-api-byok"
UPSTREAM_GEMINI_HOST = "generativelanguage.googleapis.com"
NONSECRET_PROXY_KEY = "ANTIGRAVITY_LOCAL_PROXY_NONSECRET_KEY"
IMAGE_SIZE = 224
BOX_SIZE = 64
STEP_SIZE = 32
BUDGETS = {1: 36, 2: 78}
EXPECTED_EPISODES = 100
SHUFFLE_SEED = 42
# The language server starts before its stdio MCP child has completed the
# initialize/tools/list handshake.  A single immediate state query therefore
# classified healthy episodes as infrastructure failures.  Keep this bounded
# and controller-side; it does not change the agent's episode timeout.
MCP_READY_TIMEOUT_SECONDS = 90.0
PORT_RE = re.compile(r"random port at (\d+) for (HTTPS|HTTP)")
# The app-account keychain is available to normal standalone LS mode.  With
# the newer account, the ``--headless`` bootstrap path incorrectly reports no
# auth even though the same credential is accepted immediately by normal mode.
# Keep the launch mode otherwise unchanged and accept either terminal auth
# message as the controller-side readiness signal.
AUTH_RE = re.compile(r"Headless auth: authenticated as|Auth succeeded, refreshing features and managers")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True, ensure_ascii=True).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def png_bytes(image: Image.Image) -> bytes:
    stream = io.BytesIO()
    image.convert("RGB").save(stream, format="PNG")
    return stream.getvalue()


def load_environment_module() -> Any:
    if sha256_file(ENVIRONMENT_PY) != PINNED["environment"]:
        raise RuntimeError("pinned ActiveGlimpse environment changed")
    spec = importlib.util.spec_from_file_location("mnistpro_native_environment", ENVIRONMENT_PY)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not import ActiveGlimpse environment")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class EpisodeSpec:
    level: int
    episode_id: int
    indices: tuple[int, ...]


def load_specs(level: int) -> tuple[list[EpisodeSpec], str]:
    path = LEVEL_SUMMARIES[level]
    expected = PINNED[f"summary_{level}"]
    if sha256_file(path) != expected:
        raise RuntimeError(f"pinned Level {level} schedule manifest changed")
    raw = json.loads(path.read_text(encoding="utf-8"))
    specs: list[EpisodeSpec] = []
    for row in raw["episodes"]:
        indices = (int(row["eval_index"]),) if level == 1 else tuple(int(x) for x in row["indices"])
        specs.append(EpisodeSpec(level, int(row["episode_id"]), indices))
    if sorted(item.episode_id for item in specs) != list(range(EXPECTED_EPISODES)):
        raise RuntimeError("episode ids are not exactly 0 through 99")
    random.Random(SHUFFLE_SEED + level).shuffle(specs)
    return specs, expected


def preload(level: int, specs: list[EpisodeSpec]) -> dict[int, tuple[Image.Image, int]]:
    data_root = BASE / "work/mnist_pro_cleanroom/controller_private/data"
    dataset = MNIST(root=str(data_root), train=False, download=False)
    result: dict[int, tuple[Image.Image, int]] = {}
    for index in sorted({idx for spec in specs for idx in spec.indices}):
        image, label = dataset[index]
        result[index] = (image.copy(), int(label))
    return result


def make_environment(module: Any, spec: EpisodeSpec, cache: dict[int, tuple[Image.Image, int]]) -> tuple[Any, str, Image.Image]:
    images = [cache[index][0].copy() for index in spec.indices]
    labels = [cache[index][1] for index in spec.indices]
    if spec.level == 1:
        label = str(labels[0])
        env = module.MnistActiveVisionEnv(images[0], labels[0], box_size=BOX_SIZE, image_size=IMAGE_SIZE, step_size=STEP_SIZE)
    else:
        label = "".join(str(value) for value in labels)
        env = module.MultiDigitActiveVisionEnv(images, label, box_size=BOX_SIZE, image_size=IMAGE_SIZE, step_size=STEP_SIZE)
    return env, label, env.reset()


def extract_notes_block(path: Path, arm: str) -> str:
    source = path.read_text(encoding="utf-8")
    blocks = re.findall(r"```(?:markdown)?\s*\n(.*?)\n```", source, flags=re.DOTALL | re.IGNORECASE)
    if not blocks:
        raise RuntimeError(f"{arm} notes document has no fenced instruction block")
    block = blocks[0]
    # The user explicitly removed the 2000-character cap for this run.  Keep
    # the supplied instruction otherwise byte-for-byte, and make the amendment
    # explicit so it is auditable in the prompt hash.
    block = block.replace(
        "You may rewrite NOTES.md in full at any point. It is capped at 2000 characters.\nWhen it is full, decide what to keep and what to drop.\n",
        "You may rewrite NOTES.md in full at any point. There is no size cap.\n",
    )
    return block.strip()


def prompt_for(level: int, arm: str, notes_a1: str, notes_a2: str) -> str:
    if level == 1:
        level_text = "For Level 1, the hidden canvas contains one MNIST digit from 0 through 9. Submit one digit."
    else:
        level_text = "For Level 2, the hidden canvas contains two horizontally concatenated MNIST digits. Submit the two digits in left-to-right order, preserving a leading zero when applicable."
    baseline = (
        "You are solving one partially observed MNIST episode.\n\n"
        f"{level_text}\n\n"
        "Most of the canvas is masked. The attached image is your initial observation. "
        "You can reveal another observation with:\n\n"
        "- `ag move up`\n"
        "- `ag move down`\n"
        "- `ag move left`\n"
        "- `ag move right`\n\n"
        "Inspect the returned image after each move. Submit exactly once with:\n\n"
        "`ag submit VALUE`\n\n"
        f"You have at most {BUDGETS[level]} visual observations, including the attached initial observation. "
        "Use the local tools and workspace autonomously. Finish by submitting your best answer."
    )
    if arm == "A1":
        return baseline + "\n\n" + notes_a1
    if arm == "A2":
        return baseline + "\n\n" + notes_a2
    return baseline


def auth_ok(payload: dict[str, Any], token: str) -> bool:
    supplied = payload.get("auth")
    unsigned = dict(payload)
    unsigned.pop("auth", None)
    if not isinstance(supplied, str):
        return False
    expected = hmac.new(token.encode("utf-8"), canonical(unsigned), hashlib.sha256).hexdigest()
    return hmac.compare_digest(supplied, expected)


def sanitize(value: Any, key: str | None = None) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for child_key, child in value.items():
            if child_key in {"thinking", "thinkingSignature", "userDenyInstruction"}:
                continue
            if child_key in {"inlineData", "data"} and isinstance(child, str):
                try:
                    raw = base64.b64decode(child, validate=True)
                    result[child_key] = {"byte_count": len(raw), "sha256": sha256_bytes(raw)}
                except Exception:
                    result[child_key] = {"text_length": len(child)}
            elif child_key == "env" and isinstance(child, dict):
                sensitive = {"AG_MCP_TOKEN", "GEMINI_API_KEY", "GOOGLE_API_KEY"}
                result[child_key] = {k: ("<redacted>" if k in sensitive else sanitize(v, k)) for k, v in child.items()}
            else:
                result[child_key] = sanitize(child, child_key)
        return result
    if isinstance(value, list):
        return [sanitize(child, key) for child in value]
    return value


def read_dotenv_secret(path: Path, name: str = "GEMINI_API_KEY") -> str:
    """Read one dotenv value without sourcing the file or mutating os.environ."""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() != name:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if len(value) < 20:
            raise RuntimeError(f"{name} in dotenv file is empty or malformed")
        return value
    raise RuntimeError(f"{name} is absent from the selected dotenv file")


def _walk_api_payload(value: Any, audit: dict[str, Any]) -> None:
    """Collect model-call structure without retaining prompt text or image bytes."""
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = re.sub(r"[^a-z]", "", key.lower())
            if normalized in {"thinkinglevel", "thinkinglevel"} and isinstance(child, str):
                audit.setdefault("thinking_levels", []).append(child)
            if normalized in {"inlinedata", "data"} and isinstance(child, str):
                try:
                    raw = base64.b64decode(child, validate=True)
                except Exception:
                    pass
                else:
                    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
                        audit.setdefault("images", []).append({"byte_count": len(raw), "sha256": sha256_bytes(raw)})
                        continue
            if normalized == "text" and isinstance(child, str):
                audit.setdefault("text_parts", []).append({"length": len(child), "sha256": sha256_bytes(child.encode("utf-8"))})
            if normalized in {"functioncall", "functionresponse", "functiondeclaration"}:
                audit[normalized + "_count"] = int(audit.get(normalized + "_count", 0)) + 1
            _walk_api_payload(child, audit)
    elif isinstance(value, list):
        for child in value:
            _walk_api_payload(child, audit)


class GeminiAuditProxy:
    """Local key-injecting reverse proxy with sanitized per-call receipts.

    The real API key exists only in this controller process.  Antigravity and
    every tool child receive a non-secret placeholder key, while the proxy
    replaces any key header/query immediately before forwarding upstream.
    """

    def __init__(self, api_key: str, audit_path: Path):
        self._api_key = api_key
        self.audit_path = audit_path
        self._server: http.server.ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._sequence = 0
        self.calls: list[dict[str, Any]] = []

    @property
    def url(self) -> str:
        if self._server is None:
            raise RuntimeError("Gemini audit proxy is not running")
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> None:
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:
                return

            def do_GET(self) -> None:  # noqa: N802
                owner._forward(self)

            def do_POST(self) -> None:  # noqa: N802
                owner._forward(self)

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, name="gemini-audit-proxy", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=10)
        self._server = None
        self._thread = None
        self._api_key = ""

    def summary(self) -> dict[str, Any]:
        model_calls = [call for call in self.calls if call.get("is_generate_content")]
        planner_calls = [call for call in model_calls if API_MODEL in call.get("models", [])]
        auxiliary_calls = [call for call in model_calls if API_MODEL not in call.get("models", [])]
        return {
            "http_requests": len(self.calls),
            "model_calls": len(model_calls),
            "models": sorted({model for call in model_calls for model in call.get("models", [])}),
            "planner_model_calls": len(planner_calls),
            "planner_models": sorted({model for call in planner_calls for model in call.get("models", [])}),
            "auxiliary_model_calls": len(auxiliary_calls),
            "auxiliary_models": sorted({model for call in auxiliary_calls for model in call.get("models", [])}),
            "thinking_levels": sorted({level for call in model_calls for level in call.get("thinking_levels", [])}),
            "model_call_image_counts": [len(call.get("images", [])) for call in model_calls],
            "model_call_image_sha256": [[image["sha256"] for image in call.get("images", [])] for call in model_calls],
            "planner_model_call_image_sha256": [[image["sha256"] for image in call.get("images", [])] for call in planner_calls],
            "call_routes": [
                {
                    "models": call.get("models", []),
                    "image_count": len(call.get("images", [])),
                    "function_call_count": int(call.get("functioncall_count", 0) or 0),
                    "function_response_count": int(call.get("functionresponse_count", 0) or 0),
                    "status": call.get("response_status"),
                }
                for call in model_calls
            ],
            "statuses": [call.get("response_status") for call in model_calls],
        }

    def _record(self, row: dict[str, Any]) -> None:
        with self._lock:
            self._sequence += 1
            row["sequence"] = self._sequence
            self.calls.append(row)
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())

    def _forward(self, handler: http.server.BaseHTTPRequestHandler) -> None:
        length = int(handler.headers.get("Content-Length", "0") or "0")
        body = handler.rfile.read(length) if length else b""
        split = urlsplit(handler.path)
        safe_query = [(key, value) for key, value in parse_qsl(split.query, keep_blank_values=True) if key.lower() != "key"]
        upstream_path = urlunsplit(("", "", split.path, urlencode(safe_query), ""))
        headers = {
            key: value
            for key, value in handler.headers.items()
            if key.lower() not in {"host", "connection", "proxy-connection", "content-length", "transfer-encoding", "authorization", "x-goog-api-key"}
        }
        headers["Host"] = UPSTREAM_GEMINI_HOST
        headers["x-goog-api-key"] = self._api_key
        headers["Accept-Encoding"] = "identity"
        request_audit: dict[str, Any] = {}
        try:
            decoded = json.loads(body.decode("utf-8")) if body else {}
        except Exception:
            decoded = None
        if decoded is not None:
            _walk_api_payload(decoded, request_audit)
        models = re.findall(r"/models/([^/:?]+)", split.path)
        row: dict[str, Any] = {
            "timestamp": utc_now(),
            "method": handler.command,
            "path": upstream_path,
            "request_byte_count": len(body),
            "request_sha256": sha256_bytes(body),
            "content_type": handler.headers.get("Content-Type"),
            "models": models,
            "is_generate_content": bool(re.search(r":(?:stream)?generateContent$", split.path, flags=re.IGNORECASE)),
            **request_audit,
        }
        try:
            connection = http.client.HTTPSConnection(UPSTREAM_GEMINI_HOST, timeout=360)
            connection.request(handler.command, upstream_path, body=body if body else None, headers=headers)
            response = connection.getresponse()
            response_body = response.read()
            row.update({
                "response_status": response.status,
                "response_byte_count": len(response_body),
                "response_sha256": sha256_bytes(response_body),
                "response_content_type": response.getheader("Content-Type"),
            })
            handler.send_response(response.status)
            excluded = {"connection", "transfer-encoding", "content-length", "content-encoding"}
            for key, value in response.getheaders():
                if key.lower() not in excluded:
                    handler.send_header(key, value)
            handler.send_header("Content-Length", str(len(response_body)))
            handler.send_header("Connection", "close")
            handler.end_headers()
            handler.wfile.write(response_body)
            connection.close()
        except Exception as error:
            row.update({"proxy_error": f"{type(error).__name__}: {error}"})
            payload = json.dumps({"error": {"message": "local Gemini proxy upstream failure"}}).encode("utf-8")
            handler.send_response(502)
            handler.send_header("Content-Type", "application/json")
            handler.send_header("Content-Length", str(len(payload)))
            handler.send_header("Connection", "close")
            handler.end_headers()
            handler.wfile.write(payload)
        finally:
            self._record(row)


class LocalLanguageServer:
    def __init__(
        self,
        root: Path,
        project_id: str,
        csrf: str,
        workspace: Path,
        mcp_env: dict[str, str],
        stderr_path: Path,
        transport: str = TRANSPORT_APP_ACCOUNT,
        proxy_url: str | None = None,
    ):
        self.root, self.project_id, self.csrf, self.workspace = root, project_id, csrf, workspace
        self.mcp_env = mcp_env
        self.stderr_path = stderr_path
        self.transport = transport
        self.proxy_url = proxy_url
        self.process: subprocess.Popen[str] | None = None
        self.state: dict[str, Any] = {"https": None, "http": None, "authenticated": False}
        self._log_handle: Any = None

    def start(self) -> None:
        command = [
            str(LANGUAGE_SERVER), "--standalone", "--persistent_mode",
            f"--gemini_dir={self.root / 'profile'}", "--https_server_port=0", "--http_server_port=0",
            f"--csrf_token={self.csrf}", f"--workspace_id={self.project_id}",
            "--override_ide_name=antigravity", "--override_ide_version=2.11.0",
            "--override_user_agent_name=antigravity", "--subclient_type=hub",
            "--app_data_dir=antigravity",
            "--disable_telemetry=true", "--use_ls_chrome_devtools_mcp=false",
        ]
        launch_env = os.environ.copy()
        if self.transport == TRANSPORT_GEMINI_API:
            if not self.proxy_url:
                raise RuntimeError("Gemini API transport requires a local proxy URL")
            command.extend([
                "--model_api_client_type=gemini",
                f"--api_server_url={self.proxy_url}",
            ])
            # Only a non-secret placeholder reaches Antigravity or its tool
            # descendants. The controller-local proxy injects the real key.
            launch_env["GEMINI_API_KEY"] = NONSECRET_PROXY_KEY
            launch_env.pop("GOOGLE_API_KEY", None)
            launch_env["GOOGLE_GEMINI_BASE_URL"] = self.proxy_url
        else:
            command.extend([
                "--model_api_client_type=ccpa",
                "--api_server_url=https://generativelanguage.googleapis.com",
                "--cloud_code_endpoint=https://daily-cloudcode-pa.googleapis.com",
            ])
        self._log_handle = self.stderr_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,
            env=launch_env,
        )

        def drain() -> None:
            assert self.process is not None and self.process.stderr is not None
            for line in self.process.stderr:
                self._log_handle.write(line)
                self._log_handle.flush()
                match = PORT_RE.search(line.rstrip())
                if match:
                    self.state[match.group(2).lower()] = int(match.group(1))
                if AUTH_RE.search(line):
                    self.state["authenticated"] = True

        threading.Thread(target=drain, daemon=True).start()
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and self.process.poll() is None:
            ready = self.state["https"] and (
                self.state["authenticated"] or self.transport == TRANSPORT_GEMINI_API
            )
            if ready:
                break
            time.sleep(0.05)
        if not self.state["https"]:
            raise RuntimeError("Antigravity language server did not expose HTTPS")
        if self.transport == TRANSPORT_APP_ACCOUNT and not self.state["authenticated"]:
            raise RuntimeError("Antigravity language server did not authenticate")

    @property
    def base(self) -> str:
        if not self.state["https"]:
            raise RuntimeError("language server HTTPS port is unavailable")
        return f"https://127.0.0.1:{self.state['https']}"

    def rpc(self, path: str, body: object) -> dict[str, Any]:
        payload = json.dumps(body, separators=(",", ":")).encode("utf-8")
        result = subprocess.run([
            "/usr/bin/curl", "-sS", "-k", "--max-time", "180", "--http2", "-X", "POST",
            "-H", "Content-Type: application/json", "-H", f"x-codeium-csrf-token: {self.csrf}",
            "--data-binary", "@-", f"{self.base}{path}",
        ], input=payload, capture_output=True, check=False)
        if result.returncode:
            raise RuntimeError(result.stderr.decode("utf-8", errors="replace")[:500])
        if not result.stdout:
            return {}
        try:
            decoded = json.loads(result.stdout.decode("utf-8"))
        except json.JSONDecodeError as error:
            raise RuntimeError(f"non-JSON language-server response: {result.stdout[:500]!r}") from error
        if not isinstance(decoded, dict):
            raise RuntimeError("language-server response was not an object")
        return decoded

    def stop(self) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + 20
            while process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.05)
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait(timeout=20)
        if self._log_handle is not None:
            self._log_handle.close()


class EpisodeBridge:
    """Controller-side evaluator behind the stdio MCP server."""

    def __init__(self, env: Any, label: str, level: int, arm: str, workspace: Path):
        self.env, self.label, self.level, self.arm = env, label, level, arm
        self.workspace = workspace
        self.observation_dir = workspace / "observations"
        self.observation_dir.mkdir(parents=True, exist_ok=True)
        self.mailbox = workspace / ".ag-ipc"
        (self.mailbox / "requests").mkdir(parents=True, exist_ok=True)
        (self.mailbox / "responses").mkdir(parents=True, exist_ok=True)
        self.token = secrets.token_hex(32)
        self.records: list[dict[str, Any]] = []
        self.path_bytes: dict[str, bytes] = {}
        self.path_to_record: dict[str, dict[str, Any]] = {}
        self.positions = {(int(env.x), int(env.y))}
        self.position_history = [(int(env.x), int(env.y))]
        self.move_requests = 0
        self.accepted_moves = 0
        self.effective_moves = 0
        self.wall_bumps = 0
        self.effective_revisits = 0
        self.viewer_calls = 0
        self.submit_calls = 0
        self.invalid_requests = 0
        self.delivery_receipts = 0
        self.delivery_mismatches = 0
        self.answer: str | None = None
        self.correct: bool | None = None
        self.locked = False
        self._request_map: dict[str, dict[str, Any]] = {}
        self._handled: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._watch, daemon=True)

    def _write_observation(self, image: Image.Image, source: str) -> dict[str, Any]:
        data = png_bytes(image)
        name = f"{secrets.token_hex(16)}.png"
        path = self.observation_dir / name
        path.write_bytes(data)
        record = {
            "observation_index": len(self.records),
            "image_id": secrets.token_hex(16),
            "path": str(path),
            "png_sha256": sha256_bytes(data),
            "byte_count": len(data),
            "source": source,
        }
        self.records.append(record)
        self.path_bytes[str(path)] = data
        self.path_to_record[str(path)] = record
        return record

    def seed_initial(self, image: Image.Image) -> tuple[dict[str, Any], bytes]:
        record = self._write_observation(image, "initial")
        return record, self.path_bytes[record["path"]]

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def _response(self, request_id: str, ok: bool = True, **extra: Any) -> None:
        value = {"request_id": request_id, "ok": ok, **extra}
        path = self.mailbox / "responses" / f"{request_id}.json"
        temporary = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
        temporary.write_text(json.dumps(value, separators=(",", ":")), encoding="utf-8")
        os.replace(temporary, path)

    def _watch(self) -> None:
        while not self._stop.is_set():
            for request_path in sorted((self.mailbox / "requests").glob("*.json")):
                try:
                    request = json.loads(request_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if not isinstance(request, dict):
                    continue
                request_id = request.get("request_id")
                operation = request.get("operation")
                if not isinstance(request_id, str) or not isinstance(operation, str):
                    continue
                if not auth_ok(request, self.token):
                    self._response(request_id, False, message="request authentication failed")
                    request_path.unlink(missing_ok=True)
                    continue
                try:
                    self._dispatch(request_id, operation, request)
                except Exception as error:
                    self._response(request_id, False, message=str(error))
                request_path.unlink(missing_ok=True)
            time.sleep(0.02)

    def _dispatch(self, request_id: str, operation: str, request: dict[str, Any]) -> None:
        self._handled.append({"operation": operation, "request_id": request_id, "arguments": {k: v for k, v in request.items() if k not in {"auth", "request_id", "protocol", "operation"}}})
        if operation == "move":
            self.move_requests += 1
            direction = request.get("direction")
            if self.locked or direction not in {"up", "down", "left", "right"}:
                self.invalid_requests += 1
                self._response(request_id, False, message="invalid or locked move")
                return
            if len(self.records) >= BUDGETS[self.level]:
                self.invalid_requests += 1
                self._response(request_id, False, message="observation budget exhausted")
                return
            before = (int(self.env.x), int(self.env.y))
            observation, _, _, _ = self.env.step(json.dumps({"action": "move", "direction": direction}))
            after = (int(self.env.x), int(self.env.y))
            self.accepted_moves += 1
            if before == after:
                self.wall_bumps += 1
            else:
                self.effective_moves += 1
                if after in self.positions:
                    self.effective_revisits += 1
            self.positions.add(after)
            self.position_history.append(after)
            record = self._write_observation(observation, "move")
            self._request_map[request_id] = record
            self._response(request_id, True, observation=record["path"])
            return
        if operation == "delivery_receipt":
            self.delivery_receipts += 1
            move_id = request.get("move_request_id")
            expected = self._request_map.get(move_id) if isinstance(move_id, str) else None
            if expected is None or request.get("sha256") != expected["png_sha256"] or request.get("byte_count") != expected["byte_count"]:
                self.delivery_mismatches += 1
                self._response(request_id, False, message="delivery receipt mismatch")
            else:
                self._response(request_id, True, message="delivery recorded")
            return
        if operation == "submit":
            self.submit_calls += 1
            raw = request.get("value")
            value = str(raw) if isinstance(raw, (str, int)) else ""
            valid = bool(re.fullmatch(r"[0-9]", value) if self.level == 1 else re.fullmatch(r"[0-9]{2}", value))
            if self.locked or not valid:
                self.invalid_requests += 1
                self._response(request_id, False, message="invalid answer format")
                return
            self.locked = True
            self.answer = value
            self.correct = value == self.label
            message = ("CORRECT" if self.correct else "INCORRECT") if self.arm == "A2" else "submission accepted"
            self._response(request_id, True, message=message)
            return
        self._response(request_id, False, message=f"unknown operation: {operation}")

    def counters(self) -> dict[str, Any]:
        return {
            "glimpses_exposed": len(self.records),
            "move_requests": self.move_requests,
            "accepted_moves": self.accepted_moves,
            "effective_moves": self.effective_moves,
            "wall_bumps": self.wall_bumps,
            "effective_revisits": self.effective_revisits,
            "revisit_rate": self.effective_revisits / self.effective_moves if self.effective_moves else None,
            "wall_bump_rate": self.wall_bumps / self.accepted_moves if self.accepted_moves else None,
            "position_history": self.position_history,
            "viewer_calls": self.viewer_calls,
            "submit_calls": self.submit_calls,
            "submitted_answer": self.answer,
            "correct": self.correct,
            "invalid_requests": self.invalid_requests,
            "delivery_receipts": self.delivery_receipts,
            "delivery_mismatches": self.delivery_mismatches,
            "records": self.records,
            "handled_operations": self._handled,
        }


def write_project_config(profile: Path, project_id: str, workspace: Path, bridge: EpisodeBridge) -> None:
    write_json(
        profile / "config" / "projects" / f"{project_id}.json",
        {
            "id": project_id,
            "name": secrets.token_hex(8),
            "projectResources": {"resources": [{"folderUri": workspace.as_uri()}]},
            "settings": {
                "fileAccessPolicy": "AGENT_SETTING_POLICY_ALLOW",
                "internetPolicy": "AGENT_SETTING_POLICY_DENY",
                "sandboxMode": True,
                "autoExecutionPolicy": "CASCADE_COMMANDS_AUTO_EXECUTION_EAGER",
                "artifactReviewMode": "ARTIFACT_REVIEW_MODE_TURBO",
            },
            "isWorkspaceOnly": True,
        },
    )
    write_json(
        profile / "config" / "mcp_config.json",
        {
            "mcpServers": {
                "ag": {
                    "command": "/opt/anaconda3/bin/python3",
                    "args": [str(MCP_SOURCE)],
                    "env": {
                        "AG_MCP_MAILBOX": str(bridge.mailbox),
                        "AG_MCP_TOKEN": bridge.token,
                        "AG_MCP_OBSERVATION_DIR": str(bridge.observation_dir),
                        "AG_MCP_WORKSPACE": str(workspace),
                    },
                }
            }
        },
    )


def approve_waiting_steps(ls: LocalLanguageServer, cascade: str, trajectory: dict[str, Any], approved: set[tuple[int, str]]) -> list[dict[str, Any]]:
    body = trajectory.get("trajectory", {})
    if not isinstance(body, dict):
        return []
    trajectory_id = body.get("trajectoryId")
    if not isinstance(trajectory_id, str):
        return []
    approvals: list[dict[str, Any]] = []
    for step in body.get("steps", []):
        if not isinstance(step, dict) or step.get("status") != "CORTEX_STEP_STATUS_WAITING":
            continue
        requested = step.get("requestedInteraction")
        if not isinstance(requested, dict):
            continue
        info = step.get("metadata", {}).get("sourceTrajectoryStepInfo", {})
        step_index = info.get("stepIndex")
        if not isinstance(step_index, int):
            continue
        kinds = [key for key in ("permission", "mcp", "runCommand", "filePermission", "approvalInteraction", "sendCommandInput") if key in requested]
        if not kinds:
            continue
        kind = kinds[0]
        marker = (step_index, kind)
        if marker in approved:
            continue
        interaction: dict[str, Any] = {"trajectoryId": trajectory_id, "stepIndex": step_index}
        if kind == "permission":
            interaction["permission"] = {"allow": True}
        elif kind == "mcp":
            interaction["mcp"] = {"confirm": True}
        elif kind == "runCommand":
            spec = requested.get("runCommand", {})
            submitted = spec.get("proposedCommandLine") if isinstance(spec, dict) else None
            response: dict[str, Any] = {"confirm": True}
            if isinstance(submitted, str) and submitted:
                response["submittedCommandLine"] = submitted
            interaction["runCommand"] = response
        elif kind == "filePermission":
            interaction["filePermission"] = {"allow": True}
        else:
            interaction[kind] = {"confirm": True}
        response = ls.rpc("/exa.language_server_pb.LanguageServerService/HandleCascadeUserInteraction", {"cascadeId": cascade, "interaction": interaction})
        approved.add(marker)
        approvals.append({"step_index": step_index, "kind": kind, "response": sanitize(response)})
    return approvals


def trajectory_tool_rows(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    body = trajectory.get("trajectory", {})
    if not isinstance(body, dict):
        return rows
    for index, step in enumerate(body.get("steps", [])):
        if not isinstance(step, dict):
            continue
        if step.get("type") == "CORTEX_STEP_TYPE_MCP_TOOL":
            metadata = step.get("metadata", {})
            call = metadata.get("toolCall", {})
            mcp = step.get("mcpTool", {})
            row = {
                "trajectory_index": index,
                "type": step.get("type"),
                "status": step.get("status"),
                "server": mcp.get("serverName"),
                "tool": mcp.get("toolCall", {}).get("name") if isinstance(mcp, dict) else None,
                "arguments": None,
                "result_string": mcp.get("resultString") if isinstance(mcp, dict) else None,
                "media": [],
            }
            arguments_json = call.get("argumentsJson") if isinstance(call, dict) else None
            if isinstance(arguments_json, str):
                try:
                    wrapper = json.loads(arguments_json)
                    row["arguments"] = wrapper.get("Arguments") if isinstance(wrapper, dict) else wrapper
                except json.JSONDecodeError:
                    row["arguments"] = None
            for media in mcp.get("media", []) if isinstance(mcp, dict) else []:
                if isinstance(media, dict):
                    inline = media.get("inlineData")
                    if isinstance(inline, str):
                        try:
                            raw = base64.b64decode(inline, validate=True)
                            row["media"].append({"mimeType": media.get("mimeType"), "byte_count": len(raw), "sha256": sha256_bytes(raw)})
                        except Exception:
                            row["media"].append({"mimeType": media.get("mimeType"), "byte_count": None})
            rows.append(row)
        elif step.get("type") == "CORTEX_STEP_TYPE_PLANNER_RESPONSE":
            response = step.get("plannerResponse", {})
            if isinstance(response, dict):
                rows.append({
                    "trajectory_index": index,
                    "type": step.get("type"),
                    "status": step.get("status"),
                    "response": response.get("response"),
                    "tool_call_count": len(response.get("toolCalls", [])) if isinstance(response.get("toolCalls"), list) else 0,
                })
    return rows


def verify_byok_model_route(
    trajectory: dict[str, Any],
    registry: dict[str, Any],
    api_audit: dict[str, Any],
    ordered_model_input_hashes: list[str],
) -> dict[str, Any]:
    """Fail closed if direct API routing diverges from the registered M299 route."""
    body = trajectory.get("trajectory", {})
    metadata = body.get("generatorMetadata", []) if isinstance(body, dict) else []
    enums: list[str] = []
    if isinstance(metadata, list):
        for row in metadata:
            if not isinstance(row, dict):
                continue
            chat = row.get("chatModel", {})
            if not isinstance(chat, dict):
                continue
            for candidate in (chat.get("model"), chat.get("customMetadata", {}).get("model_enum")):
                if isinstance(candidate, str) and candidate:
                    enums.append(candidate)
    registry_text = json.dumps(registry, ensure_ascii=True, sort_keys=True)
    registry_medium = MODEL_ENUM in registry_text and MODEL_LABEL in registry_text
    models = list(api_audit.get("planner_models", []))
    auxiliary_models = list(api_audit.get("auxiliary_models", []))
    thinking_levels = [str(value).lower() for value in api_audit.get("thinking_levels", [])]
    statuses = list(api_audit.get("statuses", []))
    per_call_images = api_audit.get("planner_model_call_image_sha256", [])
    call_routes = api_audit.get("call_routes", [])
    auxiliary_calls_are_image_and_action_free = all(
        API_MODEL in route.get("models", [])
        or (
            int(route.get("image_count", 0) or 0) == 0
            and int(route.get("function_call_count", 0) or 0) == 0
            and int(route.get("function_response_count", 0) or 0) == 0
        )
        for route in call_routes
        if isinstance(route, dict)
    )
    exact_media_request = any(row == ordered_model_input_hashes for row in per_call_images if isinstance(row, list))
    passed = (
        bool(enums)
        and set(enums) == {MODEL_ENUM}
        and registry_medium
        and models == [API_MODEL]
        and bool(statuses)
        and all(status == 200 for status in statuses)
        and exact_media_request
        and auxiliary_calls_are_image_and_action_free
        and (not thinking_levels or set(thinking_levels) == {THINKING_LEVEL})
    )
    result = {
        "passed": passed,
        "antigravity_model_enums": sorted(set(enums)),
        "registry_maps_enum_to_medium": registry_medium,
        "api_models": models,
        "auxiliary_models": auxiliary_models,
        "auxiliary_calls_are_image_and_action_free": auxiliary_calls_are_image_and_action_free,
        "explicit_api_thinking_levels": sorted(set(thinking_levels)),
        "thinking_level_basis": (
            "explicit Gemini API request"
            if thinking_levels
            else "Antigravity M299 registry mapping; Gemini 3.7 Flash API default"
        ),
        "all_model_http_statuses_200": bool(statuses) and all(status == 200 for status in statuses),
        "exact_ordered_media_present_in_model_request": exact_media_request,
    }
    if not passed:
        raise RuntimeError(f"BYOK model-route verification failed: {result}")
    return result


def verify_native_delivery(trajectory: dict[str, Any], bridge: EpisodeBridge) -> dict[str, Any]:
    body = trajectory.get("trajectory", {})
    steps = body.get("steps", []) if isinstance(body, dict) else []
    initial_ok = False
    initial_hash: str | None = None
    move_hashes: list[str] = []
    mcp_media_count = 0
    for step in steps:
        if not isinstance(step, dict):
            continue
        if step.get("type") == "CORTEX_STEP_TYPE_USER_INPUT":
            media = step.get("userInput", {}).get("media", [])
            if media and isinstance(media[0], dict) and isinstance(media[0].get("inlineData"), str):
                try:
                    initial_hash = sha256_bytes(base64.b64decode(media[0]["inlineData"], validate=True))
                    initial_ok = initial_hash == bridge.records[0]["png_sha256"]
                except Exception:
                    initial_ok = False
        if step.get("type") == "CORTEX_STEP_TYPE_MCP_TOOL":
            mcp = step.get("mcpTool", {})
            if not isinstance(mcp, dict):
                continue
            # Only `move` is expected to produce the observation sequence.  An
            # agent may legitimately call `view_image` on a previously saved
            # PNG; its image must not be compared against the move sequence or
            # counted as an extra move observation.
            tool_call = mcp.get("toolCall", {})
            if not isinstance(tool_call, dict) or tool_call.get("name") != "move":
                continue
            media = mcp.get("media", [])
            if isinstance(media, list) and media:
                for node in media:
                    if isinstance(node, dict) and isinstance(node.get("inlineData"), str):
                        try:
                            move_hashes.append(sha256_bytes(base64.b64decode(node["inlineData"], validate=True)))
                            mcp_media_count += 1
                        except Exception:
                            pass
    expected_move_hashes = [r["png_sha256"] for r in bridge.records[1:]]
    expected_observation_hashes = [r["png_sha256"] for r in bridge.records]
    model_input_hashes: list[str] = []
    generator_metadata = body.get("generatorMetadata", []) if isinstance(body, dict) else []
    if isinstance(generator_metadata, list):
        # The final metadata entry contains the detailed, rehydrated message
        # history that Antigravity supplied to the model.
        for metadata in reversed(generator_metadata):
            if not isinstance(metadata, dict):
                continue
            prompts = metadata.get("chatModel", {}).get("messagePrompts")
            if not isinstance(prompts, list):
                continue
            for prompt in prompts:
                if not isinstance(prompt, dict):
                    continue
                for media in prompt.get("media", []) if isinstance(prompt.get("media"), list) else []:
                    if not isinstance(media, dict) or not isinstance(media.get("inlineData"), str):
                        continue
                    try:
                        model_input_hashes.append(sha256_bytes(base64.b64decode(media["inlineData"], validate=True)))
                    except Exception:
                        pass
            break

    expected_cursor = 0
    for observed_hash in model_input_hashes:
        if expected_cursor < len(expected_observation_hashes) and observed_hash == expected_observation_hashes[expected_cursor]:
            expected_cursor += 1
    return {
        "initial_media_present": initial_ok,
        "initial_media_sha256": initial_hash,
        "move_media_count": mcp_media_count,
        "move_media_sha256": move_hashes,
        "move_media_all_match": move_hashes == expected_move_hashes,
        "expected_move_count": len(expected_move_hashes),
        "model_input_media_count": len(model_input_hashes),
        "model_input_media_sha256": model_input_hashes,
        # Diagnostic only. Antigravity controls later context compaction and
        # may legitimately retain a selected subset of earlier observations.
        # Scored validity is based on native delivery at the initial user
        # input and at every MCP move result, not cumulative reattachment of
        # the complete image history to a later planner request.
        "model_input_contains_all_observations_in_order": expected_cursor == len(expected_observation_hashes),
        "expected_observation_count": len(expected_observation_hashes),
    }


def wait_for_mcp_ready(ls: LocalLanguageServer, timeout_seconds: float = MCP_READY_TIMEOUT_SECONDS) -> dict[str, Any]:
    """Wait for the configured `ag` MCP server to finish its startup handshake.

    Antigravity reports the language server as authenticated before its stdio
    child is ready.  Polling the state endpoint also tolerates transient RPC
    errors while the MCP manager is still registering the server.  We select
    the named `ag` server rather than assuming it is the first state entry.
    """
    deadline = time.monotonic() + timeout_seconds
    last_state: dict[str, Any] = {}
    last_error: str | None = None
    while time.monotonic() < deadline:
        if ls.process is not None and ls.process.poll() is not None:
            raise RuntimeError(f"language server exited before MCP became READY (code {ls.process.returncode})")
        try:
            state = ls.rpc("/exa.language_server_pb.LanguageServerService/GetMcpServerStates", {})
            if isinstance(state, dict):
                last_state = state
            states = state.get("states", []) if isinstance(state, dict) else []
            selected = None
            for candidate in states:
                if not isinstance(candidate, dict):
                    continue
                spec = candidate.get("spec", {})
                server_name = spec.get("serverName") if isinstance(spec, dict) else None
                server_info = candidate.get("serverInfo", {})
                info_name = server_info.get("name") if isinstance(server_info, dict) else None
                if server_name == "ag" or info_name == "activeglimpse":
                    selected = candidate
                    break
            if isinstance(selected, dict) and selected.get("status") == "MCP_SERVER_STATUS_READY":
                tools = selected.get("tools", [])
                names = {tool.get("name") for tool in tools if isinstance(tool, dict)}
                if {"move", "view_image", "submit"}.issubset(names):
                    return state
                last_error = f"MCP READY without required tools: {sorted(names)}"
            else:
                last_error = "MCP server is still starting"
        except Exception as error:
            last_error = f"state query failed: {type(error).__name__}: {error}"
        time.sleep(0.5)
    detail = last_error or "no state returned"
    raise RuntimeError(f"local ag MCP server did not reach READY within {timeout_seconds:.0f}s ({detail})")


def leakage_test(notes: bytes, level: int, label: str, workspace: Path) -> dict[str, Any]:
    text = notes.decode("utf-8", errors="replace")
    findings: list[str] = []
    # This is an audit, not an answer oracle supplied to the agent.  A2's
    # permitted one-bit verdict is explicitly excluded from the test.
    if label and re.search(rf"(?<!\d){re.escape(label)}(?!\d)", text):
        findings.append("hidden_label_token")
    for marker in ("episode_id", "level_1_episode", "level_2_episode", "eval_index", "frequency", "tally", "accuracy", "ground_truth"):
        if marker.lower() in text.lower():
            findings.append(marker)
    for path_marker in (str(workspace), str(BASE), str(ACTIVEGLIMPSE)):
        if path_marker in text:
            findings.append("protected_path")
    return {"passed": not findings, "findings": sorted(set(findings)), "byte_count": len(notes), "line_count": text.count("\n") + (1 if text else 0)}


def run_episode(
    spec: EpisodeSpec,
    arm: str,
    run_root: Path,
    execution_position: int,
    module: Any,
    cache: dict[int, tuple[Image.Image, int]],
    incoming_notes: bytes,
    notes_a1: str,
    notes_a2: str,
    timeout_seconds: int,
    transport: str,
    gemini_api_key: str | None,
) -> tuple[dict[str, Any], bytes]:
    trace = run_root / "traces" / f"slot_{execution_position:03d}"
    trace.mkdir(parents=True, exist_ok=False)
    temp_root = Path(tempfile.mkdtemp(prefix="ag-episode-", dir="/private/tmp"))
    profile, workspace = temp_root / "profile", temp_root / "workspace"
    workspace.mkdir()
    if arm != "A0" and incoming_notes:
        (workspace / "NOTES.md").write_bytes(incoming_notes)
    env, label, initial_image = make_environment(module, spec, cache)
    bridge = EpisodeBridge(env, label, spec.level, arm, workspace)
    initial_record, initial_data = bridge.seed_initial(initial_image)
    project_id = str(uuid.uuid4())
    csrf = secrets.token_urlsafe(32)
    write_project_config(profile, project_id, workspace, bridge)
    proxy: GeminiAuditProxy | None = None
    if transport == TRANSPORT_GEMINI_API:
        if not gemini_api_key:
            raise RuntimeError("Gemini API transport was selected without a controller-side key")
        proxy = GeminiAuditProxy(gemini_api_key, trace / "model_calls.jsonl")
        proxy.start()
    ls = LocalLanguageServer(
        temp_root,
        project_id,
        csrf,
        workspace,
        {},
        trace / "language_server.stderr.log",
        transport=transport,
        proxy_url=proxy.url if proxy is not None else None,
    )
    bridge.start()
    start_time = time.monotonic()
    trajectory: dict[str, Any] = {}
    approvals: list[dict[str, Any]] = []
    run_error: str | None = None
    try:
        ls.start()
        registry = ls.rpc("/exa.language_server_pb.LanguageServerService/GetCascadeModelConfigData", {})
        mcp_state = wait_for_mcp_ready(ls)
        write_json(trace / "model_registry.json", sanitize(registry))
        write_json(trace / "mcp_state.json", sanitize(mcp_state))
        start = ls.rpc(
            "/exa.language_server_pb.LanguageServerService/StartCascade",
            {"source": "CORTEX_TRAJECTORY_SOURCE_AGENT_API", "trajectoryType": "CORTEX_TRAJECTORY_TYPE_USER_MAINLINE", "projectEnvConfig": {"projectId": project_id, "defaultProjectEnvironment": {}}},
        )
        cascade = start.get("cascadeId") or start.get("response", {}).get("cascadeId")
        if not isinstance(cascade, str) or not cascade:
            raise RuntimeError(f"StartCascade omitted cascade id: {start}")
        initial_prompt = prompt_for(spec.level, arm, notes_a1, notes_a2)
        send = ls.rpc(
            "/exa.language_server_pb.LanguageServerService/SendUserCascadeMessage",
            {
                "cascadeId": cascade,
                "items": [{"text": initial_prompt}],
                "media": [{"mimeType": "image/png", "description": "initial observation", "inlineData": base64.b64encode(initial_data).decode("ascii")}],
                "cascadeConfig": {"plannerConfig": {"google": {}, "planModel": MODEL_ENUM}},
                "blocking": False,
                "propagateError": True,
            },
        )
        write_json(trace / "start.json", sanitize(start))
        write_json(trace / "send_initial.json", {"response": sanitize(send), "request_media": {"mime_type": "image/png", "byte_count": len(initial_data), "sha256": sha256_bytes(initial_data)}, "prompt_sha256": sha256_bytes(initial_prompt.encode("utf-8"))})
        approved: set[tuple[int, str]] = set()
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            time.sleep(1.0)
            trajectory = ls.rpc("/exa.language_server_pb.LanguageServerService/GetCascadeTrajectory", {"cascadeId": cascade, "disableRehydration": False})
            approvals.extend(approve_waiting_steps(ls, cascade, trajectory, approved))
            status = trajectory.get("status")
            if status in {"CASCADE_RUN_STATUS_ERROR", "CASCADE_RUN_STATUS_FAILED"}:
                raise RuntimeError(f"cascade ended with {status}")
            if bridge.submit_calls == 1 and status == "CASCADE_RUN_STATUS_IDLE":
                break
        else:
            raise RuntimeError("episode timed out while waiting for the cascade")
        if bridge.submit_calls != 1 or bridge.answer is None:
            raise RuntimeError("cascade ended without exactly one valid submission")
        if bridge.delivery_mismatches:
            raise RuntimeError("native MCP delivery receipt mismatch")
        delivery = verify_native_delivery(trajectory, bridge)
        if (
            not delivery["initial_media_present"]
            or not delivery["move_media_all_match"]
        ):
            raise RuntimeError(f"native image delivery verification failed: {delivery}")
        api_audit = proxy.summary() if proxy is not None else None
        route_verification = (
            verify_byok_model_route(
                trajectory,
                registry,
                api_audit,
                delivery["model_input_media_sha256"],
            )
            if api_audit is not None
            else None
        )
        notes = (workspace / "NOTES.md").read_bytes() if arm != "A0" and (workspace / "NOTES.md").is_file() else b""
        leakage = leakage_test(notes, spec.level, label, workspace) if arm != "A0" else {"passed": True, "findings": [], "byte_count": 0, "line_count": 0}
        if arm != "A0":
            history_path = run_root / "notes_history" / f"ep_{execution_position:03d}.md"
            history_path.parent.mkdir(parents=True, exist_ok=True)
            history_path.write_bytes(notes)
            write_json(trace / "notes_leakage.json", leakage)
        report = {
            "execution_position": execution_position,
            "episode_id": spec.episode_id,
            "level": spec.level,
            "arm": arm,
            "validity": "VALID",
            "answer": bridge.answer,
            "ground_truth_label": label,
            "correctness": bridge.correct,
            "counters": bridge.counters(),
            "native_image_delivery": delivery,
            "tool_calls": trajectory_tool_rows(trajectory),
            "approvals": approvals,
            "model": {
                "label": MODEL_LABEL,
                "enum": MODEL_ENUM,
                "api_model": API_MODEL,
                "thinking_level": THINKING_LEVEL,
                "transport": transport,
            },
            "model_api_audit": api_audit,
            "model_route_verification": route_verification,
            "credential_isolation": {
                "real_key_supplied_to_language_server": False if proxy is not None else None,
                "injection_boundary": "controller-localhost-proxy" if proxy is not None else "app-account-keychain",
            },
            "fresh_language_server": True,
            "fresh_profile": True,
            "fresh_workspace": True,
            "elapsed_seconds": time.monotonic() - start_time,
            "trajectory_status": trajectory.get("status"),
            "registry_snapshot": sanitize(registry),
        }
    except Exception as error:
        run_error = f"{type(error).__name__}: {error}"
        notes = (workspace / "NOTES.md").read_bytes() if arm != "A0" and (workspace / "NOTES.md").is_file() else b""
        if arm != "A0":
            history_path = run_root / "notes_history" / f"ep_{execution_position:03d}.md"
            history_path.parent.mkdir(parents=True, exist_ok=True)
            history_path.write_bytes(notes)
            write_json(trace / "notes_leakage.json", leakage_test(notes, spec.level, label, workspace))
        report = {
            "execution_position": execution_position, "episode_id": spec.episode_id, "level": spec.level, "arm": arm,
            "validity": "INVALID", "answer": bridge.answer, "ground_truth_label": label,
            "correctness": bridge.correct, "counters": bridge.counters(), "error": run_error,
            "native_image_delivery": verify_native_delivery(trajectory, bridge) if trajectory else None,
            "tool_calls": trajectory_tool_rows(trajectory) if trajectory else [], "approvals": approvals,
            "model": {
                "label": MODEL_LABEL,
                "enum": MODEL_ENUM,
                "api_model": API_MODEL,
                "thinking_level": THINKING_LEVEL,
                "transport": transport,
            },
            "model_api_audit": proxy.summary() if proxy is not None else None,
            "model_route_verification": None,
            "credential_isolation": {
                "real_key_supplied_to_language_server": False if proxy is not None else None,
                "injection_boundary": "controller-localhost-proxy" if proxy is not None else "app-account-keychain",
            },
            "fresh_language_server": True,
            "fresh_profile": True, "fresh_workspace": True, "elapsed_seconds": time.monotonic() - start_time,
        }
    finally:
        bridge.stop()
        try:
            ls.stop()
        finally:
            if proxy is not None:
                proxy.stop()
            if (temp_root / "profile" / "antigravity" / "brain").is_dir():
                shutil.copytree(temp_root / "profile" / "antigravity" / "brain", trace / "brain", dirs_exist_ok=True)
            if (temp_root / "profile" / "antigravity" / "conversations").is_dir():
                shutil.copytree(temp_root / "profile" / "antigravity" / "conversations", trace / "conversations", dirs_exist_ok=True)
            if (temp_root / "workspace" / "observations").is_dir():
                shutil.copytree(temp_root / "workspace" / "observations", trace / "observations", dirs_exist_ok=True)
            shutil.rmtree(temp_root, ignore_errors=True)
    write_json(trace / "trajectory.json", sanitize(trajectory))
    write_json(trace / "episode.json", report)
    return report, notes


def schedule_payload(specs: list[EpisodeSpec]) -> list[dict[str, Any]]:
    return [{"execution_position": i + 1, "episode_id": spec.episode_id, "indices": list(spec.indices)} for i, spec in enumerate(specs)]


def offline_contract(
    run_root: Path,
    arm: str,
    level: int,
    suite_id: str,
    specs: list[EpisodeSpec],
    source_hash: str,
    notes_a1: str,
    notes_a2: str,
    transport: str,
) -> None:
    schedule = schedule_payload(specs)
    record = {
        "passed": True,
        "arm": arm,
        "level": level,
        "suite_id": suite_id,
        "controller_sha256": sha256_file(Path(__file__).resolve()),
        "source_manifest_sha256": source_hash,
        "schedule_sha256": sha256_bytes(canonical(schedule)),
        "notes_enabled": arm in {"A1", "A2"},
        "feedback_enabled": arm == "A2",
        "notes_size_cap_enforced": False,
        "expected_episode_count": EXPECTED_EPISODES,
        "model_transport": transport,
        "scored_api_calls": 0 if transport == TRANSPORT_APP_ACCOUNT else "recorded_per_episode_by_local_proxy",
        "api_model": API_MODEL,
        "thinking_level": THINKING_LEVEL,
        "agent": "antigravity-local-language-server",
        "underlying_model": MODEL_LABEL,
        "native_image_transport": "SendUserCascadeMessage.media.inlineData plus MCP media.inlineData",
        "prompt_sha256": sha256_bytes(prompt_for(level, arm, notes_a1, notes_a2).encode("utf-8")),
    }
    suffix = "" if transport == TRANSPORT_APP_ACCOUNT else "_gemini_api_byok"
    write_json(run_root / "preflight" / f"offline_contract{suffix}.json", record)
    write_json(run_root / f"schedule_manifest{suffix}.json", {"configuration": record, "execution_schedule": schedule})


def load_existing_episode_reports(run_root: Path) -> dict[int, dict[str, Any]]:
    """Load the latest report for each logical execution position.

    A repaired continuation must not replay successful episodes, because doing
    so would change the notes state and spend model quota twice.  Incomplete
    slot directories are deliberately ignored here and archived by the caller.
    """
    reports: dict[int, dict[str, Any]] = {}
    traces = run_root / "traces"
    if not traces.is_dir():
        return reports
    for path in sorted(traces.glob("slot_*/episode.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        position = report.get("execution_position")
        if isinstance(position, int) and 1 <= position <= EXPECTED_EPISODES:
            reports[position] = report
    return reports


def report_transport(report: dict[str, Any]) -> str:
    transport = report.get("model", {}).get("transport")
    return transport if isinstance(transport, str) and transport else TRANSPORT_APP_ACCOUNT


def aggregate_metrics(reports: list[dict[str, Any]]) -> dict[str, Any]:
    valid_reports = [report for report in reports if report.get("validity") == "VALID"]
    correct = sum(report.get("correctness") is True for report in valid_reports)
    by_transport: dict[str, dict[str, Any]] = {}
    by_enum: dict[str, int] = {}
    for report in reports:
        transport = report_transport(report)
        bucket = by_transport.setdefault(transport, {"attempted": 0, "valid": 0, "invalid": 0, "correct": 0, "model_calls": 0})
        bucket["attempted"] += 1
        if report.get("validity") == "VALID":
            bucket["valid"] += 1
            bucket["correct"] += int(report.get("correctness") is True)
            enum = report.get("model", {}).get("enum")
            if isinstance(enum, str):
                by_enum[enum] = by_enum.get(enum, 0) + 1
        else:
            bucket["invalid"] += 1
        audit = report.get("model_api_audit")
        if isinstance(audit, dict):
            bucket["model_calls"] += int(audit.get("model_calls", 0) or 0)
    for bucket in by_transport.values():
        bucket["accuracy"] = bucket["correct"] / bucket["valid"] if bucket["valid"] else None
    active_transports = sorted(transport for transport, bucket in by_transport.items() if bucket["valid"])
    return {
        "attempted": len(reports),
        "valid": len(valid_reports),
        "invalid": len(reports) - len(valid_reports),
        "correct": correct,
        "accuracy": correct / len(valid_reports) if valid_reports else None,
        "transport_breakdown": by_transport,
        "valid_model_enum_counts": by_enum,
        "mixed_model_transport": len(active_transports) > 1,
    }


def archive_stale_trace(run_root: Path, trace_dir: Path, reason: str) -> Path:
    """Move an interrupted/failed trace aside without deleting evidence."""
    # A previous interrupted controller can have already archived this same
    # slot.  Treat that as completed cleanup rather than failing the arm.
    if not trace_dir.is_dir():
        return trace_dir
    archive_root = run_root / "repair_history" / reason
    archive_root.mkdir(parents=True, exist_ok=True)
    destination = archive_root / trace_dir.name
    suffix = 1
    while destination.exists():
        destination = archive_root / f"{trace_dir.name}_{suffix:02d}"
        suffix += 1
    try:
        shutil.move(str(trace_dir), str(destination))
    except FileNotFoundError:
        # Another stale-cleanup pass won the race; no evidence remains to move.
        return trace_dir
    return destination


def retryable_infrastructure_failure(report: dict[str, Any]) -> bool:
    """Return true only for non-agent failures that require a fresh LS session.

    A quota rejection may leave Antigravity's cascade non-terminal, so the
    controller records it as an episode timeout rather than surfacing the 429
    in ``episode.json``.  Those attempts neither submitted an answer nor
    produced a score and must be retried after credentials/quota are restored.
    """
    error = str(report.get("error", ""))
    return (
        "did not reach READY" in error
        or "did not authenticate" in error
        or "no valid auth" in error
        or "failed to read authorization code" in error
        or "native image delivery verification failed" in error
        or "RESOURCE_EXHAUSTED" in error
        or "episode timed out while waiting for the cascade" in error
        or "curl: (7)" in error
        or "Failed to connect to 127.0.0.1" in error
        or "BYOK model-route verification failed" in error
    )


def load_notes_state(run_root: Path, arm: str, position: int) -> bytes:
    """Load the committed notes snapshot for a completed logical position."""
    if arm == "A0" or position <= 0:
        return b""
    path = run_root / "notes_history" / f"ep_{position:03d}.md"
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return b""


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=("A0", "A1", "A2"))
    parser.add_argument("--level", required=True, type=int, choices=(1, 2))
    parser.add_argument("--suite-id", required=True)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--credential-source", required=False, default="keychain")
    parser.add_argument("--barrier-file", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--episode-timeout-seconds", type=int, default=1800)
    parser.add_argument("--episode-limit", type=int, default=EXPECTED_EPISODES, help=argparse.SUPPRESS)
    parser.add_argument("--resume", action="store_true", help="continue an existing run root without replaying successful positions")
    parser.add_argument(
        "--model-transport",
        choices=("app-account", "gemini-api"),
        default="app-account",
        help="Antigravity model transport; both choices retain the same cascade and MCP agent loop",
    )
    parser.add_argument(
        "--gemini-env-file",
        type=Path,
        help="controller-only dotenv file containing GEMINI_API_KEY; never copied into the episode workspace",
    )
    parser.add_argument(
        "--rerun-valid-model-mismatches",
        action="store_true",
        help="archive and rerun valid positions whose recorded model enum differs from the current controller model",
    )
    args = parser.parse_args(argv)
    transport = TRANSPORT_GEMINI_API if args.model_transport == "gemini-api" else TRANSPORT_APP_ACCOUNT
    gemini_api_key: str | None = None
    if transport == TRANSPORT_GEMINI_API:
        if args.gemini_env_file is None:
            parser.error("--gemini-env-file is required with --model-transport gemini-api")
        gemini_api_key = read_dotenv_secret(args.gemini_env_file.resolve())
    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    failure = run_root / "CONTROLLER_FAILURE.txt"
    failure.unlink(missing_ok=True)
    notes_a1_path = Path("/Users/poriasoujanya/Downloads/notes_instruction_A1_no_feedback.md")
    notes_a2_path = Path("/Users/poriasoujanya/Downloads/notes_instruction_A2_with_feedback.md")
    try:
        specs, source_hash = load_specs(args.level)
        notes_a1 = extract_notes_block(notes_a1_path, "A1")
        notes_a2 = extract_notes_block(notes_a2_path, "A2")
        module = load_environment_module()
        cache = preload(args.level, specs)
        offline_contract(run_root, args.arm, args.level, args.suite_id, specs, source_hash, notes_a1, notes_a2, transport)
        barrier_name = "BARRIER_READY_GEMINI_API.json" if transport == TRANSPORT_GEMINI_API else "BARRIER_READY.json"
        write_json(run_root / barrier_name, {
            "ready": True,
            "arm": args.arm,
            "level": args.level,
            "suite_id": args.suite_id,
            "controller_pid": os.getpid(),
            "controller_sha256": sha256_file(Path(__file__).resolve()),
            "source_manifest_sha256": source_hash,
            "model_transport": transport,
            "scored_api_calls": 0 if transport == TRANSPORT_APP_ACCOUNT else "recorded_per_episode_by_local_proxy",
        })
        if args.preflight_only:
            return 0
        if args.barrier_file is not None:
            deadline = time.monotonic() + 1800
            while not args.barrier_file.is_file():
                if time.monotonic() >= deadline:
                    raise RuntimeError("preflight barrier timed out")
                time.sleep(0.2)
        reports_by_position = load_existing_episode_reports(run_root) if args.resume else {}
        if args.resume:
            # A process stop can leave a slot directory with no terminal report.
            # Preserve it under repair_history, then rerun that logical position
            # from the last committed notes snapshot.
            traces_root = run_root / "traces"
            for trace_dir in sorted(traces_root.glob("slot_*")):
                try:
                    position = int(trace_dir.name.split("_", 1)[1])
                except (ValueError, IndexError):
                    continue
                report = reports_by_position.get(position)
                if report is None and trace_dir.is_dir():
                    archive_stale_trace(run_root, trace_dir, "interrupted")

        reports: list[dict[str, Any]] = []
        notes = b""
        selected_specs = specs[: max(0, min(args.episode_limit, len(specs)))]
        retry_count = 0
        for position, spec in enumerate(selected_specs, start=1):
            existing = reports_by_position.get(position)
            existing_model_enum = (existing or {}).get("model", {}).get("enum")
            valid_model_mismatch = bool(
                args.rerun_valid_model_mismatches
                and existing is not None
                and existing.get("validity") == "VALID"
                and existing_model_enum != MODEL_ENUM
            )
            if existing is not None and existing.get("validity") == "VALID" and not valid_model_mismatch:
                reports.append(existing)
                notes = load_notes_state(run_root, args.arm, position)
                continue
            if existing is not None and not valid_model_mismatch and not retryable_infrastructure_failure(existing):
                # Preserve non-infrastructure failures as part of the report,
                # but do not silently spend another long model episode.
                reports.append(existing)
                notes = load_notes_state(run_root, args.arm, position)
                continue
            old_trace = run_root / "traces" / f"slot_{position:03d}"
            if old_trace.is_dir():
                archive_reason = "model_tier_replaced" if valid_model_mismatch else "retried"
                archive_stale_trace(run_root, old_trace, archive_reason)
                if args.arm != "A0":
                    old_notes = run_root / "notes_history" / f"ep_{position:03d}.md"
                    if old_notes.is_file():
                        archive_root = run_root / "repair_history" / "retried_notes"
                        archive_root.mkdir(parents=True, exist_ok=True)
                        archive_path = archive_root / old_notes.name
                        suffix = 1
                        while archive_path.exists():
                            archive_path = archive_root / f"{old_notes.stem}_{suffix:02d}{old_notes.suffix}"
                            suffix += 1
                        shutil.move(str(old_notes), str(archive_path))
            notes = load_notes_state(run_root, args.arm, position - 1)
            report, notes = run_episode(
                spec,
                args.arm,
                run_root,
                position,
                module,
                cache,
                notes,
                notes_a1,
                notes_a2,
                args.episode_timeout_seconds,
                transport,
                gemini_api_key,
            )
            reports_by_position[position] = report
            reports.append(report)
            retry_count += 1 if existing is not None else 0
            with (run_root / "progress.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(report, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            # Include every currently terminal logical slot, not only the
            # prefix traversed during this resume pass. This keeps periodic
            # monitoring accurate while older valid slots are skipped.
            live_reports = [reports_by_position[key] for key in sorted(reports_by_position)]
            live = aggregate_metrics(live_reports)
            live.update({
                "arm": args.arm,
                "level": args.level,
                "current_position": position,
                "active_transport": transport,
                "updated_at": utc_now(),
            })
            write_json(run_root / "live_progress.json", live)
        metrics = aggregate_metrics(reports)
        valid = int(metrics["valid"])
        correct = int(metrics["correct"])
        summary = {
            "arm": args.arm,
            "level": args.level,
            **metrics,
            "schedule_complete": len(reports) == EXPECTED_EPISODES,
            "native_model": MODEL_LABEL,
            "model_enum": MODEL_ENUM,
            "active_transport": transport,
            "viewer_calls": sum(int(report.get("counters", {}).get("viewer_calls", 0)) for report in reports),
            "mcp_move_calls": sum(int(report.get("counters", {}).get("accepted_moves", 0)) for report in reports),
            "mcp_submit_calls": sum(int(report.get("counters", {}).get("submit_calls", 0)) for report in reports),
            "completed_at": utc_now(),
            "resumed": bool(args.resume),
            "infrastructure_retries": retry_count,
        }
        write_json(run_root / "summary.json", summary)
        write_json(run_root / "ARM_COMPLETE.json", {"complete": summary["schedule_complete"], "arm": args.arm, "level": args.level, "attempted": len(reports), "valid": valid, "completed_at": utc_now()})
        return 0
    except Exception:
        failure.write_text(traceback.format_exc(), encoding="utf-8")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
