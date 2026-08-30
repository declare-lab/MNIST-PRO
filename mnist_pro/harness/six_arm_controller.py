#!/opt/anaconda3/bin/python
"""Clean-room MNIST-PRO controller for the managed Antigravity agent.

This program is controller-only.  Each scored episode creates a new Gemini
Interaction and a new remote environment.  Images are delivered twice: as an
inline base64 source mounted in that environment and as native image content in
the model input/function result.  No correctness, schedule, or label data is
placed in the remote environment.
"""

from __future__ import annotations

import argparse
import base64
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import re
import secrets
import stat
import subprocess
import sys
import tarfile
import time
import traceback
from typing import Any

import httpx
import numpy as np
from PIL import Image
from torchvision.datasets import MNIST


BASE = Path("/Users/poriasoujanya/Documents/Codex/2026-08-25/files-pasted-by-the-user-act")
ACTIVEGLIMPSE = Path("/Users/poriasoujanya/Documents/activeglimpse-main")
ENVIRONMENT_PY = ACTIVEGLIMPSE / "src/environment.py"
DATA_ROOT = BASE / "work/mnist_pro_cleanroom/controller_private/data"
LEVEL_SUMMARIES = {
    1: ACTIVEGLIMPSE / (
        "main_table_logs/gemini-3.7-flash_img224_box64_step32_maxsteps36_seed42_"
        "DefaultVisionAgent_hist-1_evalsets10_20260824_105544/results_summary.json"
    ),
    2: ACTIVEGLIMPSE / (
        "main_table_logs/multidigit_2_gemini-3.7-flash_img224_box64_step32_"
        "maxsteps78_seed42_MultiDigitDefaultVisionAgent_hist-1_evalsets10_"
        "20260824_111931/results_summary.json"
    ),
}
PINNED = {
    "environment": "d2973d5bb60ce38501f8b4066cf916510e7c26a3d4d0fa0659d2b817326736c4",
    "summary_1": "48362448e87a819e162bc42ac9104493ad8b3405bf65c1cec52418e88b21c208",
    "summary_2": "d1778e114011974e2416bd45ef2dba15047c8a065596e173b6e3b6daf620c465",
}
AGENT = "antigravity-preview-05-2026"
MODEL = "gemini-3.7-flash"
API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
IMAGE_SIZE = 224
BOX_SIZE = 64
STEP_SIZE = 32
BUDGETS = {1: 36, 2: 78}
EXPECTED_EPISODES = 100
MAX_TOTAL_TOKENS = 50_000
MAX_SOURCE_BYTES = 2_000_000
MAX_INLINE_FILE_BYTES = 1_000_000
SHUFFLE_SEED_HEX = (
    "3c86be160cd0b7dde05768c2358b13aa7255b7968f5fb1041c6b8f96f2656d04"
)

A1_NOTES = """PERSISTENT NOTES

You have a file NOTES.md in your workspace. It is the only thing that survives
after this episode ends. Everything else, including this conversation and every
image you have seen, is discarded. Future episodes will show you this file and
nothing else.

You may rewrite NOTES.md in full at any point. There is no size cap.

Write whatever you think will help you in future episodes.

You will not be told whether your answer was right."""

A2_NOTES = """PERSISTENT NOTES

You have a file NOTES.md in your workspace. It is the only thing that survives
after this episode ends. Everything else, including this conversation and every
image you have seen, is discarded. Future episodes will show you this file and
nothing else.

You may rewrite NOTES.md in full at any point. There is no size cap.

Write whatever you think will help you in future episodes.

After you submit, you will be told CORRECT or INCORRECT. You will never be told
what the right answer was. You may record this however you like."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


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
    with temporary.open("wb") as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False).encode("utf-8") + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_gzip(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    with gzip.open(temporary, "wb", compresslevel=6) as handle:
        handle.write(value)
    os.replace(temporary, path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(canonical({"timestamp": utc_now(), **value}) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def png_bytes(image: Image.Image) -> bytes:
    stream = io.BytesIO()
    image.convert("RGB").save(stream, format="PNG")
    return stream.getvalue()


def image_record(data: bytes, remote_path: str, image_id: str, index: int) -> dict[str, Any]:
    with Image.open(io.BytesIO(data)) as image:
        rgb = np.asarray(image.convert("RGB")).tobytes()
        size = list(image.size)
    return {
        "observation_index": index,
        "image_id": image_id,
        "remote_path": remote_path,
        "png_sha256": sha256_bytes(data),
        "pixel_sha256": sha256_bytes(rgb),
        "byte_count": len(data),
        "size": size,
    }


def native_image_nodes(value: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if value.get("type") == "image" and value.get("mime_type") == "image/png" and isinstance(value.get("data"), str):
            found.append(value)
        for child in value.values():
            found.extend(native_image_nodes(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(native_image_nodes(child))
    return found


def notes_instruction(arm: str) -> str | None:
    return {"A0": None, "A1": A1_NOTES, "A2": A2_NOTES}[arm]


def prompt_for(level: int, arm: str) -> str:
    if level == 1:
        level_text = (
            "For Level 1, the hidden canvas contains one MNIST digit from 0 "
            "through 9. Submit one digit."
        )
    else:
        level_text = (
            "For Level 2, the hidden canvas contains two horizontally concatenated "
            "MNIST digits. Submit the two digits in left-to-right order, preserving "
            "a leading zero when applicable."
        )
    baseline = (
        "You are solving one partially observed MNIST episode.\n\n"
        f"{level_text}\n\n"
        "Most of the canvas is masked. The attached image is your initial observation. "
        "Reveal another observation by calling `activeglimpse_move` with `direction` "
        "set to `up`, `down`, `left`, or `right`. Each successful move returns the new "
        "observation directly as native image content and gives its opaque PNG path.\n\n"
        "You may reopen any exposed observation by calling `activeglimpse_view_image`. "
        "You may also use the sandbox filesystem and code-execution tools autonomously.\n\n"
        "Submit exactly once by calling `activeglimpse_submit` with your answer in "
        "`value`.\n\n"
        f"You have at most {BUDGETS[level]} visual observations, including the attached "
        "initial observation. Use the local tools and workspace autonomously. Finish "
        "by submitting your best answer."
    )
    extra = notes_instruction(arm)
    return baseline if extra is None else baseline + "\n\n" + extra


def tool_definitions(level: int) -> list[dict[str, Any]]:
    value_pattern = "^[0-9]$" if level == 1 else "^[0-9]{2}$"
    return [
        {"type": "code_execution"},
        {
            "type": "function",
            "name": "activeglimpse_move",
            "description": "Move the reveal window once and return the resulting PNG natively.",
            "parameters": {
                "type": "object",
                "properties": {"direction": {"type": "string", "enum": ["up", "down", "left", "right"]}},
                "required": ["direction"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "activeglimpse_view_image",
            "description": "Reopen a previously exposed PNG by its opaque path or image id.",
            "parameters": {
                "type": "object",
                "properties": {"path_or_id": {"type": "string"}},
                "required": ["path_or_id"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "activeglimpse_submit",
            "description": "Submit the final digit answer exactly once.",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string", "pattern": value_pattern}},
                "required": ["value"],
                "additionalProperties": False,
            },
        },
    ]


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
        indices = (int(row["eval_index"]),) if level == 1 else tuple(row["indices"])
        specs.append(EpisodeSpec(level, int(row["episode_id"]), indices))
    if [item.episode_id for item in specs] != list(range(EXPECTED_EPISODES)):
        raise RuntimeError("episode ids are not exactly 0 through 99")
    random.Random(int(SHUFFLE_SEED_HEX, 16)).shuffle(specs)
    return specs, expected


def schedule_payload(specs: list[EpisodeSpec]) -> list[dict[str, Any]]:
    return [
        {"execution_position": i + 1, "episode_id": s.episode_id, "indices": list(s.indices)}
        for i, s in enumerate(specs)
    ]


def load_environment_module() -> Any:
    if sha256_file(ENVIRONMENT_PY) != PINNED["environment"]:
        raise RuntimeError("pinned ActiveGlimpse environment changed")
    spec = importlib.util.spec_from_file_location("antigravity_hidden_environment", ENVIRONMENT_PY)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not import ActiveGlimpse environment")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preload(specs: list[EpisodeSpec]) -> dict[int, tuple[Image.Image, int]]:
    dataset = MNIST(root=str(DATA_ROOT), train=False, download=False)
    result: dict[int, tuple[Image.Image, int]] = {}
    for index in sorted({i for spec in specs for i in spec.indices}):
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


def resolve_credential(source: str) -> str:
    if source == "keychain":
        completed = subprocess.run(
            ["/usr/bin/security", "find-generic-password", "-s", "gemini", "-a", "antigravity", "-w"],
            capture_output=True,
            check=True,
            timeout=15,
        )
        value = completed.stdout.decode("utf-8").strip()
    elif source.startswith("env:"):
        value = os.environ.get(source[4:], "").strip()
    else:
        text = source[5:] if source.startswith("file:") else source
        path = Path(text)
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid() or before.st_nlink != 1 or stat.S_IMODE(before.st_mode) != 0o600:
            raise RuntimeError("credential file must be owner-only regular file mode 0600")
        raw = path.read_text(encoding="utf-8").strip()
        if raw.startswith("{"):
            value = str(json.loads(raw).get("api_key", "")).strip()
        else:
            value = raw
    if not value or len(value) > 4096 or any(ch.isspace() for ch in value):
        raise RuntimeError("Gemini credential source did not yield one valid token")
    return value


class API:
    def __init__(self, api_key: str, trace_dir: Path, timeout_seconds: int):
        self.api_key = api_key
        self.trace_dir = trace_dir
        self.client = httpx.Client(timeout=httpx.Timeout(timeout_seconds), follow_redirects=True)
        self.model_calls = 0
        self.auxiliary_calls = 0

    @property
    def headers(self) -> dict[str, str]:
        return {
            "content-type": "application/json",
            "x-goog-api-key": self.api_key,
            "Api-Revision": "2026-05-20",
        }

    def interaction(self, body: dict[str, Any], turn: int) -> dict[str, Any]:
        request_bytes = canonical(body)
        prefix = self.trace_dir / f"turn_{turn:03d}"
        write_gzip(prefix.with_suffix(".request.json.gz"), request_bytes)
        write_json(prefix.with_suffix(".request_meta.json"), {
            "url": f"{API_ROOT}/interactions", "request_sha256": sha256_bytes(request_bytes),
            "request_bytes": len(request_bytes), "credential_logged": False,
        })
        self.model_calls += 1
        response = self.client.post(f"{API_ROOT}/interactions", content=request_bytes, headers=self.headers)
        write_gzip(prefix.with_suffix(".response.json.gz"), response.content)
        write_json(prefix.with_suffix(".response_meta.json"), {
            "status_code": response.status_code, "response_sha256": sha256_bytes(response.content),
            "response_bytes": len(response.content),
        })
        if not 200 <= response.status_code < 300:
            raise RuntimeError(f"Interactions API HTTP {response.status_code}: {response.text[:500]}")
        result = response.json()
        if not isinstance(result, dict):
            raise RuntimeError("Interactions API returned non-object JSON")
        return result

    def download_environment(self, environment_id: str) -> bytes:
        self.auxiliary_calls += 1
        response = self.client.get(
            f"{API_ROOT}/files/environment-{environment_id}:download",
            params={"alt": "media"}, headers={"x-goog-api-key": self.api_key, "Api-Revision": "2026-05-20"},
        )
        write_json(self.trace_dir / "environment_download.json", {
            "status_code": response.status_code, "byte_count": len(response.content),
            "sha256": sha256_bytes(response.content), "credential_logged": False,
        })
        if not 200 <= response.status_code < 300:
            raise RuntimeError(f"environment download failed with HTTP {response.status_code}")
        return response.content

    def delete_environment(self, environment_id: str) -> bool:
        self.auxiliary_calls += 1
        response = self.client.delete(
            f"{API_ROOT}/environments/{environment_id}",
            headers={"x-goog-api-key": self.api_key, "Api-Revision": "2026-05-20"},
        )
        write_json(self.trace_dir / "environment_delete.json", {"status_code": response.status_code, "credential_logged": False})
        return 200 <= response.status_code < 300 or response.status_code == 404

    def close(self) -> None:
        self.client.close()


class Evaluator:
    def __init__(self, env: Any, label: str, level: int, arm: str):
        self.env, self.label, self.level, self.arm = env, label, level, arm
        self.images: dict[str, bytes] = {}
        self.records: list[dict[str, Any]] = []
        self.glimpses = 0
        self.move_requests = self.accepted_moves = self.wall_bumps = self.effective_moves = 0
        self.effective_revisits = self.submit_calls = self.invalid_requests = 0
        self.positions = {(int(env.x), int(env.y))}
        self.position_history = [(int(env.x), int(env.y))]
        self.answer: str | None = None
        self.correct: bool | None = None
        self.locked = False
        self.source_bytes = 0

    def add_image(self, data: bytes) -> tuple[dict[str, Any], dict[str, Any]]:
        if len(data) > MAX_INLINE_FILE_BYTES:
            raise RuntimeError("one mounted image exceeds inline-source file limit")
        if self.source_bytes + len(data) > MAX_SOURCE_BYTES:
            raise RuntimeError("aggregate mounted source bytes would exceed 2 MB")
        self.source_bytes += len(data)
        image_id = secrets.token_hex(16)
        remote_path = f"/workspace/observations/{secrets.token_hex(16)}.png"
        record = image_record(data, remote_path, image_id, self.glimpses)
        self.glimpses += 1
        self.records.append(record)
        self.images[image_id] = data
        self.images[remote_path] = data
        source = {"type": "inline", "target": remote_path, "content": base64.b64encode(data).decode("ascii"), "encoding": "base64"}
        return record, source

    @staticmethod
    def multimodal(record: dict[str, Any], data: bytes) -> list[dict[str, Any]]:
        return [
            {"type": "text", "text": json.dumps({"image_id": record["image_id"], "path": record["remote_path"]}, separators=(",", ":"))},
            {"type": "image", "mime_type": "image/png", "data": base64.b64encode(data).decode("ascii")},
        ]

    def execute(self, name: str, arguments: Any) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
        if not isinstance(args, dict):
            args = {}
        sources: list[dict[str, Any]] = []
        audit: dict[str, Any] = {"name": name, "arguments": args}
        if name == "activeglimpse_move":
            self.move_requests += 1
            direction = args.get("direction")
            if self.locked:
                result: Any = {"error": "episode locked"}
            elif direction not in {"up", "down", "left", "right"}:
                self.invalid_requests += 1
                result = {"error": "invalid direction"}
            elif self.glimpses >= BUDGETS[self.level]:
                result = {"error": "observation budget exhausted"}
            else:
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
                data = png_bytes(observation)
                record, source = self.add_image(data)
                sources.append(source)
                result = self.multimodal(record, data)
                audit.update({"accepted": True, "image": record})
        elif name == "activeglimpse_view_image":
            key = args.get("path_or_id")
            data = self.images.get(key) if isinstance(key, str) else None
            if data is None:
                self.invalid_requests += 1
                result = {"error": "unknown image path or id"}
            else:
                record = next(row for row in self.records if row["image_id"] == key or row["remote_path"] == key)
                result = self.multimodal(record, data)
                audit.update({"accepted": True, "image": record})
        elif name == "activeglimpse_submit":
            self.submit_calls += 1
            raw = args.get("value")
            value = raw if isinstance(raw, str) else ""
            valid = bool(re.fullmatch(r"[0-9]", value) if self.level == 1 else re.fullmatch(r"[0-9]{2}", value))
            if self.locked:
                result = {"error": "episode locked"}
            elif not valid:
                self.invalid_requests += 1
                result = {"error": "invalid answer format"}
            else:
                self.locked = True
                self.answer = value
                self.correct = value == self.label
                receipt = ("CORRECT" if self.correct else "INCORRECT") if self.arm == "A2" else "submission accepted"
                result = receipt
                audit.update({"accepted": True, "receipt": receipt})
        else:
            self.invalid_requests += 1
            result = {"error": "unknown function"}
        audit["result_image_count"] = sum(isinstance(x, dict) and x.get("type") == "image" for x in result) if isinstance(result, list) else 0
        return result, sources, audit

    def counters(self) -> dict[str, Any]:
        return {
            "glimpses_exposed": self.glimpses, "move_requests": self.move_requests,
            "accepted_moves": self.accepted_moves, "effective_moves": self.effective_moves,
            "wall_bumps": self.wall_bumps, "effective_revisits": self.effective_revisits,
            "revisit_rate": self.effective_revisits / self.effective_moves if self.effective_moves else None,
            "wall_bump_rate": self.wall_bumps / self.accepted_moves if self.accepted_moves else None,
            "position_history": self.position_history, "submit_calls": self.submit_calls,
            "submitted_answer": self.answer, "correct": self.correct,
            "invalid_requests": self.invalid_requests, "mounted_source_bytes": self.source_bytes,
        }


def response_pending_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
    steps = response.get("steps", [])
    if not isinstance(steps, list):
        return []
    completed = {str(step.get("call_id")) for step in steps if isinstance(step, dict) and step.get("type") == "function_result"}
    return [
        step for step in steps
        if isinstance(step, dict) and step.get("type") == "function_call" and str(step.get("id")) not in completed
        and step.get("name") in {"activeglimpse_move", "activeglimpse_view_image", "activeglimpse_submit"}
    ]


def verify_snapshot(tar_bytes: bytes, evaluator: Evaluator, notes_enabled: bool) -> tuple[bytes, dict[str, Any]]:
    expected = {row["remote_path"].lstrip("/"): row["png_sha256"] for row in evaluator.records}
    found: dict[str, str] = {}
    notes = b""
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:*") as archive:
        for member in archive.getmembers():
            normalized = member.name.lstrip("./")
            if not member.isfile() or member.size > 20_000_000:
                continue
            if normalized in expected or normalized.endswith("/NOTES.md") or normalized == "workspace/NOTES.md":
                handle = archive.extractfile(member)
                if handle is None:
                    continue
                data = handle.read()
                if normalized in expected:
                    found[normalized] = sha256_bytes(data)
                if normalized.endswith("/NOTES.md") or normalized == "workspace/NOTES.md":
                    notes = data
    matches = len(found) == len(expected) and all(found.get(path) == digest for path, digest in expected.items())
    if not matches:
        raise RuntimeError(f"remote mounted-image verification failed ({len(found)}/{len(expected)})")
    if notes_enabled and len(notes) > MAX_INLINE_FILE_BYTES:
        raise RuntimeError("NOTES.md exceeds API inline-source transport limit")
    return notes, {"expected_images": len(expected), "found_images": len(found), "all_hashes_match": matches}


def initial_body(prompt: str, initial: bytes, initial_source: dict[str, Any], notes: bytes | None) -> dict[str, Any]:
    sources = [initial_source]
    if notes is not None:
        sources.append({"type": "inline", "target": "/workspace/NOTES.md", "content": notes.decode("utf-8", errors="replace")})
    return {
        "agent": AGENT,
        "agent_config": {"type": "antigravity", "model": MODEL, "max_total_tokens": MAX_TOTAL_TOKENS},
        "input": [
            {"type": "text", "text": prompt},
            {"type": "image", "mime_type": "image/png", "data": base64.b64encode(initial).decode("ascii")},
        ],
        "environment": {"type": "remote", "network": "disabled", "sources": sources},
        "tools": tool_definitions(1 if "Level 1" in prompt else 2),
        "store": True,
    }


def run_episode(api_key: str, run_root: Path, arm: str, spec: EpisodeSpec, execution_position: int, module: Any, cache: dict[int, tuple[Image.Image, int]], timeout_seconds: int, incoming_notes: bytes) -> tuple[dict[str, Any], bytes]:
    trace_dir = run_root / "traces" / f"slot_{execution_position:03d}"
    trace_dir.mkdir(parents=True, exist_ok=False)
    env, hidden_label, initial_image = make_environment(module, spec, cache)
    evaluator = Evaluator(env, hidden_label, spec.level, arm)
    if arm != "A0":
        if len(incoming_notes) > MAX_INLINE_FILE_BYTES:
            raise RuntimeError("incoming NOTES.md exceeds API inline-source transport limit")
        evaluator.source_bytes = len(incoming_notes)
    initial_data = png_bytes(initial_image)
    initial_record, initial_source = evaluator.add_image(initial_data)
    (trace_dir / "initial.png").write_bytes(initial_data)
    api = API(api_key, trace_dir, timeout_seconds)
    interaction_ids: list[str] = []
    environment_id: str | None = None
    tool_log = trace_dir / "tool_calls.jsonl"
    turn_summaries: list[dict[str, Any]] = []
    deadline = time.monotonic() + timeout_seconds
    try:
        body = initial_body(prompt_for(spec.level, arm), initial_data, initial_source, incoming_notes if arm != "A0" else None)
        turn = 1
        while turn <= BUDGETS[spec.level] * 3 + 20:
            if time.monotonic() >= deadline:
                raise RuntimeError("episode timeout")
            response = api.interaction(body, turn)
            interaction_id = response.get("id")
            if not isinstance(interaction_id, str) or not interaction_id:
                raise RuntimeError("interaction response omitted id")
            interaction_ids.append(interaction_id)
            current_env = response.get("environment_id")
            if environment_id is None:
                if not isinstance(current_env, str) or not current_env:
                    raise RuntimeError("first interaction did not provision an environment")
                environment_id = current_env
            elif current_env not in {None, environment_id}:
                raise RuntimeError("remote environment changed within episode")
            pending = response_pending_calls(response)
            turn_summaries.append({
                "turn": turn, "interaction_id": interaction_id,
                "previous_interaction_id": body.get("previous_interaction_id"),
                "status": response.get("status"), "pending_custom_calls": len(pending),
                "request_native_image_nodes": len(native_image_nodes(body.get("input", []))),
                "request_native_image_sha256": [
                    sha256_bytes(base64.b64decode(node["data"], validate=True))
                    for node in native_image_nodes(body.get("input", []))
                ],
            })
            if not pending:
                if response.get("status") == "completed":
                    break
                raise RuntimeError(f"interaction status {response.get('status')!r} without pending custom function")
            function_results: list[dict[str, Any]] = []
            new_sources: list[dict[str, Any]] = []
            for call in pending:
                call_id = call.get("id")
                name = call.get("name")
                if not isinstance(call_id, str) or not isinstance(name, str):
                    raise RuntimeError("invalid custom function-call step")
                result, sources, audit = evaluator.execute(name, call.get("arguments", {}))
                new_sources.extend(sources)
                function_results.append({"type": "function_result", "name": name, "call_id": call_id, "result": result})
                append_jsonl(tool_log, {"turn": turn, "call_id": call_id, **audit})
            environment: dict[str, Any] = {"type": "remote", "environment_id": environment_id, "network": "disabled"}
            if new_sources:
                environment["sources"] = new_sources
            body = {
                "agent": AGENT,
                "agent_config": {"type": "antigravity", "model": MODEL, "max_total_tokens": MAX_TOTAL_TOKENS},
                "previous_interaction_id": interaction_id,
                "environment": environment,
                "input": function_results,
                "store": True,
            }
            turn += 1
        else:
            raise RuntimeError("tool-turn safety limit reached")
        if evaluator.submit_calls != 1 or evaluator.answer is None:
            raise RuntimeError("episode ended without exactly one valid submission")
        if environment_id is None:
            raise RuntimeError("no environment was provisioned")
        tar_bytes = api.download_environment(environment_id)
        outgoing_notes, mount_audit = verify_snapshot(tar_bytes, evaluator, arm != "A0")
        if arm == "A0":
            outgoing_notes = b""
        deleted = api.delete_environment(environment_id)
        if not deleted:
            raise RuntimeError("remote environment deletion failed")
        counters = evaluator.counters()
        report = {
            "execution_position": execution_position, "episode_id": spec.episode_id,
            "level": spec.level, "arm": arm, "validity": "VALID",
            "answer": evaluator.answer, "ground_truth_label": hidden_label,
            "correctness": evaluator.correct, "counters": counters,
            "agent": AGENT, "underlying_model": MODEL,
            "fresh_interaction": True, "fresh_remote_environment": True,
            "interaction_ids": interaction_ids, "environment_id": environment_id,
            "request_chain": turn_summaries, "images": evaluator.records,
            "remote_mount_audit": mount_audit,
            "native_image_delivery": {
                "initial_attached_and_mounted_same_sha256": initial_record["png_sha256"],
                "move_results_use_multimodal_function_result": True,
            },
            "persistent_notes": {
                "enabled": arm != "A0", "incoming_sha256": sha256_bytes(incoming_notes) if arm != "A0" else None,
                "outgoing_sha256": sha256_bytes(outgoing_notes) if arm != "A0" else None,
                "byte_count": len(outgoing_notes) if arm != "A0" else 0,
            },
            "api_calls": {"model": api.model_calls, "auxiliary": api.auxiliary_calls},
        }
        write_json(trace_dir / "model_call_index.json", turn_summaries)
        write_json(trace_dir / "image_manifest.json", evaluator.records)
        return report, outgoing_notes
    finally:
        if environment_id is not None and not (trace_dir / "environment_delete.json").exists():
            try:
                api.delete_environment(environment_id)
            except Exception:
                pass
        api.close()


def offline_contract(run_root: Path, arm: str, level: int, suite_id: str, specs: list[EpisodeSpec], source_hash: str) -> dict[str, Any]:
    schedule = schedule_payload(specs)
    controller_hash = sha256_file(Path(__file__).resolve())
    result = {
        "passed": True, "arm": arm, "level": level, "suite_id": suite_id,
        "controller_sha256": controller_hash, "source_manifest_sha256": source_hash,
        "schedule_sha256": sha256_bytes(canonical(schedule)),
        "notes_enabled": arm in {"A1", "A2"}, "feedback_enabled": arm == "A2",
        "notes_size_cap_enforced": False, "expected_episode_count": EXPECTED_EPISODES,
        "scored_api_calls": 0, "agent": AGENT, "underlying_model": MODEL,
        "remote_environment_network": "disabled", "search_and_url_tools_exposed": False,
        "native_image_delivery": "base64-mounted-source-plus-multimodal-content",
    }
    write_json(run_root / "schedule_manifest.json", {"configuration": result, "execution_schedule": schedule})
    write_json(run_root / "preflight/offline_contract.json", result)
    return result


def wait_for_barrier(path: Path, suite_id: str, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                time.sleep(0.1)
                continue
            if value.get("suite_id") not in {None, suite_id}:
                raise RuntimeError("barrier suite id mismatch")
            return
        time.sleep(0.1)
    raise RuntimeError("suite barrier timed out")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=("A0", "A1", "A2"))
    parser.add_argument("--level", required=True, type=int, choices=(1, 2))
    parser.add_argument("--suite-id", required=True)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--credential-source", required=True)
    parser.add_argument("--barrier-file", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--episode-timeout-seconds", type=int, default=1800)
    parser.add_argument("--barrier-timeout-seconds", type=int, default=900)
    args = parser.parse_args(argv)
    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    failure = run_root / "CONTROLLER_FAILURE.txt"
    try:
        specs, source_hash = load_specs(args.level)
        load_environment_module()
        contract = offline_contract(run_root, args.arm, args.level, args.suite_id, specs, source_hash)
        if args.preflight_only:
            return 0
        if args.barrier_file is None:
            raise RuntimeError("--barrier-file is required for scored mode")
        ready = {**contract, "ready": True, "controller_pid": os.getpid()}
        write_json(run_root / "BARRIER_READY.json", ready)
        wait_for_barrier(args.barrier_file, args.suite_id, args.barrier_timeout_seconds)
        api_key = resolve_credential(args.credential_source)
        module = load_environment_module()
        cache = preload(specs)
        notes_state = b""
        reports: list[dict[str, Any]] = []
        for position, spec in enumerate(specs, 1):
            append_jsonl(run_root / "progress.jsonl", {"event": "episode_start", "execution_position": position, "episode_id": spec.episode_id})
            report, notes_state = run_episode(api_key, run_root, args.arm, spec, position, module, cache, args.episode_timeout_seconds, notes_state)
            reports.append(report)
            write_json(run_root / "attempts" / f"slot_{position:03d}.json", report)
            if args.arm != "A0":
                history = run_root / "notes_history" / f"ep_{position:03d}.md"
                history.parent.mkdir(parents=True, exist_ok=True)
                history.write_bytes(notes_state)
                write_json(run_root / "leakage_audits" / f"ep_{position:03d}.json", {
                    "execution_position": position, "non_gating": True,
                    "notes_sha256": sha256_bytes(notes_state), "line_count": len(notes_state.decode("utf-8", errors="replace").splitlines()),
                    "answer_related_pattern_hits": len(re.findall(r"(?i)\b(?:digit|answer|label|correct|incorrect|[0-9])\b", notes_state.decode("utf-8", errors="replace"))),
                })
            append_jsonl(run_root / "progress.jsonl", {"event": "episode_complete", "execution_position": position, "episode_id": spec.episode_id, "correctness": report["correctness"]})
        correct = sum(row["correctness"] is True for row in reports)
        summary = {
            "arm": args.arm, "level": args.level, "suite_id": args.suite_id,
            "attempted": len(reports), "valid": len(reports), "correct_valid": correct,
            "accuracy_valid": correct / len(reports) if reports else None,
            "schedule_complete": len(reports) == EXPECTED_EPISODES,
            "agent": AGENT, "underlying_model": MODEL,
            "model_api_calls": sum(row["api_calls"]["model"] for row in reports),
            "move_calls": sum(row["counters"]["move_requests"] for row in reports),
            "accepted_moves": sum(row["counters"]["accepted_moves"] for row in reports),
            "viewer_calls": sum(sum(1 for line in (run_root / "traces" / f"slot_{i:03d}" / "tool_calls.jsonl").read_text().splitlines() if 'activeglimpse_view_image' in line) for i in range(1, len(reports) + 1)),
            "all_remote_mount_hashes_match": all(row["remote_mount_audit"]["all_hashes_match"] for row in reports),
            "notes_snapshots": len(list((run_root / "notes_history").glob("ep_*.md"))) if args.arm != "A0" else 0,
            "completed_at": utc_now(),
        }
        write_json(run_root / "summary.json", summary)
        with (run_root / "results.jsonl").open("wb") as handle:
            for row in reports:
                handle.write(canonical(row) + b"\n")
        with (run_root / "results.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("execution_position", "episode_id", "answer", "ground_truth_label", "correctness", "glimpses_exposed", "accepted_moves"))
            writer.writeheader()
            for row in reports:
                writer.writerow({"execution_position": row["execution_position"], "episode_id": row["episode_id"], "answer": row["answer"], "ground_truth_label": row["ground_truth_label"], "correctness": row["correctness"], "glimpses_exposed": row["counters"]["glimpses_exposed"], "accepted_moves": row["counters"]["accepted_moves"]})
        write_json(run_root / "ARM_COMPLETE.json", summary)
        return 0
    except Exception:
        failure.write_text(traceback.format_exc(), encoding="utf-8")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
