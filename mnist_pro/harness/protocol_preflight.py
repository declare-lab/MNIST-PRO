#!/opt/anaconda3/bin/python
"""One-episode Antigravity image/file/tool protocol preflight.

The default path makes one paid managed-agent preflight.  ``--dry-run`` uses a
deterministic in-memory API and never resolves credentials or opens a socket.
Only hashes, sizes, tool names, and sanitized response steps are persisted.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import stat
import subprocess
import tarfile
from typing import Any, Protocol
from urllib.parse import quote

from PIL import Image


AGENT = "antigravity-preview-05-2026"
MODEL = "gemini-3.7-flash"
AGENT_CONFIG = {"type": "antigravity", "model": MODEL, "max_total_tokens": 16_000}
API_REVISION = "2026-05-20"
BASE_URL = "https://generativelanguage.googleapis.com"
FIRST_PATH = "/workspace/k4n8q.png"
SECOND_PATH = "/workspace/v7c2m.png"
COMPOSITE_PATH = "/workspace/composite.png"
FORBIDDEN_STEP_TYPES = {
    "google_search_call",
    "google_search_result",
    "url_context_call",
    "url_context_result",
}
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_EXTRACTED_BYTES = 128 * 1024 * 1024


class PreflightError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def make_png(which: int) -> bytes:
    image = Image.new("RGBA", (24, 24), (255, 255, 255, 255))
    pixels = image.load()
    assert pixels is not None
    for y in range(24):
        for x in range(24):
            if which == 1 and (x < 5 or abs(x - y) <= 1):
                pixels[x, y] = (190, 24, 64, 255)
            if which == 2 and (y > 17 or abs((23 - x) - y) <= 1):
                pixels[x, y] = (22, 105, 180, 255)
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)
    return output.getvalue()


def expected_composite(first: bytes, second: bytes) -> bytes:
    with Image.open(BytesIO(first)) as left, Image.open(BytesIO(second)) as right:
        left_rgba, right_rgba = left.convert("RGBA"), right.convert("RGBA")
        if left_rgba.height != right_rgba.height:
            raise PreflightError("synthetic image heights differ")
        output = Image.new(
            "RGBA", (left_rgba.width + right_rgba.width, left_rgba.height)
        )
        output.paste(left_rgba, (0, 0))
        output.paste(right_rgba, (left_rgba.width, 0))
    encoded = BytesIO()
    output.save(encoded, format="PNG", optimize=False)
    return encoded.getvalue()


def archive_key(name: str) -> str:
    if "\x00" in name or "\\" in name:
        raise PreflightError("unsafe archive member name")
    raw = PurePosixPath(name)
    if raw.is_absolute():
        raise PreflightError("absolute archive member")
    parts = [part for part in raw.parts if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        raise PreflightError("traversing archive member")
    return "/".join(parts)


def safe_tar_files(blob: bytes) -> dict[str, bytes]:
    """Read regular files without extracting and reject link/device/path attacks."""
    if len(blob) > MAX_ARCHIVE_BYTES:
        raise PreflightError("environment archive is too large")
    files: dict[str, bytes] = {}
    total = 0
    try:
        archive = tarfile.open(fileobj=BytesIO(blob), mode="r:*")
    except tarfile.TarError as error:
        raise PreflightError("invalid environment archive") from error
    with archive:
        for member in archive.getmembers():
            key = archive_key(member.name)
            if member.isdir():
                continue
            if not member.isfile():
                raise PreflightError(f"non-regular archive member: {key}")
            if key in files or member.size < 0:
                raise PreflightError(f"invalid duplicate archive member: {key}")
            total += member.size
            if total > MAX_EXTRACTED_BYTES:
                raise PreflightError("environment archive expands too large")
            handle = archive.extractfile(member)
            if handle is None:
                raise PreflightError(f"unreadable archive member: {key}")
            content = handle.read(member.size + 1)
            if len(content) != member.size:
                raise PreflightError(f"truncated archive member: {key}")
            files[key] = content
    return files


def workspace_key(path: str) -> str:
    pure = PurePosixPath(path)
    if pure.is_absolute():
        if not pure.parts or pure.parts[1:2] != ("workspace",):
            raise PreflightError("viewer path is outside /workspace")
        parts = pure.parts[1:]
    else:
        parts = ("workspace",) + pure.parts
    if any(part in ("", ".", "..") for part in parts):
        raise PreflightError("unsafe viewer path")
    return "/".join(parts)


def find_workspace_file(files: dict[str, bytes], path: str) -> bytes:
    wanted = workspace_key(path)
    candidates = (wanted, f"./{wanted}", wanted.removeprefix("workspace/"))
    for candidate in candidates:
        if candidate in files:
            return files[candidate]
    suffix_matches = [v for k, v in files.items() if k.endswith(f"/{wanted}")]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    raise PreflightError(f"workspace snapshot lacks {path}")


def validate_composite(first: bytes, second: bytes, candidate: bytes) -> None:
    with Image.open(BytesIO(expected_composite(first, second))) as expected:
        expected_rgba = expected.convert("RGBA")
    try:
        with Image.open(BytesIO(candidate)) as actual:
            actual_rgba = actual.convert("RGBA")
    except Exception as error:
        raise PreflightError("composite artifact is not a decodable image") from error
    if actual_rgba.size != expected_rgba.size or actual_rgba.tobytes() != expected_rgba.tobytes():
        raise PreflightError("composite artifact pixels are not the exact horizontal fusion")


def image_block(value: bytes) -> dict[str, Any]:
    return {
        "type": "image",
        "mime_type": "image/png",
        "data": base64.b64encode(value).decode("ascii"),
    }


def inline_source(value: bytes, target: str) -> dict[str, Any]:
    return {
        "type": "inline",
        "encoding": "base64",
        "content": base64.b64encode(value).decode("ascii"),
        "target": target,
    }


TOOLS: list[dict[str, Any]] = [
    {"type": "code_execution"},
    {
        "type": "function",
        "name": "move",
        "description": "Reveal the second synthetic observation.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "type": "function",
        "name": "view_image",
        "description": "View a PNG file currently stored under /workspace.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "submit",
        "description": "Submit verification hashes and the fused image path.",
        "parameters": {
            "type": "object",
            "properties": {
                "first_sha256": {"type": "string"},
                "second_sha256": {"type": "string"},
                "composite_path": {"type": "string"},
            },
            "required": ["first_sha256", "second_sha256", "composite_path"],
            "additionalProperties": False,
        },
    },
]


def prompt(first_hash: str, second_hash: str) -> str:
    return f"""Protocol preflight. You have one initial PNG at {FIRST_PATH}, also attached
as image content. Compute its SHA-256 and verify it equals {first_hash}. Call move exactly
once to receive a second observation. Verify that mounted PNG's SHA-256 equals
{second_hash}. Use your code-execution shell with Pillow or ImageMagick to concatenate the
two PNGs horizontally, first then second, without resizing, and save {COMPOSITE_PATH}.
Explicitly call view_image on {COMPOSITE_PATH} so you reopen the persisted fusion. Then
call submit with both observed hashes and that composite path. Do not use the network."""


def summarize_blob(value: str) -> dict[str, Any]:
    try:
        raw = base64.b64decode(value, validate=True)
    except Exception:
        raw = value.encode("utf-8")
    return {"redacted_inline_bytes": len(raw), "sha256": sha256_bytes(raw)}


def sanitize(value: Any, *, key_name: str | None = None) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if key == "data" and isinstance(item, str):
                result[key] = summarize_blob(item)
            elif key == "content" and value.get("encoding") == "base64" and isinstance(item, str):
                result[key] = summarize_blob(item)
            else:
                result[key] = sanitize(item, key_name=key)
        return result
    if isinstance(value, list):
        return [sanitize(item, key_name=key_name) for item in value]
    return value


class Api(Protocol):
    requests: list[dict[str, Any]]

    def create_interaction(self, payload: dict[str, Any]) -> dict[str, Any]: ...
    def download_environment(self, environment_id: str) -> bytes: ...
    def delete_environment(self, environment_id: str) -> None: ...


def keychain_api_key() -> str:
    try:
        result = subprocess.run(
            [
                "/usr/bin/security",
                "find-generic-password",
                "-s",
                "gemini",
                "-a",
                "antigravity",
                "-w",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=15,
            check=False,
        )
    except Exception as error:
        raise PreflightError("could not query the configured Keychain item") from error
    key = result.stdout.strip()
    if result.returncode != 0 or not key or "\n" in key or "\r" in key:
        raise PreflightError("Keychain service gemini/account antigravity is unavailable")
    return key


def credential_value(source: str) -> str:
    """Resolve an API key without persisting it or placing it in argv."""
    if source == "keychain":
        return keychain_api_key()
    if source.startswith("env:"):
        name = source[4:]
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise PreflightError("unsafe credential environment-variable name")
        value = os.environ.get(name, "").strip()
        if not value or "\n" in value or "\r" in value:
            raise PreflightError(f"credential environment variable is unavailable: {name}")
        return value
    if source.startswith("file:"):
        path = Path(source[5:])
        if not path.is_absolute():
            raise PreflightError("credential file path must be absolute")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as error:
            raise PreflightError("credential file could not be opened securely") from error
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.getuid()
                or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or not 1 <= metadata.st_size <= 65_536
            ):
                raise PreflightError("credential file failed owner-only structural checks")
            raw = os.read(descriptor, metadata.st_size + 1)
        finally:
            os.close(descriptor)
        value = raw.decode("utf-8", errors="strict").strip()
        if not value or "\n" in value or "\r" in value:
            raise PreflightError("credential file does not contain one API key")
        return value
    raise PreflightError("credential source must be keychain, env:NAME, or file:/absolute/path")


class HttpApi:
    def __init__(self, api_key: str, timeout_seconds: float = 240.0) -> None:
        import httpx

        self._key = api_key
        self._httpx = httpx
        self._client = httpx.Client(
            base_url=BASE_URL,
            headers={"x-goog-api-key": api_key, "Api-Revision": API_REVISION},
            timeout=timeout_seconds,
        )
        self.requests: list[dict[str, Any]] = []

    def _check(self, response: Any, operation: str) -> Any:
        if 200 <= response.status_code < 300:
            return response
        # Never include request headers or credential material in exceptions.
        body = response.text[:2000].replace(self._key, "[REDACTED]")
        raise PreflightError(f"{operation} failed with HTTP {response.status_code}: {body}")

    def create_interaction(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(payload)
        response = self._check(
            self._client.post("/v1beta/interactions", json=payload), "interaction"
        )
        value = response.json()
        if not isinstance(value, dict):
            raise PreflightError("interaction returned non-object JSON")
        return value

    @staticmethod
    def _resource(environment_id: str) -> str:
        name = environment_id.removeprefix("files/")
        if not name.startswith("environment-"):
            name = f"environment-{name}"
        if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
            raise PreflightError("unsafe environment identifier")
        return name

    def download_environment(self, environment_id: str) -> bytes:
        resource = quote(self._resource(environment_id), safe="")
        response = self._check(
            self._client.get(f"/v1beta/files/{resource}:download", params={"alt": "media"}),
            "environment download",
        )
        return bytes(response.content)

    def delete_environment(self, environment_id: str) -> None:
        resource = quote(self._resource(environment_id), safe="")
        self._check(self._client.delete(f"/v1beta/files/{resource}"), "environment delete")
        self._client.close()


def tar_from_files(files: dict[str, bytes]) -> bytes:
    output = BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name, content in sorted(files.items()):
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            info.mode = 0o600
            archive.addfile(info, BytesIO(content))
    return output.getvalue()


class DryRunApi:
    """Deterministic protocol simulator used by --dry-run and unit tests."""

    def __init__(self, first: bytes, second: bytes) -> None:
        self.first, self.second = first, second
        self.files = {workspace_key(FIRST_PATH): first}
        self.requests: list[dict[str, Any]] = []
        self.deleted = False
        self.turn = 0

    def _mount(self, payload: dict[str, Any]) -> None:
        environment = payload.get("environment")
        if not isinstance(environment, dict):
            return
        for source in environment.get("sources", []):
            if source.get("type") == "inline" and source.get("encoding") == "base64":
                self.files[workspace_key(source["target"])] = base64.b64decode(source["content"])

    def create_interaction(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(payload)
        self._mount(payload)
        self.turn += 1
        base = {
            "id": f"dry-interaction-{self.turn}",
            "environment_id": "dry-env-1",
            "agent": AGENT,
            "status": "requires_action",
        }
        if self.turn == 1:
            base["steps"] = [
                {"type": "code_execution_call", "arguments": {"code": "sha256sum /workspace/k4n8q.png"}},
                {"type": "code_execution_result", "result": "ok"},
                {"type": "function_call", "id": "call-move", "name": "move", "arguments": {}},
            ]
        elif self.turn == 2:
            self.files[workspace_key(COMPOSITE_PATH)] = expected_composite(self.first, self.second)
            base["steps"] = [
                {"type": "function_call", "id": "call-view", "name": "view_image", "arguments": {"path": COMPOSITE_PATH}}
            ]
        elif self.turn == 3:
            base["steps"] = [
                {
                    "type": "function_call",
                    "id": "call-submit",
                    "name": "submit",
                    "arguments": {
                        "first_sha256": sha256_bytes(self.first),
                        "second_sha256": sha256_bytes(self.second),
                        "composite_path": COMPOSITE_PATH,
                    },
                }
            ]
        else:
            base["status"] = "completed"
            base["steps"] = [{"type": "model_output", "content": [{"type": "text", "text": "done"}]}]
        return base

    def download_environment(self, environment_id: str) -> bytes:
        if environment_id != "dry-env-1":
            raise PreflightError("wrong dry-run environment")
        return tar_from_files(self.files)

    def delete_environment(self, environment_id: str) -> None:
        if environment_id != "dry-env-1":
            raise PreflightError("wrong dry-run environment")
        self.deleted = True


@dataclass
class ProtocolPreflight:
    api: Api
    output_dir: Path
    dry_run: bool = False

    def _audit(self, event: dict[str, Any]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with (self.output_dir / "audit.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(sanitize({"at": utc_now(), **event}), sort_keys=True) + "\n")

    def run(self) -> dict[str, Any]:
        first, second = make_png(1), make_png(2)
        first_hash, second_hash = sha256_bytes(first), sha256_bytes(second)
        initial = {
            "agent": AGENT,
            "agent_config": dict(AGENT_CONFIG),
            "store": True,
            "environment": {
                "type": "remote",
                "network": "disabled",
                "sources": [inline_source(first, FIRST_PATH)],
            },
            "tools": TOOLS,
            "input": [
                {"type": "text", "text": prompt(first_hash, second_hash)},
                image_block(first),
            ],
        }
        environment_id: str | None = None
        handled: set[str] = set()
        observed_steps: list[str] = []
        calls = {"move": 0, "view_image": 0, "submit": 0}
        viewed_composite = False
        submitted = False
        cleanup_error: str | None = None
        try:
            payload = initial
            for turn in range(1, 17):
                self._audit({"event": "request", "turn": turn, "payload": payload})
                response = self.api.create_interaction(payload)
                self._audit({"event": "response", "turn": turn, "response": response})
                response_steps = response.get("steps") or []
                if not isinstance(response_steps, list):
                    raise PreflightError("interaction steps are not a list")
                candidate_environment_id = response.get("environment_id")
                if isinstance(candidate_environment_id, str) and candidate_environment_id:
                    environment_id = candidate_environment_id
                step_types = [step.get("type") for step in response_steps if isinstance(step, dict)]
                observed_steps.extend(str(item) for item in step_types)
                forbidden = FORBIDDEN_STEP_TYPES.intersection(step_types)
                if forbidden:
                    raise PreflightError(f"forbidden search/URL step observed: {sorted(forbidden)}")
                current_id = response.get("id")
                if not isinstance(current_id, str) or not current_id:
                    raise PreflightError("interaction has no id")
                if not isinstance(environment_id, str) or not environment_id:
                    raise PreflightError("interaction has no environment id")
                status = response.get("status")
                if status == "completed":
                    break
                if status != "requires_action":
                    raise PreflightError(f"unexpected interaction status: {status}")
                results: list[dict[str, Any]] = []
                sources: list[dict[str, Any]] = []
                for step in response_steps:
                    if not isinstance(step, dict) or step.get("type") != "function_call":
                        continue
                    call_id, name = step.get("id"), step.get("name")
                    args = step.get("arguments") or {}
                    if not isinstance(call_id, str) or call_id in handled:
                        continue
                    handled.add(call_id)
                    if name not in calls:
                        raise PreflightError(f"unknown function call: {name}")
                    calls[name] += 1
                    if name == "move":
                        if calls[name] != 1:
                            result: Any = "move may be called only once"
                            is_error = True
                        else:
                            sources.append(inline_source(second, SECOND_PATH))
                            result = [
                                {"type": "text", "text": f"revealed and mounted at {SECOND_PATH}"},
                                image_block(second),
                            ]
                            is_error = False
                    elif name == "view_image":
                        try:
                            requested = str(args["path"])
                            snapshot = safe_tar_files(self.api.download_environment(environment_id))
                            viewed = find_workspace_file(snapshot, requested)
                            with Image.open(BytesIO(viewed)) as decoded:
                                decoded.verify()
                            viewed_composite |= workspace_key(requested) == workspace_key(COMPOSITE_PATH)
                            result = [
                                {"type": "text", "text": f"viewed {requested}"},
                                image_block(viewed),
                            ]
                            is_error = False
                        except Exception:
                            result, is_error = "image could not be viewed", True
                    else:
                        try:
                            if args.get("first_sha256") != first_hash or args.get("second_sha256") != second_hash:
                                raise PreflightError("submitted source hashes differ")
                            path = str(args.get("composite_path", ""))
                            snapshot = safe_tar_files(self.api.download_environment(environment_id))
                            validate_composite(first, second, find_workspace_file(snapshot, path))
                            submitted = True
                            result, is_error = "submission accepted", False
                        except Exception:
                            result, is_error = "submission validation failed", True
                    results.append(
                        {
                            "type": "function_result",
                            "call_id": call_id,
                            "name": name,
                            "result": result,
                            "is_error": is_error,
                        }
                    )
                if not results:
                    raise PreflightError("requires_action response has no unhandled function call")
                environment: dict[str, Any] = {
                    "type": "remote",
                    "environment_id": environment_id,
                    "network": "disabled",
                }
                if sources:
                    environment["sources"] = sources
                payload = {
                    "agent": AGENT,
                    "agent_config": dict(AGENT_CONFIG),
                    "store": True,
                    "previous_interaction_id": current_id,
                    "environment": environment,
                    "tools": TOOLS,
                    "input": results,
                }
            else:
                raise PreflightError("interaction exceeded preflight turn limit")

            if calls["move"] < 1 or calls["view_image"] < 1 or calls["submit"] < 1:
                raise PreflightError(f"unexpected function counts: {calls}")
            if not viewed_composite or not submitted:
                raise PreflightError("composite was not both viewed and submitted")
            snapshot_blob = self.api.download_environment(environment_id)
            snapshot = safe_tar_files(snapshot_blob)
            first_saved = find_workspace_file(snapshot, FIRST_PATH)
            second_saved = find_workspace_file(snapshot, SECOND_PATH)
            composite_saved = find_workspace_file(snapshot, COMPOSITE_PATH)
            if sha256_bytes(first_saved) != first_hash or sha256_bytes(second_saved) != second_hash:
                raise PreflightError("mounted PNG bytes changed in the environment")
            validate_composite(first, second, composite_saved)
            (self.output_dir / "environment_snapshot.tar.gz").write_bytes(snapshot_blob)
            result = {
                "passed": True,
                "dry_run": self.dry_run,
                "agent": AGENT,
                "model": MODEL,
                "agent_config": AGENT_CONFIG,
                "network": "disabled",
                "first_sha256": first_hash,
                "second_sha256": second_hash,
                "composite_sha256": sha256_bytes(composite_saved),
                "function_calls": calls,
                "observed_step_types": observed_steps,
                "search_or_url_steps": 0,
                "interaction_requests": len(self.api.requests),
                "same_environment_continuations": all(
                    request.get("environment", {}).get("environment_id") == environment_id
                    for request in self.api.requests[1:]
                ),
            }
        finally:
            if environment_id is not None:
                try:
                    self.api.delete_environment(environment_id)
                    self._audit({"event": "environment_deleted", "environment_id": environment_id})
                except Exception as error:
                    cleanup_error = f"{type(error).__name__}: {error}"
                    self._audit({"event": "environment_delete_failed", "error": cleanup_error})
        if cleanup_error is not None:
            raise PreflightError(f"environment cleanup failed: {cleanup_error}")
        result["environment_deleted"] = True
        atomic_json(self.output_dir / "PREFLIGHT_COMPLETE.json", result)
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--credential-source",
        default="keychain",
        help="keychain, env:NAME, or file:/absolute/owner-only/path (ignored by --dry-run)",
    )
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        first, second = make_png(1), make_png(2)
        api: Api = DryRunApi(first, second) if args.dry_run else HttpApi(
            credential_value(args.credential_source), timeout_seconds=args.timeout_seconds
        )
        result = ProtocolPreflight(api, args.output_dir, dry_run=args.dry_run).run()
        print(json.dumps({"passed": result["passed"], "dry_run": args.dry_run}))
        return 0
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        atomic_json(args.output_dir / "PREFLIGHT_FAILURE.json", {"passed": False, "error": message})
        print(message)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
