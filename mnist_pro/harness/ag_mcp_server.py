#!/opt/anaconda3/bin/python
"""Minimal local MCP bridge for guaranteed ActiveGlimpse image delivery.

The server has no label, schedule, or coordinate access. It relays opaque
move/submit requests and controller-provided submit receipts without deriving
correctness itself. A successful move is returned to Codex as native MCP
ImageContent; an opaque absolute pathname is also retained as an auxiliary
local-file affordance.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from pathlib import Path
import stat
import sys
import time
import uuid
from typing import Any


POLL_SECONDS = 0.025
TIMEOUT_SECONDS = 120.0
MAX_PNG_BYTES = 4_000_000
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def write_message(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message, separators=(",", ":"), ensure_ascii=True))
    sys.stdout.write("\n")
    sys.stdout.flush()


def rpc_result(request_id: Any, result: dict[str, Any]) -> None:
    write_message({"jsonrpc": "2.0", "id": request_id, "result": result})


def rpc_error(request_id: Any, code: int, message: str) -> None:
    write_message(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }
    )


def mailbox_exchange(operation: str, **arguments: Any) -> tuple[str, dict[str, Any]]:
    mailbox_text = os.environ.get("AG_MCP_MAILBOX", "")
    token = os.environ.get("AG_MCP_TOKEN", "")
    if not mailbox_text or not token:
        raise RuntimeError("episode interface unavailable")
    mailbox = Path(mailbox_text)
    requests = mailbox / "requests"
    responses = mailbox / "responses"
    request_id = uuid.uuid4().hex
    payload = {
        "protocol": 1,
        "request_id": request_id,
        "operation": operation,
        **arguments,
    }
    authenticated_bytes = json.dumps(
        payload,
        separators=(",", ":"),
        sort_keys=True,
        ensure_ascii=True,
    ).encode("utf-8")
    payload["auth"] = hmac.new(
        token.encode("utf-8"), authenticated_bytes, hashlib.sha256
    ).hexdigest()
    request_path = requests / f"{request_id}.json"
    temporary_path = requests / f".{request_id}.{os.getpid()}.tmp"
    response_path = responses / f"{request_id}.json"
    try:
        with temporary_path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"), ensure_ascii=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, request_path)
    except OSError as error:
        raise RuntimeError("episode interface unavailable") from error

    deadline = time.monotonic() + TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            with response_path.open("r", encoding="utf-8") as handle:
                response = json.load(handle)
        except FileNotFoundError:
            time.sleep(POLL_SECONDS)
            continue
        except (OSError, json.JSONDecodeError):
            time.sleep(POLL_SECONDS)
            continue
        if not isinstance(response, dict) or response.get("request_id") != request_id:
            raise RuntimeError("episode interface returned an invalid response")
        if response.get("ok") is not True:
            message = response.get("message")
            raise RuntimeError(message if isinstance(message, str) else "request rejected")
        return request_id, response
    raise RuntimeError("episode interface timed out")


def read_controller_png(path_text: str) -> tuple[bytes, str]:
    root_text = os.environ.get("AG_MCP_OBSERVATION_DIR", "")
    if not root_text:
        raise RuntimeError("episode observation store unavailable")
    root = Path(root_text)
    path = Path(path_text)
    if not path.is_absolute():
        raise RuntimeError("controller returned a non-absolute observation path")
    try:
        if os.path.commonpath((str(root), str(path))) != str(root):
            raise RuntimeError("controller returned an out-of-store observation path")
    except ValueError as error:
        raise RuntimeError("controller returned an invalid observation path") from error
    if path.parent != root:
        raise RuntimeError("controller returned a nested observation path")

    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.getuid()
            or before.st_nlink != 1
            or before.st_size <= len(PNG_SIGNATURE)
            or before.st_size > MAX_PNG_BYTES
        ):
            raise RuntimeError("controller observation failed structural validation")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1 << 20))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or remaining != 0
    ):
        raise RuntimeError("controller observation changed during delivery")
    data = b"".join(chunks)
    if not data.startswith(PNG_SIGNATURE):
        raise RuntimeError("controller observation is not a PNG")
    return data, hashlib.sha256(data).hexdigest()


def read_workspace_png(path_text: str) -> tuple[bytes, str]:
    root_text = os.environ.get("AG_MCP_WORKSPACE", "")
    if not root_text:
        raise RuntimeError("episode workspace unavailable")
    root = Path(root_text)
    path = Path(path_text)
    if not path.is_absolute():
        raise RuntimeError("image path must be absolute")
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise RuntimeError("image path is outside the episode workspace") from error
    if not relative.parts or any(
        component in {"", ".", ".."} for component in relative.parts
    ):
        raise RuntimeError("invalid image path components")

    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    root_descriptor = os.open(root, directory_flags)
    parent_descriptor = root_descriptor
    opened_parent_descriptors: list[int] = []
    descriptor: int | None = None
    try:
        for component in relative.parts[:-1]:
            next_descriptor = os.open(
                component,
                directory_flags | os.O_NONBLOCK,
                dir_fd=parent_descriptor,
            )
            opened_parent_descriptors.append(next_descriptor)
            parent_descriptor = next_descriptor
        descriptor = os.open(
            relative.parts[-1],
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=parent_descriptor,
        )
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.getuid()
            or before.st_nlink != 1
            or before.st_size <= len(PNG_SIGNATURE)
            or before.st_size > MAX_PNG_BYTES
        ):
            raise RuntimeError("image failed structural validation")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1 << 20))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        for opened in reversed(opened_parent_descriptors):
            os.close(opened)
        os.close(root_descriptor)
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or remaining != 0
    ):
        raise RuntimeError("image changed while it was being opened")
    data = b"".join(chunks)
    if not data.startswith(PNG_SIGNATURE):
        raise RuntimeError("image is not a PNG")
    return data, hashlib.sha256(data).hexdigest()


def call_move(arguments: Any) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        raise RuntimeError("move arguments must be an object")
    direction = arguments.get("direction")
    if direction not in {"up", "down", "left", "right"}:
        raise RuntimeError("direction must be up, down, left, or right")
    move_request_id, response = mailbox_exchange("move", direction=direction)
    observation = response.get("observation")
    if not isinstance(observation, str) or not observation:
        raise RuntimeError("episode interface omitted the observation")
    png_bytes, digest = read_controller_png(observation)
    mailbox_exchange(
        "delivery_receipt",
        move_request_id=move_request_id,
        sha256=digest,
        byte_count=len(png_bytes),
        delivery_format="mcp_image_content",
    )
    return {
        "content": [
            {"type": "text", "text": observation},
            {
                "type": "image",
                "data": base64.b64encode(png_bytes).decode("ascii"),
                "mimeType": "image/png",
            },
        ],
        "isError": False,
    }


def call_submit(arguments: Any) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        raise RuntimeError("submit arguments must be an object")
    value = arguments.get("value")
    if isinstance(value, int):
        value = str(value)
    if not isinstance(value, str) or not value:
        raise RuntimeError("value must be a nonempty string or integer")
    _, response = mailbox_exchange("submit", value=value)
    message = response.get("message")
    if not isinstance(message, str) or message not in {
        "submission accepted",
        "CORRECT",
        "INCORRECT",
    }:
        raise RuntimeError("episode interface returned an invalid submission receipt")
    return {
        "content": [{"type": "text", "text": message}],
        "isError": False,
    }


def call_view_image(arguments: Any) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        raise RuntimeError("view_image arguments must be an object")
    path = arguments.get("path")
    if not isinstance(path, str) or not path:
        raise RuntimeError("path must be a nonempty absolute PNG path")
    png_bytes, _ = read_workspace_png(path)
    return {
        "content": [
            {"type": "text", "text": path},
            {
                "type": "image",
                "data": base64.b64encode(png_bytes).decode("ascii"),
                "mimeType": "image/png",
            },
        ],
        "isError": False,
    }


TOOLS = [
    {
        "name": "move",
        "description": (
            "Move the viewing window one step and return the resulting masked "
            "canvas as native image content plus an opaque absolute path to the "
            "same PNG."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["up", "down", "left", "right"],
                }
            },
            "required": ["direction"],
            "additionalProperties": False,
        },
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": False,
        },
    },
    {
        "name": "view_image",
        "description": (
            "Open an absolute PNG path inside the current episode workspace "
            "and return that file as native image content."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    },
    {
        "name": "submit",
        "description": "Submit one final digit answer and lock the episode.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "value": {
                    "anyOf": [{"type": "string"}, {"type": "integer"}]
                }
            },
            "required": ["value"],
            "additionalProperties": False,
        },
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": False,
        },
    },
]


def handle(message: Any) -> None:
    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
        return
    request_id = message.get("id")
    method = message.get("method")
    if request_id is None:
        return
    if method == "initialize":
        params = message.get("params")
        version = (
            params.get("protocolVersion")
            if isinstance(params, dict)
            and isinstance(params.get("protocolVersion"), str)
            else "2025-06-18"
        )
        rpc_result(
            request_id,
            {
                "protocolVersion": version,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "activeglimpse", "version": "1.0.0"},
            },
        )
        return
    if method == "ping":
        rpc_result(request_id, {})
        return
    if method == "tools/list":
        rpc_result(request_id, {"tools": TOOLS})
        return
    if method == "tools/call":
        params = message.get("params")
        try:
            if not isinstance(params, dict):
                raise RuntimeError("invalid tool request")
            name = params.get("name")
            arguments = params.get("arguments", {})
            if name == "move":
                result = call_move(arguments)
            elif name == "view_image":
                result = call_view_image(arguments)
            elif name == "submit":
                result = call_submit(arguments)
            else:
                raise RuntimeError("unknown tool")
        except Exception as error:
            result = {
                "content": [{"type": "text", "text": str(error)}],
                "isError": True,
            }
        rpc_result(request_id, result)
        return
    rpc_error(request_id, -32601, "method not found")


def main() -> int:
    for raw_line in sys.stdin.buffer:
        if not raw_line.strip():
            continue
        try:
            message = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        handle(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
