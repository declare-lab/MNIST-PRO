"""Controller side of the MCP episode mailbox.

`ag_mcp_server.py` never touches the environment directly. It writes an
HMAC-authenticated request into a mailbox directory and waits for a response, so the
server process holds no label, no schedule and no coordinates -- an agent cannot read
the answer out of the harness even if it inspects the tool surface.

This module implements the other end: it validates the HMAC, applies the operation to
a real `ActiveGlimpseEnv`, and writes the response. Together with `mcp_client.py` it
makes the MCP harness runnable, and testable, without a model or a network.

Wire format, verbatim from the server:

    request   $AG_MCP_MAILBOX/requests/<id>.json
              {"protocol": 1, "request_id", "operation", ..., "auth": <hmac>}
              auth = HMAC-SHA256(token, canonical_json_without_auth)
              canonical = separators (",",":"), sort_keys, ensure_ascii
    response  $AG_MCP_MAILBOX/responses/<id>.json
              {"request_id", "ok": true, ...}   or   {"request_id", "ok": false, "message"}
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

PROTOCOL_VERSION = 1
POLL_SECONDS = 0.01


def canonical_bytes(payload: dict) -> bytes:
    """The exact serialisation the server signs."""
    body = {k: v for k, v in payload.items() if k != "auth"}
    return json.dumps(body, separators=(",", ":"), sort_keys=True,
                      ensure_ascii=True).encode("utf-8")


def sign(token: str, payload: dict) -> str:
    return hmac.new(token.encode("utf-8"), canonical_bytes(payload),
                    hashlib.sha256).hexdigest()


def verify(token: str, payload: dict) -> bool:
    """Constant-time comparison; an unsigned or mis-signed request is rejected."""
    provided = payload.get("auth")
    if not isinstance(provided, str):
        return False
    return hmac.compare_digest(provided, sign(token, payload))


@dataclass
class Mailbox:
    root: Path
    token: str = field(default_factory=lambda: secrets.token_hex(32))

    def __post_init__(self):
        self.root = Path(self.root)
        self.requests.mkdir(parents=True, exist_ok=True)
        self.responses.mkdir(parents=True, exist_ok=True)

    @property
    def requests(self) -> Path:
        return self.root / "requests"

    @property
    def responses(self) -> Path:
        return self.root / "responses"

    def env(self, observation_dir: Path | str, workspace: Path | str) -> dict:
        """Environment variables the MCP server subprocess needs.

        `AG_MCP_OBSERVATION_DIR` is the flat store the server may read move results
        from; `AG_MCP_WORKSPACE` is the wider tree `view_image` is confined to. Both
        are validated by the server, which refuses relative, nested, symlinked or
        out-of-tree paths.
        """
        return {"AG_MCP_MAILBOX": str(self.root), "AG_MCP_TOKEN": self.token,
                "AG_MCP_OBSERVATION_DIR": str(observation_dir),
                "AG_MCP_WORKSPACE": str(workspace)}

    def write_response(self, request_id: str, payload: dict) -> None:
        path = self.responses / f"{request_id}.json"
        tmp = self.responses / f".{request_id}.{os.getpid()}.tmp"
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump({"request_id": request_id, **payload}, handle,
                      separators=(",", ":"), ensure_ascii=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)

    def pending(self) -> list[tuple[str, dict]]:
        out = []
        for path in sorted(self.requests.glob("*.json")):
            try:
                with path.open(encoding="utf-8") as handle:
                    out.append((path.stem, json.load(handle)))
            except (OSError, json.JSONDecodeError):
                continue
            path.unlink(missing_ok=True)
        return out


class EpisodeController:
    """Serves mailbox requests from a live environment, on a background thread.

    Every request is authenticated before it reaches the environment, and every
    observation is written to the observation directory the server is allowed to read
    from. The controller owns the label; the server never sees it.
    """

    def __init__(self, env, mailbox: Mailbox, observation_dir: Path | str,
                 arm: str = "A0"):
        self.env = env
        self.mailbox = mailbox
        self.observation_dir = Path(observation_dir)
        self.observation_dir.mkdir(parents=True, exist_ok=True)
        self.arm = arm
        self.finished = False
        self.answer = None
        self.success = None
        self.moves = 0
        self.rejected = 0
        self.deliveries: list[dict] = []
        self.exposed: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._n = 0

    # --- lifecycle ------------------------------------------------------------

    def start(self):
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False

    # --- serving --------------------------------------------------------------

    def _serve(self):
        while not self._stop.is_set():
            for request_id, payload in self.mailbox.pending():
                self.mailbox.write_response(request_id, self._handle(payload))
            time.sleep(POLL_SECONDS)

    def _handle(self, payload: dict) -> dict:
        if payload.get("protocol") != PROTOCOL_VERSION:
            return {"ok": False, "message": "unsupported protocol"}
        if not verify(self.mailbox.token, payload):
            self.rejected += 1
            return {"ok": False, "message": "request rejected"}
        operation = payload.get("operation")
        if self.finished and operation in {"move", "submit"}:
            return {"ok": False, "message": "episode already finished"}
        if operation == "move":
            return self._move(payload.get("direction"))
        if operation == "submit":
            return self._submit(payload.get("value"))
        if operation == "delivery_receipt":
            return self._delivery_receipt(payload)
        return {"ok": False, "message": "unknown operation"}

    def audit(self) -> dict:
        """What the agent was actually able to read, and what the server delivered.

        `exposed` is what the controller made readable; `delivered` is what the server
        reported handing over. They should correspond one-to-one, and any observation
        in `delivered` that is not in `exposed` is a protocol violation.
        """
        exposed = {row["sha256"] for row in self.exposed}
        delivered = [row for row in self.deliveries if row.get("sha256")]
        unexpected = [row for row in delivered if row["sha256"] not in exposed]
        return {"n_exposed": len(self.exposed), "n_delivered": len(delivered),
                "unexpected_deliveries": unexpected,
                "rejected_requests": self.rejected,
                "consistent": not unexpected}

    def _move(self, direction):
        obs, _, terminated, truncated, info = self.env.step(
            {"action": "move", "direction": direction})
        if info.get("error"):
            return {"ok": False, "message": info["error"]}
        self.moves += 1
        path = self._write_observation(obs)
        if terminated or truncated:
            self.finished = True
        # The server reads this path, checks it against the observation store, and
        # then reports back what it delivered via a `delivery_receipt` exchange.
        return {"ok": True, "observation": str(path.resolve()),
                "steps": self.env.steps}

    def _submit(self, value):
        _, _, _, _, info = self.env.step({"action": "answer", "value": value})
        self.finished = True
        self.answer = info.get("answer")
        self.success = bool(info.get("success"))
        # Only arm A2 is told whether the answer was right.
        receipt = ("CORRECT" if self.success else "INCORRECT") if self.arm == "A2" \
            else "submission accepted"
        return {"ok": True, "message": receipt}

    def _delivery_receipt(self, payload: dict) -> dict:
        """The server reports the digest and byte count it actually delivered.

        Recorded so an episode can be audited for what the agent really received,
        rather than for what the controller believes it sent.
        """
        self.deliveries.append({
            "move_request_id": payload.get("move_request_id"),
            "sha256": payload.get("sha256"),
            "byte_count": payload.get("byte_count"),
            "delivery_format": payload.get("delivery_format"),
        })
        return {"ok": True}

    def _write_observation(self, obs):
        """Write an observation under an opaque, unguessable name.

        Sequential names would tell the agent its own step index, how many glimpses
        it has taken, and the ordering of anything it re-opened -- none of which it
        is supposed to infer from the filesystem. The original controller uses
        `secrets.token_hex(16)` for exactly this reason; so does this one.

        Every exposure is audited by digest, so a run can be checked afterwards for
        what was actually made readable.
        """
        self._n += 1
        path = self.observation_dir / f"{secrets.token_hex(16)}.png"
        obs.save(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.exposed.append({"index": self._n, "path": str(path.resolve()),
                             "sha256": digest, "bytes": path.stat().st_size})
        return path

    def write_initial_observation(self, obs):
        return self._write_observation(obs)
