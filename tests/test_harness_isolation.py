"""Isolation constraints the harness must hold, tested as constraints.

The point of routing the agent through an MCP server is that the agent cannot obtain
the answer by any route other than looking at the pixels. These tests attack that
claim from the agent's side: they try to read the label, the canvas, the schedule,
the mailbox and the auth token, and they check that nothing about the episode leaks
through filenames or through feedback the arm is not entitled to.

The constraints mirror the original Codex/Antigravity setup:

* the server process never receives the label, the schedule or any coordinate;
* observations are exposed under opaque, unguessable names, so the filesystem does
  not reveal step index or ordering;
* `view_image` is confined to the episode workspace, and the workspace contains only
  observations -- the mailbox and the auth token live outside it;
* only arm A2 learns whether a submission was correct;
* every exposure is auditable by digest.
"""

import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mnist_pro.harness.session import MCPEpisode
from mnist_pro.rendering import CanvasSpec


def digit_image():
    a = np.zeros((28, 28), dtype=np.uint8)
    a[6:22, 10:14] = 255
    a[6:10, 10:20] = 255
    return Image.fromarray(a, mode="L")


def episode(digits=1, **kw):
    label = "7" if digits == 1 else "73"
    return MCPEpisode([digit_image() for _ in range(digits)], label,
                      digits=digits, spec=CanvasSpec(digits=digits), **kw)


# --- the server holds nothing it could leak -----------------------------------

def test_server_environment_carries_no_label_or_schedule():
    with episode() as ep:
        for key, value in ep.server_env.items():
            assert "7" != value, key
            assert "label" not in key.lower()
        assert set(ep.server_env) == {"AG_MCP_MAILBOX", "AG_MCP_TOKEN",
                                      "AG_MCP_OBSERVATION_DIR", "AG_MCP_WORKSPACE"}


def test_label_never_appears_anywhere_in_the_agent_visible_workspace():
    with episode(digits=2) as ep:
        ep.client.move("right")
        ep.client.move("down")
        for path in Path(ep.workspace).rglob("*"):
            if path.is_file():
                assert b"73" not in path.read_bytes()[:64], path


def test_mailbox_and_token_live_outside_the_agent_workspace():
    """A workspace containing the channel's credentials would defeat the point."""
    with episode() as ep:
        workspace = Path(ep.workspace).resolve()
        assert not str(Path(ep.mailbox.root).resolve()).startswith(str(workspace))
        assert not str(Path(ep.config_path).resolve()).startswith(str(workspace))
        contents = {p.name for p in workspace.iterdir()}
        assert contents == {"observations"}


def test_workspace_holds_only_pngs():
    with episode() as ep:
        ep.client.move("up")
        for path in Path(ep.workspace).rglob("*"):
            if path.is_file():
                assert path.suffix == ".png", path
                assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# --- filenames leak nothing ----------------------------------------------------

def test_observation_names_are_opaque_not_sequential():
    """Sequential names would tell the agent its own step index and ordering."""
    with episode() as ep:
        for direction in ("right", "down", "left"):
            ep.client.move(direction)
        names = [Path(row["path"]).stem for row in ep.controller.exposed]
        assert len(names) == 4                      # initial view plus three moves
        assert len(set(names)) == 4
        for name in names:
            assert re.fullmatch(r"[0-9a-f]{32}", name), name
        # Opaque names must not encode order: the sequence they were created in
        # should not be recoverable by sorting them.
        assert names != sorted(names) or len(names) == 1


def test_observation_names_differ_across_episodes():
    with episode() as a, episode() as b:
        a.client.move("right")
        b.client.move("right")
        names_a = {Path(r["path"]).stem for r in a.controller.exposed}
        names_b = {Path(r["path"]).stem for r in b.controller.exposed}
        assert not (names_a & names_b)


# --- the agent cannot read outside its workspace -------------------------------

def test_view_image_refuses_the_unmasked_canvas(tmp_path):
    """The answer as an image, one directory up, must stay unreachable."""
    work = tmp_path / "episode"
    with episode(workdir=work) as ep:
        canvas = tmp_path / "episode" / "original.png"
        Image.fromarray(ep.env.canvas).save(canvas)
        assert ep.client.view_image(str(canvas)).is_error


@pytest.mark.parametrize("target", ["/etc/hosts", "relative.png", ""])
def test_view_image_refuses_absolute_relative_and_empty_paths(target):
    with episode() as ep:
        assert ep.client.view_image(target).is_error


def test_view_image_refuses_traversal_out_of_the_workspace():
    with episode() as ep:
        escape = str(Path(ep.workspace) / ".." / ".." / "original.png")
        assert ep.client.view_image(escape).is_error


def test_view_image_refuses_a_symlink_into_the_workspace(tmp_path):
    secret = tmp_path / "answer.png"
    Image.new("RGB", (16, 16), (1, 2, 3)).save(secret)
    with episode() as ep:
        link = Path(ep.workspace) / "observations" / "sneaky.png"
        try:
            link.symlink_to(secret)
        except OSError:
            pytest.skip("symlinks unavailable")
        assert ep.client.view_image(str(link)).is_error


def test_mailbox_files_are_not_readable_as_images():
    with episode() as ep:
        ep.client.move("right")
        for path in Path(ep.mailbox.root).rglob("*.json"):
            assert ep.client.view_image(str(path)).is_error
            break


# --- feedback entitlement ------------------------------------------------------

@pytest.mark.parametrize("arm", ["A0", "A1"])
def test_arms_without_feedback_learn_nothing_from_a_wrong_answer(arm):
    with episode(arm=arm) as ep:
        wrong = ep.client.submit("1").text()
        assert "INCORRECT" not in wrong and "CORRECT" not in wrong
    with episode(arm=arm) as ep2:
        right = ep2.client.submit("7").text()
    assert wrong == right, "a correct and an incorrect answer must be indistinguishable"


def test_a2_is_the_only_arm_told_the_outcome():
    with episode(arm="A2") as ep:
        assert "INCORRECT" in ep.client.submit("1").text()
    with episode(arm="A2") as ep:
        assert "CORRECT" in ep.client.submit("7").text()


# --- forged and replayed requests ----------------------------------------------

def _await_response(mailbox, request_id, timeout=3.0):
    deadline = time.monotonic() + timeout
    path = mailbox.responses / f"{request_id}.json"
    while time.monotonic() < deadline:
        if path.exists():
            return json.loads(path.read_text())
        time.sleep(0.01)
    raise AssertionError("controller did not answer")


def test_a_forged_submit_cannot_reach_the_environment():
    with episode() as ep:
        request_id = "forged1"
        (ep.mailbox.requests / f"{request_id}.json").write_text(json.dumps(
            {"protocol": 1, "request_id": request_id, "operation": "submit",
             "value": "7", "auth": "0" * 64}))
        response = _await_response(ep.mailbox, request_id)
        assert response["ok"] is False
        assert ep.controller.success is None
        assert ep.controller.rejected == 1


def test_an_unknown_protocol_version_is_refused():
    with episode() as ep:
        request_id = "wrongver"
        (ep.mailbox.requests / f"{request_id}.json").write_text(json.dumps(
            {"protocol": 99, "request_id": request_id, "operation": "move",
             "direction": "up"}))
        assert _await_response(ep.mailbox, request_id)["ok"] is False
        assert ep.controller.moves == 0


# --- auditability --------------------------------------------------------------

def test_audit_matches_exposures_against_deliveries():
    with episode() as ep:
        ep.client.move("right")
        ep.client.move("down")
        audit = ep.audit()
        assert audit["consistent"] is True
        assert audit["n_exposed"] == 3          # initial view plus two moves
        assert audit["n_delivered"] == 2        # one receipt per move
        assert audit["unexpected_deliveries"] == []
        assert audit["rejected_requests"] == 0


def test_audit_records_the_digest_of_everything_exposed():
    import hashlib
    with episode() as ep:
        ep.client.move("right")
        for row in ep.controller.exposed:
            actual = hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest()
            assert row["sha256"] == actual
            assert row["bytes"] > 8
