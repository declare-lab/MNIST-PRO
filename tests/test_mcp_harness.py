"""The MCP harness, driven end to end with no model and no network.

The server is spawned as a real subprocess and driven over stdio JSON-RPC, so this
exercises the actual protocol: initialize, tools/list, tools/call, the HMAC-signed
mailbox, and image delivery. Previously `mnist_pro/harness` had no tests at all.
"""

import base64
import io
import json

import numpy as np
import pytest
from PIL import Image

from mnist_pro.harness.mcp_client import MCPError, MCPStdioClient
from mnist_pro.harness.protocol import Mailbox, canonical_bytes, sign, verify
from mnist_pro.harness.session import MCPEpisode, write_mcp_config
from mnist_pro.rendering import CanvasSpec


def digit_image():
    a = np.zeros((28, 28), dtype=np.uint8)
    a[6:22, 10:14] = 255
    a[6:10, 10:20] = 255
    return Image.fromarray(a, mode="L")


def episode(digits=1, **kw):
    label = "0" if digits == 1 else "00"
    return MCPEpisode([digit_image() for _ in range(digits)], label,
                      digits=digits, spec=CanvasSpec(digits=digits), **kw)


# --- signing ------------------------------------------------------------------

def test_signature_covers_the_payload_but_not_the_signature():
    payload = {"protocol": 1, "request_id": "abc", "operation": "move",
               "direction": "up"}
    payload["auth"] = sign("t0ken", payload)
    assert verify("t0ken", payload)
    assert b'"auth"' not in canonical_bytes(payload)


def test_tampering_or_a_wrong_token_is_rejected():
    payload = {"protocol": 1, "request_id": "abc", "operation": "move",
               "direction": "up"}
    payload["auth"] = sign("t0ken", payload)
    assert not verify("other", payload)
    tampered = {**payload, "direction": "down"}
    assert not verify("t0ken", tampered)
    assert not verify("t0ken", {k: v for k, v in payload.items() if k != "auth"})


def test_unsigned_request_never_reaches_the_environment():
    with episode() as ep:
        ep.mailbox.write_response  # controller is running
        request_id = "deadbeef"
        (ep.mailbox.requests / f"{request_id}.json").write_text(json.dumps(
            {"protocol": 1, "request_id": request_id, "operation": "move",
             "direction": "up"}))
        deadline_moves = ep.controller.moves
        for _ in range(200):
            path = ep.mailbox.responses / f"{request_id}.json"
            if path.exists():
                break
            import time
            time.sleep(0.01)
        response = json.loads((ep.mailbox.responses / f"{request_id}.json").read_text())
        assert response["ok"] is False
        assert ep.controller.moves == deadline_moves
        assert ep.controller.rejected == 1


# --- protocol -----------------------------------------------------------------

def test_initialize_and_tools_list():
    with episode() as ep:
        assert ep.client.server_info.get("name") == "activeglimpse"
        assert sorted(ep.client.tool_names()) == ["move", "submit", "view_image"]


def test_move_tool_returns_native_image_content():
    with episode() as ep:
        result = ep.client.move("right")
        assert not result.is_error
        images = result.images()
        assert len(images) == 1
        raw = base64.b64decode(images[0]["data"])
        assert raw[:8] == b"\x89PNG\r\n\x1a\n"
        decoded = Image.open(io.BytesIO(raw))
        assert decoded.size == (224, 224)


def test_move_advances_the_real_environment():
    with episode() as ep:
        before = ep.env.window
        ep.client.move("right")
        assert ep.env.window != before
        assert ep.controller.moves == 1
        assert len(ep.result().windows) == 2


def test_bad_direction_is_an_error_not_a_crash():
    with episode() as ep:
        result = ep.client.move("sideways")
        assert result.is_error
        assert ep.controller.moves == 0
        ep.client.move("up")          # server still usable afterwards
        assert ep.controller.moves == 1


def test_submit_scores_and_locks_the_episode():
    with episode() as ep:
        result = ep.client.submit("0")
        assert not result.is_error
        assert ep.controller.success is True
        assert ep.result().termination_reason == "answered"
        after = ep.client.move("up")
        assert after.is_error


def test_two_digit_episode_uses_string_answers():
    with episode(digits=2) as ep:
        assert ep.env.canvas.shape == (224, 448)
        ep.client.submit("00")
        assert ep.controller.success is True


@pytest.mark.parametrize("arm,expected", [
    ("A0", "submission accepted"),
    ("A1", "submission accepted"),
    ("A2", "CORRECT"),
])
def test_only_arm_a2_receives_correctness_feedback(arm, expected):
    with episode(arm=arm) as ep:
        text = ep.client.submit("0").text()
        assert expected in text


def test_a2_reports_an_incorrect_answer_as_incorrect():
    with episode(arm="A2") as ep:
        assert "INCORRECT" in ep.client.submit("9").text()


def test_step_limit_is_enforced_through_the_harness():
    with episode(max_steps=3) as ep:
        for _ in range(3):
            ep.client.move("right")
        assert ep.result().termination_reason == "step_limit"
        assert ep.client.move("right").is_error


def test_view_image_reads_an_observation_back():
    with episode() as ep:
        ep.client.move("right")
        observation = ep.result().observations[-1]
        result = ep.client.view_image(observation)
        assert not result.is_error
        assert result.images()


def test_view_image_refuses_a_path_outside_the_episode(tmp_path):
    outside = tmp_path / "secret.png"
    Image.new("RGB", (8, 8)).save(outside)
    with episode() as ep:
        assert ep.client.view_image(str(outside)).is_error


# --- external runtimes --------------------------------------------------------

def test_mcp_config_is_written_for_external_agents(tmp_path):
    with episode() as ep:
        config = json.loads(ep.config_path.read_text())
        server = config["mcpServers"]["activeglimpse"]
        assert server["args"][0].endswith("ag_mcp_server.py")
        assert set(server["env"]) == {"AG_MCP_MAILBOX", "AG_MCP_TOKEN",
                                      "AG_MCP_OBSERVATION_DIR", "AG_MCP_WORKSPACE"}


def test_config_token_is_per_episode():
    with episode() as a, episode() as b:
        ta = json.loads(a.config_path.read_text())["mcpServers"]["activeglimpse"]["env"]["AG_MCP_TOKEN"]
        tb = json.loads(b.config_path.read_text())["mcpServers"]["activeglimpse"]["env"]["AG_MCP_TOKEN"]
        assert ta != tb


def test_client_reports_a_dead_server_clearly(tmp_path):
    client = MCPStdioClient(command=["/usr/bin/false"])
    with pytest.raises((MCPError, OSError)):
        client.start()
    client.close()
