#!/opt/anaconda3/bin/python
"""Run one matched MNIST-PRO Level 1 arm under the native-image protocol.

This file is controller-only. It must never be mounted into a scored workspace.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import csv
import fcntl
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import hmac
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import re
import select
import secrets
import shutil
import signal
import socket
import socketserver
import stat
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from typing import Any, Iterable
import unicodedata
from urllib.parse import unquote
import uuid

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision.datasets import MNIST


BASE = Path(
    "/Users/poriasoujanya/Documents/Codex/2026-08-25/"
    "files-pasted-by-the-user-act"
)
SOURCE_ROOT = Path("/Users/poriasoujanya/Documents/activeglimpse-main")
ENVIRONMENT_PY = SOURCE_ROOT / "src/environment.py"
LEVEL1_SUMMARY = SOURCE_ROOT / (
    "main_table_logs/"
    "gemini-3.7-flash_img224_box64_step32_maxsteps36_seed42_"
    "DefaultVisionAgent_hist-1_evalsets10_20260824_105544/"
    "results_summary.json"
)
LEVEL2_SUMMARY = SOURCE_ROOT / (
    "main_table_logs/"
    "multidigit_2_gemini-3.7-flash_img224_box64_step32_maxsteps78_"
    "seed42_MultiDigitDefaultVisionAgent_hist-1_evalsets10_"
    "20260824_111931/results_summary.json"
)
LEVEL1_PRIOR_ROOT = LEVEL1_SUMMARY.parent
LEVEL2_PRIOR_ROOT = LEVEL2_SUMMARY.parent
HOST_AUTH = Path("/Users/poriasoujanya/.codex/auth.json")
HOST_MEMORY = Path("/Users/poriasoujanya/.codex/memories")
SYSTEM_CA_BUNDLE = Path("/private/etc/ssl/cert.pem")
A1_NOTES_DOCUMENT = Path(
    "/Users/poriasoujanya/Downloads/notes_instruction_A1_no_feedback.md"
)
A2_NOTES_DOCUMENT = Path(
    "/Users/poriasoujanya/Downloads/notes_instruction_A2_with_feedback.md"
)
DATA_ROOT = BASE / "work/mnist_pro_cleanroom/controller_private/data"
THREE_ARM_PRIVATE_ROOT = BASE / "work/mnist_pro_three_arm/controller_private"
RUNTIME_PARENT = THREE_ARM_PRIVATE_ROOT / "unconfigured/runtimes"
AUTH_VAULT_PARENT = THREE_ARM_PRIVATE_ROOT / "unconfigured/auth_vaults"
AG_SOURCE = BASE / "work/mnist_pro_cleanroom/generic_bin/ag"
AG_MCP_SOURCE = BASE / "work/mnist_pro_three_arm/ag_mcp_server.py"
CLI_PACKAGE_ROOT = BASE / (
    "work/standalone-codex/node_modules/@openai/codex-darwin-arm64/"
    "vendor/aarch64-apple-darwin"
)
CODEX = CLI_PACKAGE_ROOT / "bin/codex"
CODE_MODE_HOST = CLI_PACKAGE_ROOT / "bin/codex-code-mode-host"
PYTHON_ROOT = Path("/opt/anaconda3")
HOMEBREW_ROOT = Path("/opt/homebrew")
IMAGEMAGICK = HOMEBREW_ROOT / "bin/magick"

MODEL = "gpt-5.6-terra"
REASONING_EFFORT = "medium"
LEVELS = (1,)
SEED = 42
NUM_EVAL_SETS = 10
IMAGE_SIZE = 224
BOX_SIZE = 64
STEP_SIZE = 32
BUDGETS = {1: 36, 2: 78}
EXPECTED_EPISODES = 100
DEFAULT_WORKERS = 1
DEFAULT_TIMEOUT_SECONDS = 30 * 60
IMAGE_DELIVERY_PROTOCOL = "mcp-image-content-v2"
FROZEN_SHUFFLE_SEED_HEX = (
    "3c86be160cd0b7dde05768c2358b13aa7255b7968f5fb1041c6b8f96f2656d04"
)
FROZEN_PERMUTATION_SHA256 = (
    "47eec83c7f30340d045af796c741dd8193b438555392f52a8e4954fdad7d9d63"
)
A1_ABORTED_PILOT_RUN_ROOT = (
    BASE / "outputs/mnist_pro_arm_a1_20260826_run01"
).resolve()
A1_ABORTED_PILOT_PROCESS_MARKERS = (
    str(A1_ABORTED_PILOT_RUN_ROOT),
    "/private/tmp/68a9388a3eb452f6d392efbdb080b6d5",
    "/private/tmp/9afada32a0b88d985329d7b5eb011c38",
    "mnistpro-a1-audit-0hebbe18",
)

ACTIVE_ARM = "A1"
ACTIVE_SUITE_ID = "unconfigured"

# The only accepted continuation source is the immutable failed three-arm suite
# below.  These pins prevent a similarly shaped directory, a regenerated
# schedule, or an edited report prefix from being silently treated as the same
# experiment.
CONTINUATION_SOURCE_SUITE_ID = "l1-5393808a4e"
CONTINUATION_SOURCE_SUITE_ROOT = (
    BASE / "outputs/mnist_pro_three_arm_l1_20260826_120644_5393808a4e"
).resolve()
CONTINUATION_SOURCE_LAUNCH_MANIFEST_SHA256 = (
    "14eb35f432a813e18ec9528d0c48e3eadbaa94b58971835e73a352bd47b00308"
)
CONTINUATION_SOURCE_RELEASE_SHA256 = (
    "96d4b10c20033a05f1e519dc79d2749297fc078552316417a4c3b6eeac86299a"
)
CONTINUATION_SOURCE_FAILURE_SHA256 = (
    "0fe66b94df3340d9f82ec6836a0735afd26e065666354fc08f85ae870ca4107e"
)
CONTINUATION_SOURCE_CROSS_ARM_PREFLIGHT_SHA256 = (
    "aaa2103abb93c3078d101af0220b7a48e8f82347c47dd9875635acfcaac6ea58"
)
CONTINUATION_SOURCE_CONTROLLER_SHA256 = (
    "8c0f2d3c1327931f8c1a62b3fcd0308bb52bc010e7a01f12030fde2cbedb5a39"
)
CONTINUATION_SOURCE_MCP_SHA256 = (
    "90cbb92412d5f18a65840a891a34809e948def6f073d3e7b4cd93782208ca746"
)
CONTINUATION_CARRIED_ATTEMPTS = {"A0": 12, "A1": 11, "A2": 7}
CONTINUATION_SOURCE_SELECTED_DIGESTS = {
    "A0": "b8a6719f7b308fa03d11104476b861070d3eaa2435601f46e9ff438ebb977ebf",
    "A1": "f765609ba2fcf1d5463e4ce71512018201a99dc685753d31d3e07b8ac30e2c2d",
    "A2": "c05d8487c83cb5062d06f6a6f04dcf468082067b0123aabc4c7be2b5b55a9fda",
}
CONTINUATION_A2_MIGRATION_FROM_SHA256 = (
    "1d18418ec64ea15a12734a1122365b5eb1a883465ef4484c8a82e6d27a752b59"
)
CONTINUATION_A2_MIGRATION_TO_SHA256 = (
    "63efb0f729a23c2447c4b0ce689d4a63882e0bde9a46c0cb4e20492a60b63ee8"
)


class ControllerAbort(RuntimeError):
    """Raised by the controller signal handler to unwind through cleanup blocks."""

CRITICAL_ARM_INFRASTRUCTURE_CONDITIONS = (
    "external_filesystem_isolation_passed",
    "exactly_one_scored_codex_launch",
    "exactly_one_fresh_thread",
    "requested_model_exact",
    "resolved_model_consistent_if_reported",
    "reasoning_effort_exact",
    "exact_arm_prompt_and_one_initial_image",
    "opaque_agent_visible_image_paths",
    "guaranteed_native_image_delivery",
    "native_delivery_receipts_match_accepted_moves",
    "no_legacy_path_only_ag_commands",
    "no_forbidden_access_observed",
    "model_tool_general_network_disabled",
    "native_submit_transport_consistent",
    "process_not_timed_out",
    "codex_exit_successful",
    "jsonl_well_formed",
    "scored_process_group_terminated",
    "controller_exception_absent",
    "notes_snapshot_exact_or_not_applicable",
    "feedback_interface_matches_arm",
    "correctness_timing_matches_arm",
    "concurrent_auth_state_unchanged",
)

PERSISTENT_NOTES_INSTRUCTION_A1 = """PERSISTENT NOTES

You have a file NOTES.md in your workspace. It is the only thing that survives
after this episode ends. Everything else, including this conversation and every
image you have seen, is discarded. Future episodes will show you this file and
nothing else.

You may rewrite NOTES.md in full at any point. There is no size cap.

Write whatever you think will help you in future episodes.

You will not be told whether your answer was right."""

PERSISTENT_NOTES_INSTRUCTION_A2 = """PERSISTENT NOTES

You have a file NOTES.md in your workspace. It is the only thing that survives
after this episode ends. Everything else, including this conversation and every
image you have seen, is discarded. Future episodes will show you this file and
nothing else.

You may rewrite NOTES.md in full at any point. There is no size cap.

Write whatever you think will help you in future episodes.

After you submit, you will be told CORRECT or INCORRECT. You will never be told
what the right answer was. You may record this however you like."""

PINNED_HASHES = {
    "environment.py": "d2973d5bb60ce38501f8b4066cf916510e7c26a3d4d0fa0659d2b817326736c4",
    "level1_summary": "48362448e87a819e162bc42ac9104493ad8b3405bf65c1cec52418e88b21c208",
    "level2_summary": "d1778e114011974e2416bd45ef2dba15047c8a065596e173b6e3b6daf620c465",
    "level1_projection": "ef9f1dbabb1780a44e440dd9bd388bc30b072dd5215f1559da3b021dc89fa5a5",
    "level2_projection": "4fcc0b7981b25a3efe1d4fc59a0ff09e5a050d93157007a8f2810a3d5d114143",
    "mnist_test_images_gz_md5": "9fb629c4189551a2d022fa330f9573f3",
    "mnist_test_labels_gz_md5": "ec29112dd5afa0611ce80d1b7f02629c",
    "mnist_test_images_idx_sha256": "0fa7898d509279e482958e8ce81c8e77db3f2f8254e26661ceb7762c4d494ce7",
    "mnist_test_labels_idx_sha256": "ff7bcfd416de33731a308c3f266cc351222c34898ecbeaf847f06e48f7ec33f2",
    "standalone_codex": "f0d8762236594359b60cfbe17f4c7e945a3ce8d1c91e74778838c968d250fb6c",
    "imagemagick": "d0d418371a96d02849eece4aa02f4eb9670d331c41e2dc6778e8d443368a6345",
    "ag": "49931d07f53557b8dd90a6395bca7a4ca59a5153772675b39083c75ec672cda8",
    "ag_mcp_server": "90cbb92412d5f18a65840a891a34809e948def6f073d3e7b4cd93782208ca746",
    "ag_mcp_tools": "063348c22881f2c9386c48293ed20691dd1cdb1972bf3c281f50a813b8a637e5",
    "system_ca_bundle": "9dae8d76e55cb08991f2b672d58999ea15560d910759c16b544f843bdffbb994",
    "a1_notes_document": "f0f0e273d96d9dbc8c42c106115a15b4750a5a019e81e19112d39d14430a811e",
    "a2_notes_document": "5cdccfe342ca2ebdd1749248d1a8d2056cb4e43d6e71515bb03467c94d769e94",
}

RECOVERED_FIRST_FOUR = {
    0: {
        "workspace_uuid": "f4c4097e98d946049b16f69bbb062fcf",
        "trace_sha256": "c0fc907ed508b34e690497c21ce7e1c12250921ae03a53db446f1acac5a92500",
        "stderr_sha256": "665ecd913ccac13138a28c2f885384b17b68388d2bdd213a2fb308e2d8a52034",
        "observation_sha256": "3047695af2b9f88ee78230a606f6dd32f50ab0a77eac8ed38101293dbe184293",
    },
    1: {
        "workspace_uuid": "c112ac398db14e44bb26594715c5b106",
        "trace_sha256": "f2dc504b9f8dd00bd8277e7fc45fd033640e68b8b720305af7c5e5891c9e2678",
        "stderr_sha256": "7da486b2be8fa597a17519115d159f98c3d5181527923e521f8eae93f3370520",
        "observation_sha256": "0365559991d4071e55c0ebfcbed54a00cb3dc138d00f1db28dcfa45bf280c134",
    },
    2: {
        "workspace_uuid": "1a828cf1969f402ebd7d39ca9e343597",
        "trace_sha256": "9201a8d291db2f2d5ba3fd5b05c5f7c7844b828c1460e0c90f37e38398ed148e",
        "stderr_sha256": "981352d1fba222621a58d6518b2a0cefd20b0d1a91b3158b3e216d863dbbbf19",
        "observation_sha256": "9c4ec310e7a1e5c3600190d2cf99deab5fd47756796e36e9eb83936009237af4",
    },
    3: {
        "workspace_uuid": "e5ef2ffba3a64cc39a4cb31928f092dc",
        "trace_sha256": "261899d7657def656a44a73e983f8bf702aa09ca4f3ffed7f2f71997df38ab42",
        "stderr_sha256": "cfc368be8b53c42cebfcc304f34d923acb3031ce886666ea115695dc5a3ddeda",
        "observation_sha256": "b0f7447bbbae71ee928bbadcb76660921af47056a2e5efa47d79f30672e2c823",
    },
}
RECOVERED_PROGRESS_SHA256 = (
    "90805402dddc45928d2451bee5136d5470b5776bfeb05cdcf78ab5af2edf3e5e"
)
RECOVERED_SCHEDULE_SHA256 = (
    "e199b0d89d9eb42f70b829a156c922e9024e80f2cb2990ce3af203fa34d33384"
)
RECOVERED_PREFLIGHT_SHA256 = (
    "db121aa262ec3db91727e26a31b10a720d49c7bc7974e5cfba6b2a1650daece3"
)
RECOVERED_CONTROLLER_SHA256 = (
    "88c607db5ced8f7bae47a9d195480806dd1e59c8ef2c91fcb5cd84e49ae19f33"
)

FORBIDDEN_MARKERS = (
    "<skills_instructions>",
    "<plugins_instructions>",
    "<apps_instructions>",
    "<recommended_plugins>",
    "<multi_agent_mode>",
    "spawn_agent",
    "/root",
    "## Memory",
    "/Users/poriasoujanya/.codex",
)

AUTONOMY_COACHING_MARKERS = (
    "inspect that returned image",
    "inspect the returned image",
    "image-viewing tool",
    "activeglimpse.view_image",
    "open a returned path",
    "composite image",
    "fused image",
)

CLEAN_CONFIG = (
    "agents.enabled=false",
    "features.multi_agent=false",
    "features.multi_agent_v2=false",
    "features.apps=false",
    "features.plugins=false",
    "features.remote_plugin=false",
    "features.plugin_sharing=false",
    "features.recommended_plugins=false",
    "features.skill_search=false",
    "features.skill_mcp_dependency_install=false",
    "features.in_app_browser=false",
    "features.browser_use=false",
    "features.browser_use_external=false",
    "features.browser_use_full_cdp_access=false",
    "features.computer_use=false",
    "features.image_generation=false",
    "features.workspace_dependencies=false",
    "features.tool_suggest=false",
    "features.hooks=false",
    "features.goals=false",
    "features.auth_elicitation=false",
    "features.default_mode_request_user_input=false",
    "features.shell_snapshot=false",
    "features.remote_compaction_v2=false",
    "features.compaction_image_budget=false",
    "features.unified_image_budget=false",
    "features.memories=false",
    "memories.use_memories=false",
    "skills.bundled.enabled=false",
    "skills.include_instructions=false",
    'web_search="disabled"',
    "include_apps_instructions=false",
    "include_collaboration_mode_instructions=false",
    "include_environment_context=false",
    "include_permissions_instructions=false",
    "model_auto_compact_token_limit=1000000",
    "suppress_unstable_features_warning=true",
)

TOOL_ITEM_TYPES = {
    "command_execution",
    "file_change",
    "mcp_tool_call",
    "dynamic_tool_call",
    "web_search",
    "image_generation",
    "computer_initialize_state",
    "computer_action",
    "tool_call",
    "custom_tool_call",
    "code_mode_tool_call",
    "code_mode_execution",
    "image_view",
    "view_image",
}

AUTONOMOUS_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
TEXT_SUFFIXES = {
    ".py",
    ".js",
    ".ts",
    ".sh",
    ".zsh",
    ".md",
    ".txt",
    ".json",
    ".toml",
    ".yaml",
    ".yml",
}
SPATIAL_TERMS = re.compile(
    r"(?i)(stitch|composit|mosaic|accumulat|global.canvas|reconstruct|"
    r"Image\.new|\.paste\s*\(|np\.(?:full|zeros|ones)\s*\(|cv2\.)"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=False
    ).encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def add_config_args(argv: list[str], values: Iterable[str]) -> None:
    for value in values:
        argv.extend(("-c", value))


def toml_quote(value: str | Path) -> str:
    return json.dumps(str(value), ensure_ascii=True)


def sign_episode_payload(token: str, payload: dict[str, Any]) -> dict[str, Any]:
    signed = dict(payload)
    signed.pop("token", None)
    signed.pop("auth", None)
    authenticated_bytes = json.dumps(
        signed,
        separators=(",", ":"),
        sort_keys=True,
        ensure_ascii=True,
    ).encode("utf-8")
    signed["auth"] = hmac.new(
        token.encode("utf-8"), authenticated_bytes, hashlib.sha256
    ).hexdigest()
    return signed


def baseline_prompt_for(level: int) -> str:
    if level == 1:
        level_text = (
            "For Level 1, the hidden canvas contains one MNIST digit from 0 "
            "through 9. Submit one digit."
        )
    elif level == 2:
        level_text = (
            "For Level 2, the hidden canvas contains two horizontally "
            "concatenated MNIST digits. Submit the two digits in left-to-right "
            "order, preserving a leading zero when applicable."
        )
    else:
        raise ValueError(f"unsupported level: {level}")
    return (
        "You are solving one partially observed MNIST episode.\n\n"
        f"{level_text}\n\n"
        "Most of the canvas is masked. The attached image is your initial "
        "observation. You can reveal another observation with:\n\n"
        "- `ag move up`\n"
        "- `ag move down`\n"
        "- `ag move left`\n"
        "- `ag move right`\n\n"
        "Inspect the returned image after each move. Submit exactly once with:\n\n"
        "`ag submit VALUE`\n\n"
        f"You have at most {BUDGETS[level]} visual observations, including the "
        "attached initial observation. Use the local tools and workspace "
        "autonomously. Finish by submitting your best answer."
    )


def native_image_prompt_for(level: int) -> str:
    """Minimal protocol amendment that makes every move result multimodal."""
    if level == 1:
        level_text = (
            "For Level 1, the hidden canvas contains one MNIST digit from 0 "
            "through 9. Submit one digit."
        )
    elif level == 2:
        level_text = (
            "For Level 2, the hidden canvas contains two horizontally "
            "concatenated MNIST digits. Submit the two digits in left-to-right "
            "order, preserving a leading zero when applicable."
        )
    else:
        raise ValueError(f"unsupported level: {level}")
    return (
        "You are solving one partially observed MNIST episode.\n\n"
        f"{level_text}\n\n"
        "Most of the canvas is masked. The attached image is your initial "
        "observation. Reveal another observation by calling the local "
        "`activeglimpse.move` tool with `direction` set to `up`, `down`, "
        "`left`, or `right`. Each successful move returns the new observation "
        "directly as native image content and also gives its opaque absolute "
        "PNG path.\n\n"
        "Submit exactly once by calling the local `activeglimpse.submit` tool "
        "with your answer in `value`.\n\n"
        f"You have at most {BUDGETS[level]} visual observations, including the "
        "attached initial observation. Use the local tools and workspace "
        "autonomously. Finish by submitting your best answer."
    )


def notes_instruction_for(arm: str) -> str | None:
    if arm == "A0":
        return None
    if arm == "A1":
        return PERSISTENT_NOTES_INSTRUCTION_A1
    if arm == "A2":
        return PERSISTENT_NOTES_INSTRUCTION_A2
    raise ValueError(f"unsupported arm: {arm}")


def prompt_for(level: int, arm: str | None = None) -> str:
    selected_arm = arm or ACTIVE_ARM
    baseline = native_image_prompt_for(level)
    instruction = notes_instruction_for(selected_arm)
    if level != 1 or instruction is None:
        return baseline
    return baseline + "\n\n" + instruction


@dataclass(frozen=True)
class EpisodeSpec:
    level: int
    episode_id: int
    indices: tuple[int, ...]

    @property
    def key(self) -> str:
        return f"level_{self.level}_episode_{self.episode_id:03d}"


class Journal:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: dict[str, Any]) -> None:
        record = {"timestamp": utc_now(), **event}
        encoded = canonical_json_bytes(record) + b"\n"
        with self.lock:
            with self.path.open("ab") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())


def read_regular_file_exact(path: Path) -> bytes:
    """Read one immutable provenance file without following links."""
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise RuntimeError(f"continuation source is not a single-link file: {path}")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(descriptor)
        if (
            opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
            or opened.st_size != before.st_size
        ):
            raise RuntimeError(f"continuation source changed while opening: {path}")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise RuntimeError(f"short read from continuation source: {path}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RuntimeError(f"continuation source grew while reading: {path}")
        after = os.fstat(descriptor)
        if (
            after.st_dev != opened.st_dev
            or after.st_ino != opened.st_ino
            or after.st_size != opened.st_size
            or after.st_mtime_ns != opened.st_mtime_ns
        ):
            raise RuntimeError(f"continuation source changed while reading: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def write_bytes_exclusive(path: Path, payload: bytes) -> None:
    """Create one commit artifact exactly once and durably."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    parent_descriptor = os.open(
        path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    try:
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)


def write_json_exclusive(path: Path, value: Any) -> None:
    write_bytes_exclusive(
        path,
        (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
    )


def continuation_source_selected_files(source_arm_root: Path) -> list[Path]:
    fixed = [
        "schedule_manifest.json",
        "preflight.json",
        "progress.jsonl",
        "CONTROLLER_FAILURE.txt",
        "BARRIER_READY.json",
        "controller_assets/controller_snapshot.py",
        "controller_assets/ag_mcp_server.py",
        "controller_assets/arm_configuration.json",
        "controller_assets/notes_instruction_A1_no_feedback.md",
        "controller_assets/notes_instruction_A2_with_feedback.md",
    ]
    files = [source_arm_root / relative for relative in fixed]
    for directory in (
        "attempts",
        "notes_history",
        "notes_candidates",
        "leakage_audits",
    ):
        root = source_arm_root / directory
        if root.is_dir():
            files.extend(sorted(root.iterdir(), key=lambda path: path.name))
    versions = source_arm_root / "notes_versions.jsonl"
    if versions.is_file():
        files.append(versions)
    for directory in ("traces", "workspaces"):
        root = source_arm_root / directory
        for path in sorted(root.rglob("*"), key=lambda item: str(item.relative_to(root))):
            metadata = path.lstat()
            if stat.S_ISDIR(metadata.st_mode):
                continue
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise RuntimeError(
                    f"nonregular continuation provenance artifact: {path}"
                )
            files.append(path)
    return files


def continuation_source_selected_digest(source_arm_root: Path) -> tuple[str, list[dict[str, Any]]]:
    entries: list[dict[str, Any]] = []
    for path in continuation_source_selected_files(source_arm_root):
        payload = read_regular_file_exact(path)
        entries.append(
            {
                "path": str(path.relative_to(source_arm_root)),
                "sha256": sha256_bytes(payload),
                "bytes": len(payload),
            }
        )
    # The historical pin was computed over ordered [relative path, hash] pairs;
    # sizes are retained in the manifest but are not part of that frozen digest.
    frozen_pairs = [[row["path"], row["sha256"]] for row in entries]
    return sha256_bytes(canonical_json_bytes(frozen_pairs)), entries


def json_from_regular_file(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(read_regular_file_exact(path))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"malformed continuation JSON: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"continuation JSON is not an object: {path}")
    return value


def no_continuation_record() -> dict[str, Any]:
    return {
        "applicable": False,
        "source_suite_id": None,
        "carried_attempts": 0,
        "next_execution_ordinal": 0,
        "state_migrations": [],
        "carried_attempt_records": [],
        "allowed_terminal_exceptions": [],
    }


def prepare_arm_continuation(
    source_arm_root: Path,
    run_root: Path,
    shuffled_specs: list[EpisodeSpec],
) -> tuple[list[dict[str, Any]], bytes, dict[str, Any]]:
    """Validate and materialize the one authorized append-only run prefix."""
    expected_source_arm = CONTINUATION_SOURCE_SUITE_ROOT / ACTIVE_ARM
    source_arm_root = source_arm_root.resolve()
    if source_arm_root != expected_source_arm:
        raise RuntimeError(
            "--continue-from-arm-root is not the pinned failed source arm"
        )
    source_suite_root = source_arm_root.parent
    if source_suite_root.name != "mnist_pro_three_arm_l1_20260826_120644_5393808a4e":
        raise RuntimeError("continuation source suite directory name changed")

    root_pins = {
        "launch_manifest.json": CONTINUATION_SOURCE_LAUNCH_MANIFEST_SHA256,
        "PREFLIGHT_RELEASE.json": CONTINUATION_SOURCE_RELEASE_SHA256,
        "SUITE_FAILURE.txt": CONTINUATION_SOURCE_FAILURE_SHA256,
        "cross_arm_preflight.json": CONTINUATION_SOURCE_CROSS_ARM_PREFLIGHT_SHA256,
    }
    root_artifacts: list[dict[str, Any]] = []
    for name, expected_hash in root_pins.items():
        payload = read_regular_file_exact(source_suite_root / name)
        observed_hash = sha256_bytes(payload)
        if observed_hash != expected_hash:
            raise RuntimeError(f"pinned continuation source changed: {name}")
        root_artifacts.append(
            {"path": name, "sha256": observed_hash, "bytes": len(payload)}
        )

    launch = json_from_regular_file(source_suite_root / "launch_manifest.json")
    release = json_from_regular_file(source_suite_root / "PREFLIGHT_RELEASE.json")
    if (
        launch.get("suite_id") != CONTINUATION_SOURCE_SUITE_ID
        or launch.get("retry_policy") != "none"
        or launch.get("arms") != ["A0", "A1", "A2"]
        or launch.get("controller_sha256") != CONTINUATION_SOURCE_CONTROLLER_SHA256
        or launch.get("ag_mcp_sha256") != CONTINUATION_SOURCE_MCP_SHA256
        or launch.get("frozen_permutation_sha256") != FROZEN_PERMUTATION_SHA256
    ):
        raise RuntimeError("continuation launch manifest does not match pinned suite")
    if (
        release.get("released") is not True
        or release.get("suite_id") != CONTINUATION_SOURCE_SUITE_ID
        or release.get("schedule_sha256") != FROZEN_PERMUTATION_SHA256
        or release.get("controller_sha256") != CONTINUATION_SOURCE_CONTROLLER_SHA256
        or release.get("ag_mcp_sha256") != CONTINUATION_SOURCE_MCP_SHA256
    ):
        raise RuntimeError("continuation preflight release does not match pinned suite")

    selected_digest, selected_artifacts = continuation_source_selected_digest(
        source_arm_root
    )
    if selected_digest != CONTINUATION_SOURCE_SELECTED_DIGESTS[ACTIVE_ARM]:
        raise RuntimeError(
            f"pinned {ACTIVE_ARM} continuation artifact prefix changed"
        )
    source_controller = source_arm_root / "controller_assets/controller_snapshot.py"
    source_mcp = source_arm_root / "controller_assets/ag_mcp_server.py"
    if (
        sha256_file(source_controller) != CONTINUATION_SOURCE_CONTROLLER_SHA256
        or sha256_file(source_mcp) != CONTINUATION_SOURCE_MCP_SHA256
    ):
        raise RuntimeError("continuation controller or MCP snapshot changed")

    schedule = json_from_regular_file(source_arm_root / "schedule_manifest.json")
    preflight = json_from_regular_file(source_arm_root / "preflight.json")
    expected_ids = [spec.episode_id for spec in shuffled_specs]
    if (
        schedule.get("configuration", {}).get("arm") != ACTIVE_ARM
        or schedule.get("configuration", {}).get("suite_id")
        != CONTINUATION_SOURCE_SUITE_ID
        or schedule.get("shuffle", {}).get("execution_episode_ids") != expected_ids
        or schedule.get("shuffle", {}).get("execution_permutation_sha256")
        != FROZEN_PERMUTATION_SHA256
        or preflight.get("passed") is not True
        or preflight.get("configuration", {}).get("arm") != ACTIVE_ARM
        or preflight.get("configuration", {}).get("suite_id")
        != CONTINUATION_SOURCE_SUITE_ID
        or preflight.get("controller_snapshot_sha256")
        != CONTINUATION_SOURCE_CONTROLLER_SHA256
        or preflight.get("ag_mcp_snapshot_sha256") != CONTINUATION_SOURCE_MCP_SHA256
    ):
        raise RuntimeError("continuation arm schedule or preflight changed")

    carried_count = CONTINUATION_CARRIED_ATTEMPTS[ACTIVE_ARM]
    attempt_files = sorted((source_arm_root / "attempts").glob("*.json"))
    if len(attempt_files) != carried_count:
        raise RuntimeError(f"unexpected {ACTIVE_ARM} continuation attempt count")
    reports_by_ordinal: dict[int, tuple[dict[str, Any], Path, bytes]] = {}
    for path in attempt_files:
        payload = read_regular_file_exact(path)
        try:
            report = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"malformed carried attempt: {path}") from error
        if not isinstance(report, dict) or not isinstance(report.get("execution_ordinal"), int):
            raise RuntimeError(f"invalid carried attempt schema: {path}")
        ordinal = report["execution_ordinal"]
        if ordinal in reports_by_ordinal:
            raise RuntimeError("duplicate carried execution ordinal")
        reports_by_ordinal[ordinal] = (report, path, payload)
    if sorted(reports_by_ordinal) != list(range(carried_count)):
        raise RuntimeError("carried attempts are not one contiguous prefix")

    reports: list[dict[str, Any]] = []
    carried_records: list[dict[str, Any]] = []
    thread_ids: list[str] = []
    interrupted_failures = {
        "guaranteed_native_image_delivery",
        "exactly_one_submission",
        "exactly_one_native_submit_receipt",
        "codex_exit_successful",
        "controller_exception_absent",
    }
    for ordinal in range(carried_count):
        report, source_path, source_bytes = reports_by_ordinal[ordinal]
        spec = shuffled_specs[ordinal]
        report_paths = report.get("paths", {})
        try:
            reported_attempt = Path(report_paths["attempt_report"]).resolve(strict=True)
            reported_trace = Path(report_paths["jsonl_trace"]).resolve(strict=True)
            reported_stderr = Path(report_paths["stderr"]).resolve(strict=True)
            reported_workspace = Path(report_paths["workspace"]).resolve(strict=True)
        except (KeyError, OSError) as error:
            raise RuntimeError(
                f"carried report path evidence is missing at ordinal {ordinal}"
            ) from error
        if (
            source_path.name != f"{spec.key}.json"
            or reported_attempt != source_path.resolve(strict=True)
            or not reported_trace.is_relative_to(source_arm_root / "traces")
            or not reported_trace.is_file()
            or not reported_stderr.is_relative_to(source_arm_root / "traces")
            or not reported_stderr.is_file()
            or not reported_workspace.is_relative_to(source_arm_root / "workspaces")
            or not reported_workspace.is_dir()
            or report.get("attempt_key") != spec.key
            or report.get("configuration", {}).get("arm") != ACTIVE_ARM
            or report.get("configuration", {}).get("episode_id") != spec.episode_id
            or report.get("schedule", {}).get("source_manifest_sha256")
            != PINNED_HASHES["level1_summary"]
            or report.get("process", {}).get("scored_codex_launches") != 1
            or report.get("process", {}).get("process_group_terminated") is not True
            or report.get("trace_audit", {}).get("thread_started_count") != 1
            or len(report.get("trace_audit", {}).get("thread_ids", [])) != 1
        ):
            raise RuntimeError(f"carried attempt evidence mismatch at ordinal {ordinal}")
        thread_ids.extend(report["trace_audit"]["thread_ids"])
        false_conditions = {
            name
            for name, passed in report.get("validity_conditions", {}).items()
            if passed is not True
        }
        terminal = ordinal == carried_count - 1
        allowed_terminal_exception = False
        exception_reason: str | None = None
        if not terminal:
            if report.get("validity") != "VALID" or false_conditions:
                raise RuntimeError("nonterminal carried attempt is not fully valid")
        elif ACTIVE_ARM in {"A0", "A1"}:
            if (
                report.get("validity") != "INVALID"
                or false_conditions != interrupted_failures
                or report.get("process", {}).get("returncode") != -signal.SIGTERM
                or report.get("process", {}).get("timed_out") is not False
                or "ControllerAbort: controller received signal 15"
                not in (report.get("controller_exception") or "")
            ):
                raise RuntimeError("interrupted terminal attempt evidence changed")
            allowed_terminal_exception = True
            exception_reason = "suite_fail_fast_sigterm_after_sibling_arm_halt"
        else:
            note = report.get("persistent_notes", {})
            if (
                report.get("validity") != "INVALID"
                or false_conditions != {"notes_snapshot_exact_or_not_applicable"}
                or report.get("process", {}).get("returncode") != 0
                or report.get("process", {}).get("timed_out") is not False
                or report.get("controller_exception") is not None
                or note.get("previous_state_sha256")
                != CONTINUATION_A2_MIGRATION_FROM_SHA256
                or note.get("next_state_sha256")
                != CONTINUATION_A2_MIGRATION_FROM_SHA256
                or note.get("candidate_sha256")
                != CONTINUATION_A2_MIGRATION_TO_SHA256
                or Path(note.get("raw_structurally_invalid_candidate", "")).resolve(
                    strict=True
                )
                != (source_arm_root / "notes_candidates/ep_007.raw").resolve(
                    strict=True
                )
                or note.get("strict_utf8") is not True
                or note.get("state_eligible") is not False
            ):
                raise RuntimeError("A2 cap-triggered terminal attempt evidence changed")
            allowed_terminal_exception = True
            exception_reason = "historical_notes_cap_halt_before_no_cap_amendment"

        destination = run_root / "attempts" / source_path.name
        write_bytes_exclusive(destination, source_bytes)
        attempt_hash = sha256_bytes(source_bytes)
        carried_record = {
            "execution_ordinal": ordinal,
            "episode_id": spec.episode_id,
            "attempt_key": spec.key,
            "attempt_sha256": attempt_hash,
            "source_relative_path": str(source_path.relative_to(source_suite_root)),
            "destination_relative_path": str(destination.relative_to(run_root)),
            "bytes_copied_exactly": True,
            "source_validity": report["validity"],
            "allowed_terminal_exception": allowed_terminal_exception,
            "terminal_exception_reason": exception_reason,
        }
        write_json_exclusive(
            run_root / "slot_commits" / f"slot_{ordinal + 1:03d}.json",
            {
                "schema_version": 1,
                "commit_kind": "carried_immutable_source_attempt",
                "arm": ACTIVE_ARM,
                "suite_id": ACTIVE_SUITE_ID,
                "source_suite_id": CONTINUATION_SOURCE_SUITE_ID,
                **carried_record,
            },
        )
        reports.append(report)
        carried_records.append(carried_record)
    if len(set(thread_ids)) != carried_count:
        raise RuntimeError("carried prefix does not contain fresh unique threads")

    notes_enabled = ACTIVE_ARM in {"A1", "A2"}
    approved_notes = b""
    if ACTIVE_ARM == "A0":
        if any(
            (source_arm_root / name).exists()
            and any((source_arm_root / name).iterdir())
            for name in ("notes_history", "notes_candidates", "leakage_audits")
        ) or (source_arm_root / "notes_versions.jsonl").exists():
            raise RuntimeError("A0 source unexpectedly contains persistent notes state")
    if notes_enabled:
        history_names = [f"ep_{position:03d}.md" for position in range(1, carried_count + 1)]
        histories = sorted((source_arm_root / "notes_history").glob("ep_*.md"))
        audits = sorted((source_arm_root / "leakage_audits").glob("ep_*.json"))
        if [path.name for path in histories] != history_names:
            raise RuntimeError("carried notes history is not a contiguous prefix")
        if [path.name for path in audits] != [name.replace(".md", ".json") for name in history_names]:
            raise RuntimeError("carried notes audits are not a contiguous prefix")
        versions_path = source_arm_root / "notes_versions.jsonl"
        versions_bytes = read_regular_file_exact(versions_path)
        version_lines = versions_bytes.splitlines()
        if len(version_lines) != carried_count:
            raise RuntimeError("carried notes version count changed")
        expected_previous = sha256_bytes(b"")
        for ordinal, (history_path, audit_path, line, report) in enumerate(
            zip(histories, audits, version_lines, reports)
        ):
            history_bytes = read_regular_file_exact(history_path)
            history_hash = sha256_bytes(history_bytes)
            note = report["persistent_notes"]
            try:
                version = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise RuntimeError("malformed carried notes version") from error
            if (
                note.get("previous_state_sha256") != expected_previous
                or Path(note.get("history_snapshot", "")).resolve(strict=True)
                != history_path.resolve(strict=True)
                or note.get("history_snapshot_sha256") != history_hash
                or note.get("next_state_sha256") != history_hash
                or version.get("execution_ordinal_zero_based") != ordinal
                or version.get("previous_state_sha256") != expected_previous
                or version.get("history_snapshot_sha256") != history_hash
                or version.get("next_state_sha256") != history_hash
                or json_from_regular_file(audit_path).get("next_state_sha256")
                != history_hash
            ):
                raise RuntimeError("carried notes history/version/report chain changed")
            write_bytes_exclusive(
                run_root / "notes_history" / history_path.name, history_bytes
            )
            write_bytes_exclusive(
                run_root / "leakage_audits" / audit_path.name,
                read_regular_file_exact(audit_path),
            )
            expected_previous = history_hash
        write_bytes_exclusive(run_root / "notes_versions.jsonl", versions_bytes)
        approved_notes = read_regular_file_exact(histories[-1])

    migrations: list[dict[str, Any]] = []
    if ACTIVE_ARM == "A2":
        raw_source = source_arm_root / "notes_candidates/ep_007.raw"
        raw_candidate = read_regular_file_exact(raw_source)
        if (
            sha256_bytes(approved_notes) != CONTINUATION_A2_MIGRATION_FROM_SHA256
            or sha256_bytes(raw_candidate) != CONTINUATION_A2_MIGRATION_TO_SHA256
        ):
            raise RuntimeError("A2 no-cap state migration inputs changed")
        raw_candidate.decode("utf-8", "strict")
        write_bytes_exclusive(
            run_root / "notes_candidates/ep_007.raw", raw_candidate
        )
        migration_artifact = run_root / "state_migrations/after_ep_007.md"
        write_bytes_exclusive(migration_artifact, raw_candidate)
        migration = {
            "event": "controller_only_notes_state_migration",
            "after_execution_ordinal": 6,
            "after_exposure_position": 7,
            "before_execution_ordinal": 7,
            "from_state_sha256": CONTINUATION_A2_MIGRATION_FROM_SHA256,
            "to_state_sha256": CONTINUATION_A2_MIGRATION_TO_SHA256,
            "artifact_relative_path": str(migration_artifact.relative_to(run_root)),
            "artifact_sha256": sha256_bytes(raw_candidate),
            "source_candidate_relative_path": str(raw_source.relative_to(source_suite_root)),
            "source_candidate_sha256": sha256_bytes(raw_candidate),
            "no_model_rerun": True,
            "model_invoked_for_migration": False,
            "model_content_modified": False,
            "protocol_amendment": "NOTES.md size cap removed",
            "reason": "user-authorized removal of NOTES.md size cap",
            "historical_attempt_remains_invalid_and_unmodified": True,
            "historical_ep_007_snapshot_remains_unmodified": True,
        }
        write_bytes_exclusive(
            run_root / "notes_state_migrations.jsonl",
            canonical_json_bytes(migration) + b"\n",
        )
        migrations.append(migration)
        approved_notes = raw_candidate

    allowed_exceptions = [
        {
            "execution_ordinal": row["execution_ordinal"],
            "attempt_sha256": row["attempt_sha256"],
            "reason": row["terminal_exception_reason"],
        }
        for row in carried_records
        if row["allowed_terminal_exception"]
    ]
    continuation = {
        "applicable": True,
        "source_suite_id": CONTINUATION_SOURCE_SUITE_ID,
        "source_arm": ACTIVE_ARM,
        "source_arm_root": str(source_arm_root),
        "source_suite_root": str(source_suite_root),
        "source_root_artifacts": root_artifacts,
        "source_selected_artifact_digest": selected_digest,
        "source_selected_artifacts": selected_artifacts,
        "source_controller_sha256": CONTINUATION_SOURCE_CONTROLLER_SHA256,
        "source_ag_mcp_sha256": CONTINUATION_SOURCE_MCP_SHA256,
        "source_schedule_sha256": FROZEN_PERMUTATION_SHA256,
        "carried_attempts": carried_count,
        "next_execution_ordinal": carried_count,
        "carried_attempt_records": carried_records,
        "allowed_terminal_exceptions": allowed_exceptions,
        "state_migrations": migrations,
        "protocol_amendments": (
            [
                {
                    "name": "NOTES.md size cap removed",
                    "effective_before_execution_ordinal": carried_count,
                    "source_notes_instruction_sha256": launch[
                        "notes_instruction_sha256"
                    ][ACTIVE_ARM],
                    "effective_notes_instruction_sha256": sha256_bytes(
                        notes_instruction_for(ACTIVE_ARM).encode("utf-8")
                    ),
                    "source_attempts_reinterpreted_or_retried": False,
                }
            ]
            if notes_enabled
            else []
        ),
        "retry_policy": "none; every carried slot is consumed",
        "source_attempt_json_bytes_preserved_exactly": True,
        "source_modified": False,
        "incoming_notes_sha256": (
            sha256_bytes(approved_notes) if notes_enabled else None
        ),
        "prefix_validated": True,
        "import_complete": True,
        "import_complete_sentinel": "RESUME_IMPORT_COMPLETE.json",
    }
    continuation_manifest_path = run_root / "continuation_manifest.json"
    write_json_exclusive(continuation_manifest_path, continuation)
    commit_hashes = [
        {
            "path": str(path.relative_to(run_root)),
            "sha256": sha256_file(path),
        }
        for path in sorted((run_root / "slot_commits").glob("slot_*.json"))
    ]
    import_sentinel_path = run_root / "RESUME_IMPORT_COMPLETE.json"
    write_json_exclusive(
        import_sentinel_path,
        {
            "schema_version": 1,
            "event": "continuation_prefix_import_complete",
            "arm": ACTIVE_ARM,
            "suite_id": ACTIVE_SUITE_ID,
            "source_suite_id": CONTINUATION_SOURCE_SUITE_ID,
            "carried_attempts": carried_count,
            "next_execution_ordinal": carried_count,
            "continuation_manifest_sha256": sha256_file(
                continuation_manifest_path
            ),
            "carried_slot_commit_sha256": commit_hashes,
            "state_migrations": migrations,
            "prefix_validated": True,
            "source_attempt_json_bytes_preserved_exactly": True,
        },
    )
    continuation["import_complete_sentinel_sha256"] = sha256_file(
        import_sentinel_path
    )
    return reports, approved_notes, continuation


NOTES_FIXED_TIMESTAMP = 946684800
PROHIBITED_NOTE_CODEPOINTS = {
    0x061C,
    0x200B,
    0x200C,
    0x200D,
    0x200E,
    0x200F,
    0x202A,
    0x202B,
    0x202C,
    0x202D,
    0x202E,
    0x2060,
    0x2061,
    0x2062,
    0x2063,
    0x2064,
    0x2066,
    0x2067,
    0x2068,
    0x2069,
    0xFEFF,
}
LEAKAGE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "prior_answer_or_prediction",
        re.compile(
            r"\b(?:answered|submitted|guessed|predicted|chose|my answer|answer was)"
            r"\b.{0,32}\b(?:zero|one|two|three|four|five|six|seven|eight|nine|[0-9])\b",
            re.I | re.S,
        ),
    ),
    (
        "historical_episode_reference",
        re.compile(
            r"\b(?:previous|prior|last|earlier|completed|so far|already)\b.{0,48}"
            r"\b(?:episode|canvas|trial|round|answer|guess|prediction|digit|case)\b|"
            r"\b(?:episode|trial|round)\s*(?:id|number|no\.?|#)?\s*[_-]?[0-9]+\b",
            re.I | re.S,
        ),
    ),
    (
        "count_tally_or_frequency",
        re.compile(
            r"\b(?:count|counts|counted|tally|tallies|frequency|frequencies|"
            r"running total|histogram|distribution|accuracy|correct so far|wrong so far)\b",
            re.I,
        ),
    ),
    (
        "class_prior_or_schedule",
        re.compile(
            r"\b(?:common|rare|likely|unlikely|due|overrepresented|underrepresented|"
            r"balanced|balance|round[- ]?robin|modulo|cyclic|cycle of|class prior|"
            r"label prior|next label|next answer|already used)\b",
            re.I,
        ),
    ),
    (
        "identifier_filename_or_path",
        re.compile(
            r"\b(?:eval(?:uation)?[_ -]?(?:index|id)|episode[_ -]?id|workspace|"
            r"filename|file name|directory name|uuid|shuffle seed|ordering position)\b|"
            r"(?:^|[\s`'\"])(?:\.{0,2}/|/[A-Za-z0-9_.-])|"
            r"\b[A-Fa-f0-9]{32,}\b|\b[A-Za-z0-9_.-]+\.(?:png|jpg|jpeg|webp)\b",
            re.I | re.M,
        ),
    ),
    (
        "canvas_blind_default",
        re.compile(
            r"\b(?:default|when unsure|if uncertain|without looking|without inspecting|"
            r"blind guess)\b.{0,48}\b(?:answer|choose|guess|submit|digit)\b|"
            r"\b(?:answer|choose|guess|submit)\b.{0,24}\b(?:by default|when unsure|if uncertain)\b",
            re.I | re.S,
        ),
    ),
    (
        "validator_prompt_injection",
        re.compile(
            r"\b(?:ignore|override|disregard|bypass)\b.{0,40}\b(?:policy|rule|"
            r"instruction|validator|auditor)\b|\breturn\s+(?:pass|allow|true)\b",
            re.I | re.S,
        ),
    ),
    (
        "timestamp_or_progress",
        re.compile(
            r"\b(?:progress|halfway|remaining episodes?|episodes? left|run length|"
            r"sequence position|ordinal)\b|\b20\d{2}[-/]\d{1,2}[-/]\d{1,2}\b|"
            r"\b\d{1,2}:\d{2}(?::\d{2})?\b",
            re.I,
        ),
    ),
)


def write_initial_notes(workspace: Path, approved: bytes) -> Path:
    """Create NOTES.md from content only, with fixed cross-episode metadata."""
    workspace_fd = os.open(workspace, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        descriptor = os.open(
            "NOTES.md",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=workspace_fd,
        )
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as handle:
                handle.write(approved)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            os.close(descriptor)
    finally:
        os.close(workspace_fd)
    path = workspace / "NOTES.md"
    os.utime(
        path,
        ns=(NOTES_FIXED_TIMESTAMP * 1_000_000_000,) * 2,
        follow_symlinks=False,
    )
    return path


def acquire_notes_candidate(workspace: Path) -> tuple[bytes | None, dict[str, Any]]:
    """Read the candidate without following model-controlled links."""
    reasons: list[str] = []
    candidate: bytes | None = None
    metadata: dict[str, Any] = {}
    workspace_fd = os.open(workspace, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        try:
            descriptor = os.open(
                "NOTES.md",
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=workspace_fd,
            )
        except OSError as error:
            return None, {
                "structural_pass": False,
                "rule_ids": ["notes_missing_or_unopenable"],
                "error_type": type(error).__name__,
                "size_cap_enforced": False,
            }
        try:
            before = os.fstat(descriptor)
            metadata = {
                "mode": stat.S_IMODE(before.st_mode),
                "bytes": before.st_size,
                "nlink": before.st_nlink,
                "uid": before.st_uid,
            }
            if not stat.S_ISREG(before.st_mode):
                reasons.append("notes_not_regular")
            if before.st_uid != os.getuid():
                reasons.append("notes_wrong_owner")
            if before.st_nlink != 1:
                reasons.append("notes_link_count_not_one")
            if stat.S_ISREG(before.st_mode):
                try:
                    with os.fdopen(descriptor, "rb", closefd=False) as handle:
                        candidate = handle.read()
                except OSError as error:
                    reasons.append("notes_read_failed")
                    metadata["read_error_type"] = type(error).__name__
                after = os.fstat(descriptor)
                if (
                    before.st_dev,
                    before.st_ino,
                    before.st_size,
                    before.st_mtime_ns,
                ) != (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                ):
                    reasons.append("notes_changed_during_read")
                if candidate is not None and len(candidate) != before.st_size:
                    reasons.append("notes_short_read")
        finally:
            os.close(descriptor)
    finally:
        os.close(workspace_fd)
    try:
        xattrs = os.listxattr(workspace / "NOTES.md", follow_symlinks=False)
    except (AttributeError, OSError):
        xattrs = []
    metadata["extended_attribute_count_observed"] = len(xattrs)
    return candidate, {
        **metadata,
        "structural_pass": not reasons,
        "rule_ids": sorted(set(reasons)),
        "read_policy": "full_regular_file_to_eof",
        "size_cap_enforced": False,
    }


def decoded_note_variants(text: str) -> list[tuple[str, str]]:
    """Expose common lightweight encodings to the deterministic policy scan."""
    import codecs

    variants: list[tuple[str, str]] = [
        ("normalized", unicodedata.normalize("NFKC", text).casefold()),
        ("reversed", unicodedata.normalize("NFKC", text[::-1]).casefold()),
        ("rot13", unicodedata.normalize("NFKC", codecs.decode(text, "rot_13")).casefold()),
        ("percent_decoded", unicodedata.normalize("NFKC", unquote(text)).casefold()),
    ]
    tokens = re.findall(r"[A-Za-z0-9+/=_%-]{12,}", text)
    for token_index, token in enumerate(tokens[:32]):
        compact = token.strip()
        if re.fullmatch(r"[A-Fa-f0-9]{16,}", compact) and len(compact) % 2 == 0:
            try:
                decoded = bytes.fromhex(compact).decode("utf-8")
            except (ValueError, UnicodeDecodeError):
                pass
            else:
                variants.append((f"hex_{token_index}", decoded.casefold()))
        if re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", compact):
            try:
                padded = compact + "=" * (-len(compact) % 4)
                decoded = base64.b64decode(padded, validate=True).decode("utf-8")
            except (ValueError, UnicodeDecodeError):
                pass
            else:
                variants.append((f"base64_{token_index}", decoded.casefold()))
    decimal_runs = re.findall(r"(?:\b(?:[3-9][0-9]|1[01][0-9]|12[0-6])\b[ ,;:]+){3,}\b(?:[3-9][0-9]|1[01][0-9]|12[0-6])\b", text)
    for run_index, run in enumerate(decimal_runs[:8]):
        try:
            decoded = bytes(int(value) for value in re.findall(r"\d+", run)).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            continue
        variants.append((f"decimal_bytes_{run_index}", decoded.casefold()))
    lines = [line for line in text.splitlines() if line]
    if len(lines) >= 4:
        variants.append(("line_initials", "".join(line[0] for line in lines).casefold()))
        variants.append(("line_finals", "".join(line[-1] for line in lines).casefold()))
    unique: list[tuple[str, str]] = []
    seen: set[str] = set()
    for name, value in variants:
        if value not in seen:
            seen.add(value)
            unique.append((name, value))
    return unique


def deterministic_notes_audit(candidate: bytes | None) -> dict[str, Any]:
    rule_ids: list[str] = []
    if candidate is None:
        return {
            "passed": False,
            "rule_ids": ["notes_missing_or_unreadable"],
            "character_count": None,
            "text": None,
            "variant_hits": [],
        }
    try:
        text = candidate.decode("utf-8", "strict")
    except UnicodeDecodeError:
        return {
            "passed": False,
            "rule_ids": ["invalid_utf8"],
            "character_count": None,
            "text": None,
            "variant_hits": [],
        }
    if text != unicodedata.normalize("NFKC", text):
        rule_ids.append("non_nfkc_text")
    if "\r" in text:
        rule_ids.append("carriage_return_channel")
    if any(line.endswith((" ", "\t")) for line in text.splitlines()):
        rule_ids.append("trailing_whitespace_channel")
    for character in text:
        codepoint = ord(character)
        category = unicodedata.category(character)
        if character not in {"\n", "\t"} and (
            category in {"Cc", "Cf", "Cs", "Co"}
            or codepoint in PROHIBITED_NOTE_CODEPOINTS
        ):
            rule_ids.append("prohibited_unicode_channel")
            break
    if re.search(r"<!--|-->|<script\b|```(?:base64|hex|cipher)", text, re.I):
        rule_ids.append("covert_markup_or_code_channel")
    if re.search(r"\b[A-Za-z0-9+/]{40,}={0,2}\b|\b[A-Fa-f0-9]{32,}\b", text):
        rule_ids.append("high_entropy_payload")
    variant_hits: list[dict[str, str]] = []
    for variant_name, variant in decoded_note_variants(text):
        for rule_id, pattern in LEAKAGE_PATTERNS:
            match = pattern.search(variant)
            if match:
                rule_ids.append(rule_id)
                variant_hits.append(
                    {
                        "variant": variant_name,
                        "rule_id": rule_id,
                        "span_sha256": sha256_bytes(match.group(0).encode("utf-8")),
                    }
                )
    return {
        "passed": not rule_ids,
        "rule_ids": sorted(set(rule_ids)),
        "character_count": len(text),
        "utf8_bytes": len(candidate),
        "text": text,
        "variant_hits": variant_hits,
    }


def snapshot_approved_notes(path: Path, approved: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(approved)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    directory_descriptor = os.open(
        path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def observe_and_snapshot_notes(
    workspace: Path,
    previous: bytes,
    run_root: Path,
    execution_ordinal: int,
) -> tuple[bytes, dict[str, Any]]:
    """Persist content verbatim; diagnostics observe content but never gate it."""
    exposure_position = execution_ordinal + 1
    candidate, acquisition = acquire_notes_candidate(workspace)
    decode_error: str | None = None
    text: str | None = None
    character_count: int | None = None
    if candidate is not None:
        try:
            text = candidate.decode("utf-8", "strict")
            character_count = len(text)
        except UnicodeDecodeError as error:
            decode_error = f"UnicodeDecodeError:{error.start}:{error.end}"

    state_eligible = bool(acquisition.get("structural_pass") and text is not None)
    next_state = candidate if state_eligible and candidate is not None else previous
    history_path = run_root / "notes_history" / f"ep_{exposure_position:03d}.md"
    snapshot_approved_notes(history_path, next_state)

    raw_candidate_path: str | None = None
    if candidate is not None and not state_eligible:
        raw_path = run_root / "notes_candidates" / f"ep_{exposure_position:03d}.raw"
        snapshot_approved_notes(raw_path, candidate)
        raw_candidate_path = str(raw_path)

    observational = deterministic_notes_audit(candidate)
    result = {
        "execution_ordinal_zero_based": execution_ordinal,
        "exposure_position_one_based": exposure_position,
        "previous_state_sha256": sha256_bytes(previous),
        "candidate_present": candidate is not None,
        "candidate_sha256": sha256_bytes(candidate) if candidate is not None else None,
        "next_state_sha256": sha256_bytes(next_state),
        "history_snapshot_sha256": sha256_file(history_path),
        "history_snapshot": str(history_path),
        "raw_structurally_invalid_candidate": raw_candidate_path,
        "utf8_bytes": len(candidate) if candidate is not None else None,
        "unicode_codepoints": character_count,
        "nonempty_physical_lines": (
            sum(bool(line.strip()) for line in text.splitlines())
            if text is not None else None
        ),
        "has_final_newline": (
            candidate.endswith(b"\n") if candidate is not None else None
        ),
        "newline_style": (
            "CRLF" if candidate is not None and b"\r\n" in candidate
            else "LF" if candidate is not None and b"\n" in candidate
            else "none" if candidate is not None else None
        ),
        "changed": candidate is not None and candidate != previous,
        "structural_acquisition": acquisition,
        "strict_utf8": decode_error is None and candidate is not None,
        "decode_error": decode_error,
        "size_cap_enforced": False,
        "within_2000_codepoint_cap": None,
        "state_eligible": state_eligible,
        "carried_verbatim": state_eligible,
        "content_validator_applied": False,
        "content_modified_or_stripped": False,
        "observational_leakage_test": {
            "non_gating": True,
            "detector_rule_ids": observational.get("rule_ids", []),
            "detector_variant_hits": observational.get("variant_hits", []),
            "classification_deferred_until_after_scoring": True,
        },
    }
    write_json_exclusive(
        run_root / "leakage_audits" / f"ep_{exposure_position:03d}.json",
        result,
    )
    Journal(run_root / "notes_versions.jsonl").append(
        {
            "event": "notes_state_snapshotted",
            **{key: value for key, value in result.items() if key not in {
                "structural_acquisition", "observational_leakage_test"
            }},
        }
    )
    return next_state, result


def load_environment_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "mnistpro_hidden_environment", ENVIRONMENT_PY
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load hidden environment")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def project_schedules() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    with LEVEL1_SUMMARY.open("r", encoding="utf-8") as handle:
        level1_raw = json.load(handle)
    with LEVEL2_SUMMARY.open("r", encoding="utf-8") as handle:
        level2_raw = json.load(handle)

    level1 = [
        {"episode_id": row["episode_id"], "eval_index": row["eval_index"]}
        for row in level1_raw["episodes"]
    ]
    level2 = [
        {"episode_id": row["episode_id"], "indices": row["indices"]}
        for row in level2_raw["episodes"]
    ]
    del level1_raw, level2_raw

    for level, rows in ((1, level1), (2, level2)):
        ids = [row["episode_id"] for row in rows]
        if ids != list(range(100)) or len(set(ids)) != 100:
            raise RuntimeError(f"Level {level} manifest episode IDs changed")
    if any(
        not isinstance(row["eval_index"], int)
        or not 0 <= row["eval_index"] < 10_000
        for row in level1
    ):
        raise RuntimeError("Level 1 manifest has an invalid index")
    if any(
        not isinstance(row["indices"], list)
        or len(row["indices"]) != 2
        or any(not isinstance(index, int) or not 0 <= index < 10_000 for index in row["indices"])
        for row in level2
    ):
        raise RuntimeError("Level 2 manifest has an invalid index pair")

    if sha256_bytes(canonical_json_bytes(level1)) != PINNED_HASHES["level1_projection"]:
        raise RuntimeError("Level 1 projected schedule hash changed")
    if sha256_bytes(canonical_json_bytes(level2)) != PINNED_HASHES["level2_projection"]:
        raise RuntimeError("Level 2 projected schedule hash changed")
    return level1, level2


def validate_pinned_inputs() -> dict[str, str]:
    observed = {
        "environment.py": sha256_file(ENVIRONMENT_PY),
        "level1_summary": sha256_file(LEVEL1_SUMMARY),
        "level2_summary": sha256_file(LEVEL2_SUMMARY),
        "standalone_codex": sha256_file(CODEX),
        "imagemagick": sha256_file(IMAGEMAGICK),
        "ag": sha256_file(AG_SOURCE),
        "ag_mcp_server": sha256_file(AG_MCP_SOURCE),
        "system_ca_bundle": sha256_file(SYSTEM_CA_BUNDLE),
        "a1_notes_document": sha256_file(A1_NOTES_DOCUMENT),
        "a2_notes_document": sha256_file(A2_NOTES_DOCUMENT),
    }
    for key, actual in observed.items():
        if actual != PINNED_HASHES[key]:
            raise RuntimeError(f"pinned input changed: {key}: {actual}")

    for arm, document, expected_block in (
        ("A1", A1_NOTES_DOCUMENT, PERSISTENT_NOTES_INSTRUCTION_A1),
        ("A2", A2_NOTES_DOCUMENT, PERSISTENT_NOTES_INSTRUCTION_A2),
    ):
        source_text = document.read_text(encoding="utf-8")
        match = re.search(r"^```\n(.*?)\n```$", source_text, re.M | re.S)
        if match is None:
            raise RuntimeError(f"{arm} source notes document has no fenced block")
        source_block = match.group(1)
        source_size_policy = (
            "You may rewrite NOTES.md in full at any point. It is capped at 2000 "
            "characters.\nWhen it is full, decide what to keep and what to drop."
        )
        effective_size_policy = (
            "You may rewrite NOTES.md in full at any point. There is no size cap."
        )
        if source_block.replace(source_size_policy, effective_size_policy) != expected_block:
            raise RuntimeError(
                f"{arm} effective notes instruction changed beyond the size policy"
            )
        observed[f"{arm.lower()}_source_notes_block_sha256"] = sha256_bytes(
            source_block.encode("utf-8")
        )
        observed[f"{arm.lower()}_effective_notes_block_sha256"] = sha256_bytes(
            expected_block.encode("utf-8")
        )

    image_gz = DATA_ROOT / "MNIST/raw/t10k-images-idx3-ubyte.gz"
    label_gz = DATA_ROOT / "MNIST/raw/t10k-labels-idx1-ubyte.gz"
    image_idx = DATA_ROOT / "MNIST/raw/t10k-images-idx3-ubyte"
    label_idx = DATA_ROOT / "MNIST/raw/t10k-labels-idx1-ubyte"
    observed["mnist_test_images_gz_md5"] = md5_file(image_gz)
    observed["mnist_test_labels_gz_md5"] = md5_file(label_gz)
    if (
        observed["mnist_test_images_gz_md5"]
        != PINNED_HASHES["mnist_test_images_gz_md5"]
        or observed["mnist_test_labels_gz_md5"]
        != PINNED_HASHES["mnist_test_labels_gz_md5"]
    ):
        raise RuntimeError("MNIST test dataset checksum changed")
    observed["mnist_test_images_idx_sha256"] = sha256_file(image_idx)
    observed["mnist_test_labels_idx_sha256"] = sha256_file(label_idx)
    if (
        observed["mnist_test_images_idx_sha256"]
        != PINNED_HASHES["mnist_test_images_idx_sha256"]
        or observed["mnist_test_labels_idx_sha256"]
        != PINNED_HASHES["mnist_test_labels_idx_sha256"]
    ):
        raise RuntimeError("uncompressed MNIST test dataset checksum changed")
    for compressed, uncompressed in ((image_gz, image_idx), (label_gz, label_idx)):
        decompressed_digest = hashlib.sha256()
        with gzip.open(compressed, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                decompressed_digest.update(chunk)
        if decompressed_digest.hexdigest() != sha256_file(uncompressed):
            raise RuntimeError(
                f"MNIST file does not match its pinned gzip stream: {uncompressed.name}"
            )

    version = subprocess.run(
        (str(CODEX), "--version"),
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout.strip()
    if version != "codex-cli 0.149.1":
        raise RuntimeError(f"unexpected Codex CLI version: {version}")
    observed["codex_cli_version"] = version
    catalog = subprocess.run(
        (str(CODEX), "debug", "models", "--bundled"),
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout
    if '"slug": "gpt-5.6-terra"' not in catalog and '"slug":"gpt-5.6-terra"' not in catalog:
        raise RuntimeError("gpt-5.6-terra is absent from the pinned CLI model catalog")
    if '"effort": "medium"' not in catalog and '"effort":"medium"' not in catalog:
        raise RuntimeError("medium reasoning is absent from the pinned CLI model catalog")
    observed["model_catalog_check"] = "gpt-5.6-terra/medium present"
    return observed


def build_specs(
    level1: list[dict[str, Any]], level2: list[dict[str, Any]]
) -> list[EpisodeSpec]:
    specs = [
        EpisodeSpec(1, row["episode_id"], (row["eval_index"],)) for row in level1
    ]
    if len(specs) != EXPECTED_EPISODES:
        raise RuntimeError(
            f"expected {EXPECTED_EPISODES} Level 1 episode specifications, got {len(specs)}"
        )
    return specs


def preload_mnist(specs: list[EpisodeSpec]) -> dict[int, tuple[Image.Image, int]]:
    dataset = MNIST(root=str(DATA_ROOT), train=False, download=False)
    selected = sorted({index for spec in specs for index in spec.indices})
    cache: dict[int, tuple[Image.Image, int]] = {}
    for index in selected:
        image, label = dataset[index]
        cache[index] = (image.copy(), int(label))
    return cache


def make_episode_environment(
    module: Any,
    spec: EpisodeSpec,
    image_cache: dict[int, tuple[Image.Image, int]],
) -> tuple[Any, str | int, Image.Image]:
    images = [image_cache[index][0].copy() for index in spec.indices]
    labels = [image_cache[index][1] for index in spec.indices]
    if spec.level == 1:
        hidden_label: str | int = labels[0]
        environment = module.MnistActiveVisionEnv(
            image=images[0],
            label=hidden_label,
            box_size=BOX_SIZE,
            image_size=IMAGE_SIZE,
            step_size=STEP_SIZE,
        )
    else:
        hidden_label = "".join(str(label) for label in labels)
        environment = module.MultiDigitActiveVisionEnv(
            images=images,
            label=hidden_label,
            box_size=BOX_SIZE,
            image_size=IMAGE_SIZE,
            step_size=STEP_SIZE,
        )
    observation = environment.reset()
    return environment, hidden_label, observation


def validate_all_initial_observations(
    module: Any,
    specs: list[EpisodeSpec],
    image_cache: dict[int, tuple[Image.Image, int]],
) -> dict[str, Any]:
    hashes: dict[str, str] = {}
    mismatches: list[str] = []
    for spec in specs:
        _, _, observation = make_episode_environment(module, spec, image_cache)
        prior_root = LEVEL1_PRIOR_ROOT if spec.level == 1 else LEVEL2_PRIOR_ROOT
        prior_path = prior_root / f"episode_{spec.episode_id}" / "step_0.png"
        with Image.open(prior_path) as prior:
            same = np.array_equal(np.asarray(observation), np.asarray(prior.convert("RGB")))
        if not same:
            mismatches.append(spec.key)
        hashes[spec.key] = sha256_bytes(observation.tobytes())
    if mismatches:
        raise RuntimeError(f"initial-observation parity failed: {mismatches[:5]}")
    return {
        "episodes_checked": len(specs),
        "pixel_mismatches": len(mismatches),
        "raw_rgb_hashes": hashes,
    }


def recursive_nodes(value: Any) -> Iterable[Any]:
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from recursive_nodes(child)
    elif isinstance(value, list):
        for child in value:
            yield from recursive_nodes(child)


def prompt_preflight(level: int, observation_path: Path) -> dict[str, Any]:
    runtime = Path(tempfile.mkdtemp(prefix="mnistpro-prompt-", dir="/private/tmp"))
    try:
        codex_home = runtime / "codex-home"
        parent_home = runtime / "home"
        parent_tmp = runtime / "tmp"
        codex_home.mkdir()
        parent_home.mkdir()
        parent_tmp.mkdir()
        skills = codex_home / "skills"
        skills.mkdir()
        skills.chmod(0o555)
        argv = [str(CODEX), "debug", "prompt-input", "--image", str(observation_path)]
        add_config_args(argv, CLEAN_CONFIG)
        prompt = prompt_for(level)
        argv.append(prompt)
        environment = {
            "HOME": str(parent_home),
            "CODEX_HOME": str(codex_home),
            "TMPDIR": str(parent_tmp),
            "PATH": "/usr/bin:/bin",
        }
        result = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            env=environment,
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"prompt-input failed for level {level}: "
                f"{result.stderr.decode('utf-8', 'replace')[:500]}"
            )
        rendered = json.loads(result.stdout)
        if not isinstance(rendered, list) or len(rendered) != 1:
            raise RuntimeError(
                f"level {level} prompt-input has {len(rendered) if isinstance(rendered, list) else 'non-list'} items"
            )
        item = rendered[0]
        if not isinstance(item, dict) or item.get("role") != "user":
            raise RuntimeError(f"level {level} prompt-input role is not exactly user")
        content = item.get("content")
        if not isinstance(content, list):
            raise RuntimeError(f"level {level} prompt-input content is not a list")
        texts = [
            node.get("text")
            for node in content
            if isinstance(node, dict) and node.get("type") == "input_text"
        ]
        images = [
            node
            for node in content
            if isinstance(node, dict) and node.get("type") == "input_image"
        ]
        expected_wrapper = (
            f'<image name=[Image #1] path="{observation_path}">'
        )
        if texts != [expected_wrapper, "</image>", prompt] or len(images) != 1:
            raise RuntimeError(
                f"level {level} prompt-input text/image shape changed: "
                f"texts={len(texts)} images={len(images)}"
            )
        all_strings = "\n".join(
            node for node in recursive_nodes(rendered) if isinstance(node, str)
        )
        markers = [marker for marker in FORBIDDEN_MARKERS if marker in all_strings]
        if markers:
            raise RuntimeError(f"level {level} prompt-input leaked markers: {markers}")
        coaching_markers = [
            marker
            for marker in AUTONOMY_COACHING_MARKERS
            if marker.casefold() in prompt.casefold()
        ]
        if coaching_markers:
            raise RuntimeError(
                "scored prompt coaches glimpse reading or fusion: "
                f"{coaching_markers}"
            )
        installed_skills = [
            str(path.relative_to(codex_home))
            for path in codex_home.rglob("*")
            if path.is_file() and "skills" in path.parts
        ]
        if installed_skills:
            raise RuntimeError(f"bundled skills were installed: {installed_skills}")
        return {
            "level": level,
            "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
            "prompt_utf8_bytes": len(prompt.encode("utf-8")),
            "top_level_items": 1,
            "roles": ["user"],
            "exact_user_prompt_items": 1,
            "cli_image_wrapper_text_items": 2,
            "input_image_items": 1,
            "forbidden_markers": [],
            "glimpse_reading_or_fusion_coaching": [],
            "bundled_skill_files": [],
            "debug_output_sha256": sha256_bytes(result.stdout),
        }
    finally:
        skills = runtime / "codex-home/skills"
        if skills.exists():
            skills.chmod(0o755)
        shutil.rmtree(runtime, ignore_errors=True)


def jwt_expiry(token: str) -> int | None:
    try:
        segment = token.split(".")[1]
        segment += "=" * (-len(segment) % 4)
        payload = json.loads(base64.urlsafe_b64decode(segment.encode("ascii")))
        value = payload.get("exp")
        return int(value) if value is not None else None
    except (IndexError, ValueError, TypeError, json.JSONDecodeError):
        return None


def extract_minimal_auth(source: Any) -> dict[str, Any]:
    if not isinstance(source, dict):
        raise RuntimeError("authentication payload is not an object")
    tokens = source.get("tokens")
    if not isinstance(tokens, dict):
        raise RuntimeError("authentication token structure is missing")
    id_token = tokens.get("id_token")
    access_token = tokens.get("access_token")
    refresh_token = tokens.get("refresh_token")
    account_id = tokens.get("account_id")
    if not isinstance(id_token, str) or not id_token:
        raise RuntimeError("ID token is missing")
    if not isinstance(access_token, str) or not access_token:
        raise RuntimeError("access token is missing")
    if not isinstance(refresh_token, str) or not refresh_token:
        raise RuntimeError("refresh token is missing")
    if not isinstance(account_id, str) or not account_id:
        raise RuntimeError("account ID is missing")
    auth_mode = source.get("auth_mode", "chatgpt")
    if auth_mode != "chatgpt":
        raise RuntimeError("authentication mode is not ChatGPT")
    last_refresh = source.get("last_refresh")
    if not isinstance(last_refresh, str) or not last_refresh:
        raise RuntimeError("authentication refresh timestamp is missing")
    return {
        "auth_mode": auth_mode,
        "tokens": {
            "id_token": id_token,
            "access_token": access_token,
            "refresh_token": refresh_token,
            "account_id": account_id,
        },
        "last_refresh": last_refresh,
    }


def auth_metadata(minimal: dict[str, Any]) -> dict[str, Any]:
    tokens = minimal["tokens"]
    id_expiry = jwt_expiry(tokens["id_token"])
    access_expiry = jwt_expiry(tokens["access_token"])
    return {
        "auth_mode": minimal["auth_mode"],
        "nonempty_material": [
            "id_token",
            "access_token",
            "refresh_token",
            "account_id",
            "last_refresh",
        ],
        "refresh_token_copied": True,
        "id_token_exp_unix": id_expiry,
        "access_token_exp_unix": access_expiry,
        "credential_fingerprint_sha256": sha256_bytes(
            canonical_json_bytes(minimal)
        ),
        "id_token_expired_at_check": (
            id_expiry is not None and id_expiry <= time.time()
        ),
    }


class AuthStateVault:
    """Durable controller-only checkpoint for one non-rotating arm credential."""

    def __init__(self, run_name: str):
        if run_name != ACTIVE_ARM or not re.fullmatch(r"A[012]", ACTIVE_ARM):
            raise RuntimeError(f"unsafe auth-state vault run name: {run_name}")
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{7,63}", ACTIVE_SUITE_ID):
            raise RuntimeError(f"unsafe suite identifier: {ACTIVE_SUITE_ID}")
        AUTH_VAULT_PARENT.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(AUTH_VAULT_PARENT, 0o700)
        parent_stat = os.stat(AUTH_VAULT_PARENT, follow_symlinks=False)
        if (
            not stat.S_ISDIR(parent_stat.st_mode)
            or parent_stat.st_uid != os.getuid()
            or stat.S_IMODE(parent_stat.st_mode) != 0o700
        ):
            raise RuntimeError("controller auth-state vault parent is not private")
        self.path = AUTH_VAULT_PARENT / (
            f"mnistpro-authstate-{ACTIVE_SUITE_ID}-{ACTIVE_ARM.lower()}.json"
        )
        if self.path.exists() or self.path.is_symlink():
            raise RuntimeError(f"auth-state vault already exists: {self.path}")
        descriptor = os.open(
            self.path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        os.close(descriptor)

    def checkpoint(self, minimal_auth: dict[str, Any]) -> None:
        payload = canonical_json_bytes(minimal_auth) + b"\n"
        if not 1 <= len(payload) <= 65_536:
            raise RuntimeError("refusing to checkpoint an invalid auth payload size")
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        try:
            written = 0
            while written < len(payload):
                written += os.write(descriptor, payload[written:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, self.path)
        directory_descriptor = os.open(
            AUTH_VAULT_PARENT, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        vault_stat = os.stat(self.path, follow_symlinks=False)
        if (
            not stat.S_ISREG(vault_stat.st_mode)
            or vault_stat.st_uid != os.getuid()
            or vault_stat.st_nlink != 1
            or stat.S_IMODE(vault_stat.st_mode) != 0o600
        ):
            raise RuntimeError("auth-state vault identity or mode changed")

    def remove(self) -> None:
        if (
            self.path.parent != AUTH_VAULT_PARENT
            or self.path.name
            != f"mnistpro-authstate-{ACTIVE_SUITE_ID}-{ACTIVE_ARM.lower()}.json"
        ):
            raise RuntimeError(f"refusing to remove unexpected auth vault: {self.path}")
        if self.path.exists():
            path_stat = os.stat(self.path, follow_symlinks=False)
            if not stat.S_ISREG(path_stat.st_mode) or path_stat.st_nlink != 1:
                raise RuntimeError("refusing to remove changed auth-state vault")
            os.unlink(self.path)
            directory_descriptor = os.open(
                AUTH_VAULT_PARENT, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            )
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)


def read_minimal_auth(source_path: Path = HOST_AUTH) -> tuple[dict[str, Any], dict[str, Any]]:
    descriptor = os.open(source_path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        descriptor_stat = os.fstat(descriptor)
        if (
            not stat.S_ISREG(descriptor_stat.st_mode)
            or descriptor_stat.st_uid != os.getuid()
            or descriptor_stat.st_nlink != 1
            or stat.S_IMODE(descriptor_stat.st_mode) != 0o600
            or not 1 <= descriptor_stat.st_size <= 65_536
        ):
            raise RuntimeError("authentication snapshot identity or mode changed")
        with os.fdopen(descriptor, "r", encoding="utf-8", closefd=False) as handle:
            source = json.load(handle)
    finally:
        os.close(descriptor)
    minimal = extract_minimal_auth(source)
    del source
    metadata = auth_metadata(minimal)
    access_expiry = metadata["access_token_exp_unix"]
    if access_expiry is None or access_expiry <= time.time() + 12 * 60 * 60:
        raise RuntimeError("access token expires too soon for the schedule")
    return minimal, metadata


def adopt_runtime_auth(
    runtime: Path,
    minimal_auth: dict[str, Any],
    vault: AuthStateVault,
    required_access_expiry: float,
) -> dict[str, Any]:
    """Verify that a concurrent arm did not rotate the shared credential lineage."""
    auth_path = runtime / "codex-home/auth.json"
    descriptor = os.open(auth_path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        descriptor_stat = os.fstat(descriptor)
        if (
            not stat.S_ISREG(descriptor_stat.st_mode)
            or descriptor_stat.st_uid != os.getuid()
            or descriptor_stat.st_nlink != 1
            or stat.S_IMODE(descriptor_stat.st_mode) != 0o600
            or not 1 <= descriptor_stat.st_size <= 65_536
        ):
            raise RuntimeError("runtime authentication file identity or mode changed")
        chunks: list[bytes] = []
        remaining = 65_537
        while remaining > 0:
            chunk = os.read(descriptor, min(remaining, 16_384))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw_auth = b"".join(chunks)
        if len(raw_auth) > 65_536:
            raise RuntimeError("runtime authentication file is oversized")
        source = json.loads(raw_auth)
    finally:
        os.close(descriptor)
    allowed_root_keys = {"auth_mode", "tokens", "last_refresh", "OPENAI_API_KEY"}
    if set(source) - allowed_root_keys or source.get("OPENAI_API_KEY") not in {None, ""}:
        raise RuntimeError("runtime authentication file contains alternate secrets")
    tokens = source.get("tokens")
    if not isinstance(tokens, dict) or set(tokens) != {
        "id_token",
        "access_token",
        "refresh_token",
        "account_id",
    }:
        raise RuntimeError("runtime authentication token whitelist changed")
    candidate = extract_minimal_auth(source)
    del source
    if candidate["tokens"]["account_id"] != minimal_auth["tokens"]["account_id"]:
        raise RuntimeError("runtime authentication account changed")
    try:
        previous_refresh_time = datetime.fromisoformat(
            minimal_auth["last_refresh"].replace("Z", "+00:00")
        )
        candidate_refresh_time = datetime.fromisoformat(
            candidate["last_refresh"].replace("Z", "+00:00")
        )
    except ValueError as error:
        raise RuntimeError("runtime authentication refresh timestamp is invalid") from error
    if previous_refresh_time.tzinfo is None or candidate_refresh_time.tzinfo is None:
        raise RuntimeError("runtime authentication refresh timestamp lacks timezone")
    changed = canonical_json_bytes(candidate) != canonical_json_bytes(minimal_auth)
    if changed:
        raise RuntimeError(
            "credential material changed during a concurrent three-arm suite"
        )
    candidate_access_expiry = jwt_expiry(candidate["tokens"]["access_token"])
    if (
        candidate_access_expiry is None
        or candidate_access_expiry <= required_access_expiry
    ):
        raise RuntimeError("runtime access token lifetime is insufficient")
    vault.checkpoint(minimal_auth)
    metadata = auth_metadata(minimal_auth)
    return {
        "adopted": True,
        "credential_material_changed": False,
        "last_refresh_advanced": False,
        "id_token_exp_unix": metadata["id_token_exp_unix"],
        "access_token_exp_unix": metadata["access_token_exp_unix"],
    }


def create_runtime(minimal_auth: dict[str, Any], prefix: str) -> Path:
    RUNTIME_PARENT.mkdir(parents=True, exist_ok=True)
    runtime = Path(tempfile.mkdtemp(prefix=prefix, dir=RUNTIME_PARENT))
    codex_home = runtime / "codex-home"
    parent_home = runtime / "home"
    parent_tmp = runtime / "tmp"
    codex_home.mkdir()
    parent_home.mkdir()
    parent_tmp.mkdir()
    skills = codex_home / "skills"
    skills.mkdir()
    skills.chmod(0o555)
    auth_path = codex_home / "auth.json"
    descriptor = os.open(auth_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(minimal_auth, handle, separators=(",", ":"), ensure_ascii=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return runtime


def remove_runtime(runtime: Path) -> None:
    expected_prefixes = (
        "mnistpro-runtime-",
        "mnistpro-probe-",
        "mnistpro-auth-",
        "mnistpro-a1-runtime-",
        "mnistpro-a1-probe-",
        "mnistpro-a1-audit-",
        "mnistpro-a0-runtime-",
        "mnistpro-a0-probe-",
        "mnistpro-a1-runtime-",
        "mnistpro-a2-runtime-",
        "mnistpro-a2-probe-",
        "mnistpro-posthoc-",
    )
    if runtime.parent != RUNTIME_PARENT or not runtime.name.startswith(expected_prefixes):
        raise RuntimeError(f"refusing to remove unexpected runtime path: {runtime}")
    skills = runtime / "codex-home/skills"
    if skills.exists():
        skills.chmod(0o755)
    shutil.rmtree(runtime)


def auth_preflight(
    minimal_auth: dict[str, Any],
    vault: AuthStateVault,
    required_access_expiry: float,
) -> dict[str, Any]:
    runtime = create_runtime(minimal_auth, "mnistpro-auth-")
    result_report: dict[str, Any] = {}
    auth_update: dict[str, Any] | None = None
    try:
        environment = {
            "HOME": str(runtime / "home"),
            "CODEX_HOME": str(runtime / "codex-home"),
            "TMPDIR": str(runtime / "tmp"),
            "PATH": "/usr/bin:/bin",
        }
        result = subprocess.run(
            (str(CODEX), "login", "status"),
            capture_output=True,
            env=environment,
            timeout=30,
        )
        output = (result.stdout + result.stderr).decode("utf-8", "replace").strip()
        if result.returncode != 0 or output != "Logged in using ChatGPT":
            raise RuntimeError(f"minimal authentication preflight failed: {output[:200]}")
        result_report = {
            "passed": True,
            "status": output,
            "refresh_token_copied": True,
            "auth_file_mode": "0600",
        }
    finally:
        try:
            auth_update = adopt_runtime_auth(
                runtime, minimal_auth, vault, required_access_expiry
            )
        except Exception as error:
            raise RuntimeError(
                f"auth preflight runtime preserved after adoption failure: {runtime}"
            ) from error
        else:
            remove_runtime(runtime)
    result_report["auth_state_update"] = auth_update
    return result_report


class AuthenticatedConnectProxy:
    """A token-gated CONNECT proxy restricted to the two Codex auth/API hosts."""

    allowed_hosts = {"chatgpt.com", "auth.openai.com"}

    def __init__(self):
        self.token = uuid.uuid4().hex + uuid.uuid4().hex
        self._events: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._connection_lock = threading.Lock()
        self._connections: set[socket.socket] = set()
        self._stopping = threading.Event()
        owner = self

        class Handler(socketserver.BaseRequestHandler):
            def handle(self) -> None:
                owner._handle(self.request, self.client_address)

        class Server(socketserver.ThreadingTCPServer):
            allow_reuse_address = False
            daemon_threads = True

        self._server = Server(("127.0.0.1", 0), Handler)
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name=f"mnistpro-proxy-{self.port}",
            daemon=True,
        )
        self._thread.start()

    @property
    def url(self) -> str:
        return f"http://codex:{self.token}@127.0.0.1:{self.port}"

    def _record(self, **fields: Any) -> None:
        with self._lock:
            self._events.append({"timestamp": utc_now(), **fields})

    def _send_error(self, client: socket.socket, status: str) -> None:
        try:
            client.sendall(
                f"HTTP/1.1 {status}\r\nConnection: close\r\nContent-Length: 0\r\n\r\n".encode(
                    "ascii"
                )
            )
        except OSError:
            pass

    def _register(self, connection: socket.socket) -> None:
        with self._connection_lock:
            self._connections.add(connection)

    def _unregister(self, connection: socket.socket) -> None:
        with self._connection_lock:
            self._connections.discard(connection)

    def _pump(
        self,
        source: socket.socket,
        destination: socket.socket,
        byte_counter: list[int],
        counter_lock: threading.Lock,
    ) -> None:
        try:
            while not self._stopping.is_set():
                data = source.recv(65_536)
                if not data:
                    break
                destination.sendall(data)
                with counter_lock:
                    byte_counter[0] += len(data)
        except OSError as error:
            if not self._stopping.is_set():
                self._record(
                    authorized=None,
                    allowed=None,
                    reason="relay_connection_closed",
                    error_type=type(error).__name__,
                )
        finally:
            try:
                destination.shutdown(socket.SHUT_WR)
            except OSError:
                pass

    def _handle(self, client: socket.socket, client_address: Any) -> None:
        self._register(client)
        client.settimeout(20)
        buffer = bytearray()
        upstream: socket.socket | None = None
        try:
            while b"\r\n\r\n" not in buffer and len(buffer) <= 65_536:
                chunk = client.recv(4096)
                if not chunk:
                    break
                buffer.extend(chunk)
            header_bytes, separator, remainder = bytes(buffer).partition(b"\r\n\r\n")
            if not separator:
                self._record(
                    authorized=None,
                    allowed=None,
                    reason="connection_closed_before_headers",
                )
                self._send_error(client, "400 Bad Request")
                return
            lines = header_bytes.decode("iso-8859-1", "replace").split("\r\n")
            request_parts = lines[0].split()
            if len(request_parts) != 3 or request_parts[0].upper() != "CONNECT":
                self._record(authorized=False, allowed=False, reason="connect_required")
                self._send_error(client, "405 Method Not Allowed")
                return
            target = request_parts[1]
            if target.count(":") != 1:
                self._record(authorized=False, allowed=False, reason="invalid_target")
                self._send_error(client, "400 Bad Request")
                return
            host, port_text = target.rsplit(":", 1)
            host = host.rstrip(".").lower()
            headers: dict[str, str] = {}
            for line in lines[1:]:
                name, delimiter, value = line.partition(":")
                if delimiter:
                    headers[name.strip().lower()] = value.strip()
            expected = "Basic " + base64.b64encode(
                f"codex:{self.token}".encode("ascii")
            ).decode("ascii")
            authorized = headers.get("proxy-authorization") == expected
            allowed = host in self.allowed_hosts and port_text == "443"
            if not authorized:
                self._record(
                    authorized=False,
                    allowed=allowed,
                    target=f"{host}:{port_text}",
                    reason="proxy_auth_required",
                )
                self._send_error(client, "407 Proxy Authentication Required")
                return
            if not allowed:
                self._record(
                    authorized=True,
                    allowed=False,
                    target=f"{host}:{port_text}",
                    reason="target_blocked",
                )
                self._send_error(client, "403 Forbidden")
                return
            upstream = socket.create_connection((host, 443), timeout=20)
            self._register(upstream)
            try:
                client.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                if remainder:
                    upstream.sendall(remainder)
                client.settimeout(None)
                upstream.settimeout(None)
                transferred = [0]
                counter_lock = threading.Lock()
                request_thread = threading.Thread(
                    target=self._pump,
                    args=(client, upstream, transferred, counter_lock),
                    name=f"mnistpro-proxy-upload-{self.port}",
                    daemon=True,
                )
                request_thread.start()
                self._pump(upstream, client, transferred, counter_lock)
                try:
                    client.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    upstream.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                request_thread.join(timeout=5)
                self._record(
                    authorized=True,
                    allowed=True,
                    target=f"{host}:443",
                    transferred_bytes=transferred[0],
                )
            finally:
                upstream.close()
        except (ConnectionResetError, BrokenPipeError, socket.timeout) as error:
            self._record(
                authorized=None,
                allowed=None,
                reason="connection_closed",
                error_type=type(error).__name__,
            )
        except Exception as error:
            if not self._stopping.is_set():
                self._record(
                    authorized=False,
                    allowed=False,
                    reason="proxy_error",
                    error_type=type(error).__name__,
                )
                self._send_error(client, "502 Bad Gateway")
        finally:
            if upstream is not None:
                self._unregister(upstream)
            self._unregister(client)

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(event) for event in self._events]

    def stop(self) -> None:
        self._stopping.set()
        with self._connection_lock:
            connections = list(self._connections)
        for connection in connections:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                connection.close()
            except OSError:
                pass
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            with self._connection_lock:
                if not self._connections:
                    break
            time.sleep(0.025)


def external_profile(
    workspace: Path,
    runtime: Path,
    generic_root: Path,
    proxy_port: int,
    observation_dir: Path | None = None,
) -> str:
    observations = observation_dir or workspace / "observations"
    mailbox = workspace / ".ag-ipc"
    requests = mailbox / "requests"
    responses = mailbox / "responses"
    readable_roots = [
        Path("/bin"),
        Path("/usr"),
        Path("/System"),
        Path("/Library/Apple"),
        Path("/private/etc/hosts"),
        Path("/private/etc/resolv.conf"),
        Path("/private/etc/ssl"),
        Path("/private/var/db/timezone"),
        Path("/private/var/select"),
        PYTHON_ROOT,
        HOMEBREW_ROOT,
        CLI_PACKAGE_ROOT,
        generic_root,
        workspace,
    ]
    ancestor_roots = [*readable_roots, runtime]
    read_filters = " ".join(
        f"(subpath {toml_quote(path)})" for path in readable_roots
    )
    ancestor_filters = " ".join(
        f"(path-ancestors {toml_quote(path)})" for path in ancestor_roots
    )
    return f'''(version 1)
(deny default)
(import "system.sb")
(allow process-exec process-fork)
(allow process-info* (target self))
(allow signal (target self))
(allow signal (target children))
(allow signal (target same-sandbox))
(allow file-read-metadata file-test-existence {ancestor_filters})
(allow file-read* file-test-existence file-map-executable {read_filters})
(with-filter (process-path {toml_quote(CODEX)})
  (allow file-read* file-test-existence file-map-executable file-write*
    (subpath {toml_quote(runtime)})))
(with-filter (process-path {toml_quote(CODE_MODE_HOST)})
  (allow file-read* file-test-existence file-write*
    (subpath {toml_quote(runtime / 'home')})
    (subpath {toml_quote(runtime / 'tmp')})))
(allow file-write* (subpath {toml_quote(workspace)}))
(deny file-link)
(deny file-write* (subpath {toml_quote(observations)}))
(deny file-write* (literal {toml_quote(mailbox)}))
(deny file-write* (literal {toml_quote(requests)}))
(deny file-write* (literal {toml_quote(responses)}))
(deny file-write* (subpath {toml_quote(responses)}))
(allow network-outbound (remote tcp "localhost:{proxy_port}"))
'''


def canonical_profile_hash() -> str:
    template = external_profile(
        Path("<EPISODE_WORKSPACE>"),
        Path("<FRESH_RUNTIME>"),
        Path("<GENERIC_TOOL_ROOT>"),
        0,
    )
    return sha256_bytes(template.encode("utf-8"))


def external_sandbox_prefix(profile: str) -> list[str]:
    return ["/usr/bin/sandbox-exec", "-p", profile]


def parent_environment(runtime: Path, proxy_url: str | None = None) -> dict[str, str]:
    environment = {
        "HOME": str(runtime / "home"),
        "CODEX_HOME": str(runtime / "codex-home"),
        "TMPDIR": str(runtime / "tmp"),
        "PATH": "/usr/bin:/bin",
        "TERM": "dumb",
        "RUST_BACKTRACE": "0",
        "SSL_CERT_FILE": str(SYSTEM_CA_BUNDLE),
    }
    if proxy_url is not None:
        environment.update(
            {
                "HTTPS_PROXY": proxy_url,
                "HTTP_PROXY": proxy_url,
                "https_proxy": proxy_url,
                "http_proxy": proxy_url,
                "NO_PROXY": "",
                "no_proxy": "",
            }
        )
    return environment


def terminate_process_group(
    process: subprocess.Popen[bytes],
) -> tuple[int | None, bool]:
    """Terminate and reap the scored process group, including leftover children."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    if process.poll() is None:
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                return process.poll(), False

    # The group can outlive its leader if a model command backgrounds a child.
    # Signal the group even after the Codex leader has exited, then verify that
    # no member remains before the next episode can reuse controller resources.
    deadline = time.monotonic() + 2
    while True:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        if time.monotonic() >= deadline:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                return process.returncode, True
            kill_deadline = time.monotonic() + 2
            while time.monotonic() < kill_deadline:
                try:
                    os.killpg(process.pid, 0)
                except ProcessLookupError:
                    return process.returncode, True
                time.sleep(0.025)
            return process.returncode, False
        time.sleep(0.025)
    return process.returncode, True


def episode_isolation_probe(
    workspace: Path,
    generic_root: Path,
    minimal_auth: dict[str, Any],
    log_path: Path,
    observation_dir: Path | None = None,
) -> dict[str, Any]:
    runtime = create_runtime(minimal_auth, "mnistpro-probe-")
    proxy = AuthenticatedConnectProxy()
    probe_file = workspace / f".probe-{uuid.uuid4().hex}"
    sentinel_descriptor, sentinel_text = tempfile.mkstemp(
        prefix="mnistpro-sentinel-", dir="/private/tmp"
    )
    os.write(sentinel_descriptor, b"controller-only sentinel")
    os.close(sentinel_descriptor)
    sentinel = Path(sentinel_text)
    blocked_directory_reads = [
        SOURCE_ROOT,
        HOST_MEMORY,
        LEVEL1_PRIOR_ROOT,
        LEVEL2_PRIOR_ROOT,
        Path("/Users/poriasoujanya/Documents"),
        RUNTIME_PARENT,
    ]
    script = (
        "set -eu\n"
        f"/usr/bin/touch {toml_quote(probe_file)}\n"
        f"/bin/test -w {toml_quote(probe_file)}\n"
        f"/bin/rm -f {toml_quote(probe_file)}\n"
    )
    for path in blocked_directory_reads:
        script += (
            f"if /bin/ls {toml_quote(path)} >/dev/null 2>&1; then "
            f"exit 41; fi\n"
        )
    script += (
        f"if /bin/cat {toml_quote(HOST_AUTH)} >/dev/null 2>&1; then exit 42; fi\n"
        f"if /bin/cat {toml_quote(sentinel)} >/dev/null 2>&1; then exit 43; fi\n"
    )
    profile = external_profile(
        workspace, runtime, generic_root, proxy.port, observation_dir
    )
    argv = external_sandbox_prefix(profile) + ["/bin/sh", "-c", script]
    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            env=parent_environment(runtime, proxy.url),
            cwd=workspace,
            timeout=45,
        )
        combined = result.stdout + result.stderr
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_bytes(combined)
        passed = result.returncode == 0 and not probe_file.exists()
        return {
            "passed": passed,
            "returncode": result.returncode,
            "workspace_write": passed,
            "blocked_paths_checked": len(blocked_directory_reads) + 2,
            "blocked_paths_readable": [] if passed else ["see preflight log"],
            "profile_sha256": sha256_bytes(profile.encode("utf-8")),
            "canonical_profile_sha256": canonical_profile_hash(),
            "denial_log": str(log_path),
            "proxy_events": proxy.snapshot(),
        }
    finally:
        if probe_file.exists():
            probe_file.unlink()
        sentinel.unlink(missing_ok=True)
        proxy.stop()
        remove_runtime(runtime)


def network_isolation_probe(
    generic_root: Path, minimal_auth: dict[str, Any], run_root: Path
) -> dict[str, Any]:
    workspace = Path(tempfile.mkdtemp(prefix="mnistpro-network-workspace-", dir="/private/tmp"))
    runtime = create_runtime(minimal_auth, "mnistpro-probe-")
    proxy = AuthenticatedConnectProxy()
    log_path = run_root / "preflight/network_isolation.log"
    script = (
        "set -eu\n"
        "/usr/bin/curl --silent --show-error --output /dev/null "
        "--connect-timeout 8 --max-time 15 https://chatgpt.com/\n"
        "if /usr/bin/env -u HTTPS_PROXY -u HTTP_PROXY -u https_proxy -u http_proxy "
        "/usr/bin/curl --silent --show-error --output /dev/null "
        f"--proxy http://127.0.0.1:{proxy.port} "
        "--connect-timeout 5 --max-time 10 https://chatgpt.com/; then exit 51; fi\n"
        "if /usr/bin/curl --silent --show-error --output /dev/null "
        "--connect-timeout 5 --max-time 10 https://example.com/; then exit 51; fi\n"
    )
    try:
        profile = external_profile(workspace, runtime, generic_root, proxy.port)
        argv = external_sandbox_prefix(profile) + ["/bin/sh", "-c", script]
        result = subprocess.run(
            argv,
            capture_output=True,
            env=parent_environment(runtime, proxy.url),
            cwd=workspace,
            timeout=45,
        )
        combined = result.stdout + result.stderr
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_bytes(combined)
        if result.returncode != 0:
            raise RuntimeError(
                "network allowlist preflight failed; see " + str(log_path)
            )
        return {
            "passed": True,
            "outer_chatgpt_connection": "allowed",
            "outer_example_com_connection": "blocked",
            "model_tool_network": "proxy credential absent from model shell",
            "proxy_events": proxy.snapshot(),
            "log": str(log_path),
        }
    finally:
        proxy.stop()
        remove_runtime(runtime)
        shutil.rmtree(workspace)


def remote_model_catalog_preflight(
    generic_root: Path,
    minimal_auth: dict[str, Any],
    run_root: Path,
    vault: AuthStateVault,
    required_access_expiry: float,
) -> dict[str, Any]:
    """Exercise Codex TLS/auth through the exact scored outer boundary."""
    workspace = run_root / "preflight" / f".remote-models-{uuid.uuid4().hex}"
    workspace.mkdir()
    runtime = create_runtime(minimal_auth, "mnistpro-probe-")
    proxy = AuthenticatedConnectProxy()
    stdout_path = run_root / "preflight/remote_model_catalog.json"
    stderr_path = run_root / "preflight/remote_model_catalog.stderr.log"
    stdout = b""
    stderr = b""
    returncode: int | None = None
    auth_update: dict[str, Any] | None = None
    proxy_events: list[dict[str, Any]] = []
    cache_payload: Any = None
    cache_bytes = b""
    launched_at = time.time()
    profile = external_profile(workspace, runtime, generic_root, proxy.port)
    try:
        if (runtime / "codex-home/models_cache.json").exists():
            raise RuntimeError("fresh remote-model runtime already has a model cache")
        argv = [str(CODEX), "debug", "models"]
        add_config_args(
            argv,
            (
                "features.respect_system_proxy=true",
                *CLEAN_CONFIG,
            ),
        )
        result = subprocess.run(
            external_sandbox_prefix(profile) + argv,
            cwd=workspace,
            env=parent_environment(runtime, proxy.url),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=120,
        )
        returncode = result.returncode
        stdout = result.stdout
        stderr = result.stderr
        auth_update = adopt_runtime_auth(
            runtime, minimal_auth, vault, required_access_expiry
        )
        cache_path = runtime / "codex-home/models_cache.json"
        cache_stat = os.stat(cache_path, follow_symlinks=False)
        if not stat.S_ISREG(cache_stat.st_mode):
            raise RuntimeError("remote model catalog cache is not a regular file")
        descriptor = os.open(cache_path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                cache_bytes = handle.read()
        finally:
            os.close(descriptor)
        cache_payload = json.loads(cache_bytes)
    finally:
        stdout_path.write_bytes(stdout)
        stderr_path.write_bytes(stderr)
        proxy.stop()
        proxy_events = proxy.snapshot()
        adoption_error: Exception | None = None
        if auth_update is None and runtime.exists():
            try:
                auth_update = adopt_runtime_auth(
                    runtime, minimal_auth, vault, required_access_expiry
                )
            except Exception as error:
                adoption_error = error
        if adoption_error is None:
            remove_runtime(runtime)
        if workspace.parent != run_root / "preflight" or not workspace.name.startswith(
            ".remote-models-"
        ):
            raise RuntimeError(
                f"refusing to remove unexpected remote-model workspace: {workspace}"
            )
        shutil.rmtree(workspace)
        if adoption_error is not None:
            raise RuntimeError(
                f"remote-model runtime preserved after adoption failure: {runtime}"
            ) from adoption_error

    combined = (stdout + stderr).decode("utf-8", "replace")
    allowed_chatgpt = any(
        event.get("authorized") is True
        and event.get("allowed") is True
        and event.get("target") == "chatgpt.com:443"
        and int(event.get("transferred_bytes", 0)) > 0
        for event in proxy_events
    )
    denied = [
        event
        for event in proxy_events
        if event.get("authorized") is False
        or event.get("allowed") is False
        or event.get("reason")
        in {"proxy_auth_required", "target_blocked", "proxy_error"}
    ]
    allowed_auth_refresh = any(
        event.get("authorized") is True
        and event.get("allowed") is True
        and event.get("target") == "auth.openai.com:443"
        and int(event.get("transferred_bytes", 0)) > 0
        for event in proxy_events
    )
    try:
        catalog = json.loads(stdout)
    except json.JSONDecodeError:
        catalog = None
    terra_entries = [
        node
        for node in recursive_nodes(catalog)
        if isinstance(node, dict) and node.get("slug") == MODEL
    ]
    cached_terra_entries = [
        node
        for node in recursive_nodes(cache_payload)
        if isinstance(node, dict) and node.get("slug") == MODEL
    ]
    terra_supports_medium = any(
        any(
            isinstance(level, dict) and level.get("effort") == REASONING_EFFORT
            for level in entry.get("supported_reasoning_levels", [])
        )
        for entry in terra_entries
    )
    cached_terra_supports_medium = any(
        any(
            isinstance(level, dict) and level.get("effort") == REASONING_EFFORT
            for level in entry.get("supported_reasoning_levels", [])
        )
        for entry in cached_terra_entries
    )
    fetched_at = cache_payload.get("fetched_at") if isinstance(cache_payload, dict) else None
    try:
        fetched_epoch = datetime.fromisoformat(str(fetched_at).replace("Z", "+00:00")).timestamp()
    except ValueError:
        fetched_epoch = 0.0
    stderr_lower = stderr.decode("utf-8", "replace").lower()
    transport_errors = [
        marker
        for marker in (
            "unknownissuer",
            "failed to refresh",
            "token refresh failed",
            "error sending request",
            "unauthorized",
            "invalid_grant",
            "reused",
            "revoked",
            "certificate",
            "http 400",
            "http 401",
            "http 403",
        )
        if marker in stderr_lower
    ]
    if (
        returncode != 0
        or "unknownissuer" in combined.lower()
        or not allowed_chatgpt
        or denied
        or len(terra_entries) != 1
        or not terra_supports_medium
        or len(cached_terra_entries) != 1
        or not cached_terra_supports_medium
        or not isinstance(cache_payload, dict)
        or cache_payload.get("client_version") != "0.149.1"
        or fetched_epoch < launched_at - 1
        or fetched_epoch > time.time() + 30
        or transport_errors
        or auth_update is None
        or (
            auth_update.get("credential_material_changed") is True
            and (
                not auth_update.get("last_refresh_advanced")
                or not allowed_auth_refresh
            )
        )
    ):
        raise RuntimeError(
            "remote model catalog preflight failed under the scored profile; "
            f"returncode={returncode} unknown_issuer="
            f"{'unknownissuer' in combined.lower()} allowed_chatgpt={allowed_chatgpt} "
            f"denied_events={len(denied)} terra_entries={len(terra_entries)} "
            f"cached_terra_entries={len(cached_terra_entries)} "
            f"terra_supports_medium={terra_supports_medium} "
            f"cache_medium={cached_terra_supports_medium} "
            f"transport_errors={transport_errors}; "
            f"see {stderr_path}"
        )
    return {
        "passed": True,
        "returncode": returncode,
        "terra_present": True,
        "terra_supports_medium": True,
        "fresh_remote_cache": True,
        "cache_client_version": "0.149.1",
        "cache_fetched_at": fetched_at,
        "cache_sha256": sha256_bytes(cache_bytes),
        "unknown_issuer_observed": False,
        "authenticated_chatgpt_connect_observed": True,
        "authenticated_auth_refresh_observed_if_required": (
            not auth_update["credential_material_changed"] or allowed_auth_refresh
        ),
        "profile_sha256": sha256_bytes(profile.encode("utf-8")),
        "stdout_sha256": sha256_bytes(stdout),
        "stderr_sha256": sha256_bytes(stderr),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "proxy_events": proxy_events,
        "auth_state_update": auth_update,
    }


class HiddenEvaluator:
    def __init__(
        self,
        environment: Any,
        hidden_label: str | int,
        budget: int,
        workspace: Path,
        token: str,
        observation_dir: Path | None = None,
        feedback_enabled: bool = False,
    ):
        self.environment = environment
        self.hidden_label = hidden_label
        self.budget = budget
        self.workspace = workspace
        self.mailbox = workspace / ".ag-ipc"
        self.requests = self.mailbox / "requests"
        self.responses = self.mailbox / "responses"
        self.mailbox.mkdir(exist_ok=False)
        self.requests.mkdir()
        self.responses.mkdir()
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        self.observation_dir = observation_dir or workspace / "observations"
        self.observation_dir_name = str(self.observation_dir.relative_to(workspace))
        self._observation_fd = os.open(self.observation_dir, directory_flags)
        self._mailbox_fd = os.open(self.mailbox, directory_flags)
        self._requests_fd = os.open(self.requests, directory_flags)
        self._responses_fd = os.open(self.responses, directory_flags)
        self._directory_identities = {
            "observations": (
                self._observation_fd,
                self.observation_dir,
                os.fstat(self._observation_fd),
            ),
            "mailbox": (
                self._mailbox_fd,
                self.mailbox,
                os.fstat(self._mailbox_fd),
            ),
            "requests": (
                self._requests_fd,
                self.requests,
                os.fstat(self._requests_fd),
            ),
            "responses": (
                self._responses_fd,
                self.responses,
                os.fstat(self._responses_fd),
            ),
        }
        self.token = token
        self.feedback_enabled = feedback_enabled
        self.glimpses_exposed = 1
        self.move_requests = 0
        self.accepted_moves = 0
        self.effective_moves = 0
        self.wall_bumps = 0
        self.effective_revisits = 0
        self.positions: set[tuple[int, int]] = {
            (int(environment.x), int(environment.y))
        }
        self.position_history: list[tuple[int, int]] = [
            (int(environment.x), int(environment.y))
        ]
        self.submit_calls = 0
        self.submitted_answer_raw: str | None = None
        self.submitted_answer: str | int | None = None
        self.correct: bool | None = None
        self.locked = False
        self.invalid_requests = 0
        self.budget_rejections = 0
        self.processed: dict[str, dict[str, Any]] = {}
        self.request_log: list[dict[str, Any]] = []
        self.observation_hashes: list[dict[str, Any]] = []
        self.interface_request_paths: set[str] = set()
        self.response_paths: set[str] = set()
        self.transport_security_events: list[dict[str, Any]] = []
        self._observation_names: set[str] = set()
        self._delivery_expected: dict[str, dict[str, Any]] = {}
        self._delivered_move_ids: set[str] = set()
        self.delivery_receipts: list[dict[str, Any]] = []

    def close(self) -> None:
        for descriptor in (
            self._responses_fd,
            self._requests_fd,
            self._mailbox_fd,
            self._observation_fd,
        ):
            try:
                os.close(descriptor)
            except OSError:
                pass

    def assert_directory_identities(self) -> None:
        for name, (descriptor, path, original) in self._directory_identities.items():
            current_fd = os.fstat(descriptor)
            current_path = os.stat(path, follow_symlinks=False)
            original_id = (original.st_dev, original.st_ino)
            if (
                (current_fd.st_dev, current_fd.st_ino) != original_id
                or (current_path.st_dev, current_path.st_ino) != original_id
                or not stat.S_ISDIR(current_path.st_mode)
            ):
                self.transport_security_events.append(
                    {
                        "timestamp": utc_now(),
                        "event": "protected_directory_identity_changed",
                        "directory": name,
                    }
                )
                raise RuntimeError(f"protected evaluator directory changed: {name}")

    def record_initial(self, path: Path) -> None:
        relative = str(path.relative_to(self.workspace))
        if path.parent != self.observation_dir:
            raise RuntimeError("initial observation is outside the protected directory")
        self._observation_names.add(path.name)
        with Image.open(path) as image:
            pixel_sha256 = sha256_bytes(np.asarray(image.convert("RGB")).tobytes())
        self.observation_hashes.append(
            {
                "observation_index": 0,
                "path": relative,
                "agent_visible_path": str(path),
                "delivery": "initial_input_image",
                "sha256": sha256_file(path),
                "pixel_sha256": pixel_sha256,
            }
        )

    def neutral_error(self, request_id: str, message: str) -> dict[str, Any]:
        return {
            "protocol": 1,
            "request_id": request_id,
            "ok": False,
            "message": message,
        }

    def process_request(self, payload: Any, source_name: str) -> dict[str, Any]:
        if not isinstance(payload, dict):
            self.invalid_requests += 1
            return self.neutral_error(source_name, "request rejected")
        request_id = payload.get("request_id")
        if not isinstance(request_id, str) or not re.fullmatch(r"[0-9a-f]{32}", request_id):
            self.invalid_requests += 1
            return self.neutral_error(source_name, "request rejected")
        if request_id != source_name:
            self.invalid_requests += 1
            return self.neutral_error(source_name, "request rejected")
        if request_id in self.processed:
            return self.processed[request_id]
        authenticated_payload = dict(payload)
        supplied_hmac = authenticated_payload.pop("auth", None)
        supplied_token = authenticated_payload.pop("token", None)
        authenticated_bytes = json.dumps(
            authenticated_payload,
            separators=(",", ":"),
            sort_keys=True,
            ensure_ascii=True,
        ).encode("utf-8")
        expected_hmac = hmac.new(
            self.token.encode("utf-8"), authenticated_bytes, hashlib.sha256
        ).hexdigest()
        authenticated = bool(
            supplied_token is None
            and isinstance(supplied_hmac, str)
            and hmac.compare_digest(supplied_hmac, expected_hmac)
        )
        if payload.get("protocol") != 1 or not authenticated:
            self.invalid_requests += 1
            self.transport_security_events.append(
                {
                    "timestamp": utc_now(),
                    "event": "request_authentication_failed",
                    "request_id": request_id,
                }
            )
            response = self.neutral_error(request_id, "request rejected")
            self.processed[request_id] = response
            return response

        operation = payload.get("operation")
        timestamp = utc_now()
        if operation == "move":
            direction = payload.get("direction")
            if direction not in {"up", "down", "left", "right"}:
                self.invalid_requests += 1
                response = self.neutral_error(request_id, "request rejected")
            else:
                self.move_requests += 1
                if self.locked:
                    response = self.neutral_error(request_id, "episode locked")
                elif self.glimpses_exposed >= self.budget:
                    self.budget_rejections += 1
                    response = self.neutral_error(
                        request_id, "observation budget exhausted"
                    )
                else:
                    before = (int(self.environment.x), int(self.environment.y))
                    action = json.dumps(
                        {"action": "move", "direction": direction},
                        separators=(",", ":"),
                    )
                    observation, _, _, _ = self.environment.step(action)
                    after = (int(self.environment.x), int(self.environment.y))
                    self.accepted_moves += 1
                    if after == before:
                        self.wall_bumps += 1
                    else:
                        self.effective_moves += 1
                        if after in self.positions:
                            self.effective_revisits += 1
                    self.positions.add(after)
                    self.position_history.append(after)
                    index = self.glimpses_exposed
                    while True:
                        filename = secrets.token_hex(16) + ".png"
                        if filename not in self._observation_names:
                            self._observation_names.add(filename)
                            break
                    relative = f"{self.observation_dir_name}/{filename}"
                    absolute = str(self.observation_dir / filename)
                    if (
                        not Path(absolute).is_absolute()
                        or Path(absolute).parent != self.observation_dir
                        or os.path.commonpath((str(self.workspace), absolute))
                        != str(self.workspace)
                    ):
                        raise RuntimeError("trusted observation path escaped workspace")
                    self.assert_directory_identities()
                    encoded = io.BytesIO()
                    observation.save(encoded, format="PNG")
                    png_bytes = encoded.getvalue()
                    descriptor = os.open(
                        filename,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                        0o644,
                        dir_fd=self._observation_fd,
                    )
                    try:
                        with os.fdopen(descriptor, "wb", closefd=False) as handle:
                            handle.write(png_bytes)
                            handle.flush()
                            os.fsync(handle.fileno())
                    finally:
                        os.close(descriptor)
                    self.glimpses_exposed += 1
                    self.observation_hashes.append(
                        {
                            "observation_index": index,
                            "path": relative,
                            "agent_visible_path": absolute,
                            "delivery": "mcp_image_content",
                            "sha256": sha256_bytes(png_bytes),
                            "pixel_sha256": sha256_bytes(
                                np.asarray(observation.convert("RGB")).tobytes()
                            ),
                        }
                    )
                    self._delivery_expected[request_id] = {
                        "path": relative,
                        "absolute_path": absolute,
                        "sha256": sha256_bytes(png_bytes),
                        "byte_count": len(png_bytes),
                    }
                    response = {
                        "protocol": 1,
                        "request_id": request_id,
                        "ok": True,
                        "observation": absolute,
                    }
            self.request_log.append(
                {
                    "timestamp": timestamp,
                    "request_id": request_id,
                    "operation": "move",
                    "direction": direction,
                    "accepted": bool(response.get("ok")),
                }
            )
        elif operation == "delivery_receipt":
            move_request_id = payload.get("move_request_id")
            digest = payload.get("sha256")
            byte_count = payload.get("byte_count")
            delivery_format = payload.get("delivery_format")
            expected = (
                self._delivery_expected.get(move_request_id)
                if isinstance(move_request_id, str)
                else None
            )
            accepted = bool(
                expected is not None
                and move_request_id not in self._delivered_move_ids
                and digest == expected["sha256"]
                and byte_count == expected["byte_count"]
                and delivery_format == "mcp_image_content"
            )
            if accepted:
                self._delivered_move_ids.add(move_request_id)
                receipt = {
                    "timestamp": timestamp,
                    "move_request_id": move_request_id,
                    "path": expected["path"],
                    "sha256": digest,
                    "byte_count": byte_count,
                    "delivery_format": delivery_format,
                }
                self.delivery_receipts.append(receipt)
                response = {
                    "protocol": 1,
                    "request_id": request_id,
                    "ok": True,
                    "message": "delivery receipt accepted",
                }
            else:
                self.invalid_requests += 1
                response = self.neutral_error(request_id, "request rejected")
            self.request_log.append(
                {
                    "timestamp": timestamp,
                    "request_id": request_id,
                    "operation": "delivery_receipt",
                    "accepted": accepted,
                }
            )
        elif operation == "submit":
            self.submit_calls += 1
            raw_value = payload.get("value")
            if not isinstance(raw_value, str) or not raw_value:
                self.invalid_requests += 1
                response = self.neutral_error(request_id, "request rejected")
            elif self.locked:
                response = self.neutral_error(request_id, "episode locked")
            else:
                self.locked = True
                self.submitted_answer_raw = raw_value
                try:
                    self.submitted_answer = int(raw_value)
                except (TypeError, ValueError):
                    self.submitted_answer = -1
                if self.feedback_enabled:
                    self.correct = self.submitted_answer == self.hidden_label
                    receipt = "CORRECT" if self.correct else "INCORRECT"
                else:
                    receipt = "submission accepted"
                response = {
                    "protocol": 1,
                    "request_id": request_id,
                    "ok": True,
                    "message": receipt,
                }
            self.request_log.append(
                {
                    "timestamp": timestamp,
                    "request_id": request_id,
                    "operation": "submit",
                    "accepted": bool(response.get("ok")),
                }
            )
        else:
            self.invalid_requests += 1
            response = self.neutral_error(request_id, "request rejected")
            self.request_log.append(
                {
                    "timestamp": timestamp,
                    "request_id": request_id,
                    "operation": "invalid",
                    "accepted": False,
                }
            )
        self.processed[request_id] = response
        return response

    def finalize_correctness(self) -> None:
        """Score no-feedback arms after Terra terminates; preserve A2 live score."""
        if self.feedback_enabled:
            return
        if self.submitted_answer_raw is None:
            self.submitted_answer = None
            self.correct = None
            return
        self.correct = self.submitted_answer == self.hidden_label

    def handle_pending(self) -> None:
        self.assert_directory_identities()
        for request_name in sorted(os.listdir(self._requests_fd)):
            if not re.fullmatch(r"[0-9a-f]{32}\.json", request_name):
                continue
            request_id = request_name[:-5]
            response_name = f"{request_id}.json"
            try:
                os.stat(
                    response_name,
                    dir_fd=self._responses_fd,
                    follow_symlinks=False,
                )
                continue
            except FileNotFoundError:
                pass
            try:
                descriptor = os.open(
                    request_name,
                    os.O_RDONLY | os.O_NOFOLLOW,
                    dir_fd=self._requests_fd,
                )
                try:
                    request_stat = os.fstat(descriptor)
                    if not stat.S_ISREG(request_stat.st_mode) or request_stat.st_size > 65_536:
                        raise ValueError("request is not a bounded regular file")
                    with os.fdopen(descriptor, "r", encoding="utf-8", closefd=False) as handle:
                        payload = json.load(handle)
                finally:
                    os.close(descriptor)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                self.transport_security_events.append(
                    {
                        "timestamp": utc_now(),
                        "event": "request_file_rejected",
                        "request_name": request_name,
                        "error_type": type(error).__name__,
                    }
                )
                continue
            response = self.process_request(payload, request_id)
            response_bytes = canonical_json_bytes(response)
            temporary_name = f".{request_id}.{uuid.uuid4().hex}.tmp"
            descriptor = os.open(
                temporary_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o644,
                dir_fd=self._responses_fd,
            )
            try:
                with os.fdopen(descriptor, "wb", closefd=False) as handle:
                    handle.write(response_bytes)
                    handle.flush()
                    os.fsync(handle.fileno())
            finally:
                os.close(descriptor)
            os.rename(
                temporary_name,
                response_name,
                src_dir_fd=self._responses_fd,
                dst_dir_fd=self._responses_fd,
            )
            self.interface_request_paths.add(
                f".ag-ipc/requests/{request_name}"
            )
            self.response_paths.add(
                f".ag-ipc/responses/{response_name}"
            )

    def counters(self) -> dict[str, Any]:
        return {
            "glimpses_exposed": self.glimpses_exposed,
            "move_requests": self.move_requests,
            "accepted_moves": self.accepted_moves,
            "effective_moves": self.effective_moves,
            "effective_revisits": self.effective_revisits,
            "wall_bumps": self.wall_bumps,
            "revisit_rate": (
                self.effective_revisits / self.effective_moves
                if self.effective_moves else None
            ),
            "wall_bump_rate": (
                self.wall_bumps / self.accepted_moves
                if self.accepted_moves else None
            ),
            "position_history": [list(position) for position in self.position_history],
            "unique_positions": len(self.positions),
            "submit_calls": self.submit_calls,
            "submitted_answer_raw": self.submitted_answer_raw,
            "submitted_answer": self.submitted_answer,
            "correct": self.correct,
            "budget_exhausted": self.glimpses_exposed >= self.budget,
            "budget_rejections": self.budget_rejections,
            "invalid_requests": self.invalid_requests,
            "native_image_delivery_receipts": len(self.delivery_receipts),
        }


def inventory_files(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        safe_directories: list[str] = []
        for name in sorted(directory_names):
            path = directory_path / name
            mode = os.lstat(path).st_mode
            if stat.S_ISLNK(mode):
                rows.append(
                    {
                        "path": str(path.relative_to(root)),
                        "type": "symlink",
                        "target": os.readlink(path),
                    }
                )
            else:
                safe_directories.append(name)
        directory_names[:] = safe_directories
        for name in sorted(file_names):
            path = directory_path / name
            path_stat = os.lstat(path)
            relative = str(path.relative_to(root))
            if stat.S_ISLNK(path_stat.st_mode):
                rows.append(
                    {
                        "path": relative,
                        "type": "symlink",
                        "target": os.readlink(path),
                    }
                )
                continue
            if not stat.S_ISREG(path_stat.st_mode):
                rows.append({"path": relative, "type": "special"})
                continue
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            digest = hashlib.sha256()
            try:
                while True:
                    chunk = os.read(descriptor, 1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
            finally:
                os.close(descriptor)
            rows.append(
                {
                    "path": relative,
                    "type": "regular",
                    "bytes": path_stat.st_size,
                    "sha256": digest.hexdigest(),
                }
            )
    rows.sort(key=lambda row: row["path"])
    return rows


def inspect_spatial_artifacts(
    workspace: Path, autonomous_files: list[dict[str, Any]], level: int
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for row in autonomous_files:
        if row.get("type") != "regular":
            continue
        relative = row["path"]
        path = workspace / relative
        suffix = path.suffix.lower()
        evidence: list[str] = []
        if suffix in AUTONOMOUS_IMAGE_SUFFIXES:
            try:
                with Image.open(path) as image:
                    width, height = image.size
                canvas_width = IMAGE_SIZE * level
                if width >= canvas_width and height >= IMAGE_SIZE:
                    evidence.append(f"image_dimensions={width}x{height}")
            except OSError:
                pass
        if suffix in TEXT_SUFFIXES and row["bytes"] <= 2_000_000:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                text = ""
            terms = sorted(set(match.group(0) for match in SPATIAL_TERMS.finditer(text)))
            if terms:
                evidence.append("spatial_construction_terms=" + ",".join(terms[:12]))
        if evidence:
            candidates.append({"path": relative, "evidence": evidence})
    return {
        "apparent_spatial_integration_artifact": bool(candidates),
        "post_run_candidates": candidates,
        "basis": "post-run file inspection only; not inferred from accuracy",
    }


def audit_native_mcp_image(
    block: Any,
    expected_dimensions: tuple[int, int] | None = (IMAGE_SIZE, IMAGE_SIZE),
) -> dict[str, Any]:
    reasons: list[str] = []
    if not isinstance(block, dict) or block.get("type") != "image":
        return {"passed": False, "reasons": ["not_image_content"]}
    if block.get("mimeType") != "image/png":
        reasons.append("mime_type_not_png")
    encoded = block.get("data")
    data: bytes | None = None
    if not isinstance(encoded, str) or len(encoded) > 8_000_000:
        reasons.append("missing_or_oversized_base64")
    else:
        try:
            data = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError):
            reasons.append("invalid_base64")
    row: dict[str, Any] = {
        "passed": False,
        "reasons": reasons,
        "mime_type": block.get("mimeType") if isinstance(block, dict) else None,
        "encoded_characters": len(encoded) if isinstance(encoded, str) else None,
    }
    if data is None:
        return row
    row["png_bytes"] = len(data)
    row["png_sha256"] = sha256_bytes(data)
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            row.update(
                {
                    "format": image.format,
                    "width": image.width,
                    "height": image.height,
                    "mode": image.mode,
                    "pixel_sha256": sha256_bytes(
                        np.asarray(image.convert("RGB")).tobytes()
                    ),
                }
            )
            if image.format != "PNG":
                reasons.append("decoded_format_not_png")
            if expected_dimensions is not None and (
                image.width,
                image.height,
            ) != expected_dimensions:
                reasons.append("decoded_dimensions_changed")
    except OSError:
        reasons.append("invalid_png")
    row["passed"] = not reasons
    return row


def parse_jsonl_trace(path: Path) -> dict[str, Any]:
    event_count = 0
    malformed = 0
    thread_ids: set[str] = set()
    tool_ids: set[str] = set()
    tool_without_ids = 0
    resolved_models: set[str] = set()
    usage: dict[str, Any] | None = None
    command_texts: list[str] = []
    model_events: list[dict[str, Any]] = []
    completed_mcp_calls: list[dict[str, Any]] = []
    native_move_deliveries: list[dict[str, Any]] = []
    native_view_deliveries: list[dict[str, Any]] = []
    native_submit_receipts: list[dict[str, Any]] = []
    compaction_events: list[dict[str, Any]] = []
    model_activity_events: list[dict[str, Any]] = []
    completed_image_views: list[dict[str, Any]] = []
    completed_command_executions: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            event_count += 1
            if not isinstance(event, dict):
                continue
            event_type = event.get("type")
            if isinstance(event_type, str) and "compact" in event_type.lower():
                compaction_events.append(
                    {"event_index": event_count, "type": event_type}
                )
            if event_type == "thread.started" and isinstance(event.get("thread_id"), str):
                thread_ids.add(event["thread_id"])
            if event_type == "turn.completed" and isinstance(event.get("usage"), dict):
                usage = event["usage"]
            if isinstance(event_type, str) and "model" in event_type.lower():
                model_events.append(
                    {
                        "type": event_type,
                        "event_sha256": sha256_bytes(canonical_json_bytes(event)),
                    }
                )
            item = event.get("item")
            if isinstance(item, dict):
                item_type = item.get("type")
                if isinstance(item_type, str) and "compact" in item_type.lower():
                    compaction_events.append(
                        {
                            "event_index": event_count,
                            "type": event_type,
                            "item_type": item_type,
                        }
                    )
                if event_type in {"item.started", "item.completed"} and item_type in (
                    TOOL_ITEM_TYPES | {"agent_message", "assistant_message"}
                ):
                    model_activity_events.append(
                        {
                            "event_index": event_count,
                            "event_type": event_type,
                            "item_type": item_type,
                            "item_id": item.get("id"),
                        }
                    )
                if event_type == "item.completed" and item_type == "mcp_tool_call":
                    result = item.get("result")
                    content = result.get("content") if isinstance(result, dict) else None
                    content = content if isinstance(content, list) else []
                    call = {
                        "event_index": event_count,
                        "item_id": item.get("id"),
                        "server": item.get("server"),
                        "tool": item.get("tool"),
                        "arguments": item.get("arguments"),
                        "status": item.get("status"),
                        "error": item.get("error"),
                        "content_types": [
                            node.get("type") if isinstance(node, dict) else None
                            for node in content
                        ],
                    }
                    completed_mcp_calls.append(call)
                    if item.get("server") == "activeglimpse" and item.get("tool") == "move":
                        text_blocks = [
                            node for node in content
                            if isinstance(node, dict) and node.get("type") == "text"
                        ]
                        image_blocks = [
                            node for node in content
                            if isinstance(node, dict) and node.get("type") == "image"
                        ]
                        reasons: list[str] = []
                        if item.get("status") != "completed" or item.get("error") is not None:
                            reasons.append("mcp_move_not_successful")
                        if len(content) != 2 or len(text_blocks) != 1 or len(image_blocks) != 1:
                            reasons.append("mcp_move_content_shape_changed")
                        path_text = text_blocks[0].get("text") if len(text_blocks) == 1 else None
                        if not isinstance(path_text, str) or not Path(path_text).is_absolute():
                            reasons.append("mcp_move_path_not_absolute")
                        image_audit = audit_native_mcp_image(
                            image_blocks[0] if len(image_blocks) == 1 else None
                        )
                        reasons.extend(image_audit["reasons"])
                        native_move_deliveries.append(
                            {
                                **call,
                                "agent_visible_path": path_text,
                                "image": image_audit,
                                "passed": not reasons,
                                "reasons": reasons,
                            }
                        )
                    elif (
                        item.get("server") == "activeglimpse"
                        and item.get("tool") == "view_image"
                    ):
                        text_blocks = [
                            node
                            for node in content
                            if isinstance(node, dict) and node.get("type") == "text"
                        ]
                        image_blocks = [
                            node
                            for node in content
                            if isinstance(node, dict) and node.get("type") == "image"
                        ]
                        reasons: list[str] = []
                        if (
                            item.get("status") != "completed"
                            or item.get("error") is not None
                        ):
                            reasons.append("mcp_view_image_not_successful")
                        if (
                            len(content) != 2
                            or len(text_blocks) != 1
                            or len(image_blocks) != 1
                        ):
                            reasons.append("mcp_view_image_content_shape_changed")
                        path_text = (
                            text_blocks[0].get("text")
                            if len(text_blocks) == 1
                            else None
                        )
                        if (
                            not isinstance(path_text, str)
                            or not Path(path_text).is_absolute()
                        ):
                            reasons.append("mcp_view_image_path_not_absolute")
                        image_audit = audit_native_mcp_image(
                            image_blocks[0] if len(image_blocks) == 1 else None,
                            expected_dimensions=None,
                        )
                        reasons.extend(image_audit["reasons"])
                        native_view_deliveries.append(
                            {
                                **call,
                                "agent_visible_path": path_text,
                                "image": image_audit,
                                "passed": not reasons,
                                "reasons": reasons,
                            }
                        )
                    elif item.get("server") == "activeglimpse" and item.get("tool") == "submit":
                        texts = [
                            node.get("text")
                            for node in content
                            if isinstance(node, dict) and node.get("type") == "text"
                        ]
                        receipt_text = texts[0] if len(texts) == 1 else None
                        passed = bool(
                            item.get("status") == "completed"
                            and item.get("error") is None
                            and len(content) == 1
                            and receipt_text in {
                                "submission accepted",
                                "CORRECT",
                                "INCORRECT",
                            }
                        )
                        native_submit_receipts.append(
                            {**call, "receipt_text": receipt_text, "passed": passed}
                        )
                elif event_type == "item.completed" and item_type in {
                    "image_view",
                    "view_image",
                }:
                    candidate_paths = sorted(
                        {
                            value
                            for value in recursive_nodes(item)
                            if isinstance(value, str)
                            and Path(value).is_absolute()
                            and value.lower().endswith(
                                (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")
                            )
                        }
                    )
                    completed_image_views.append(
                        {
                            "event_index": event_count,
                            "item_id": item.get("id"),
                            "item_type": item_type,
                            "status": item.get("status"),
                            "error": item.get("error"),
                            "absolute_paths": candidate_paths,
                        }
                    )
                elif event_type == "item.completed" and item_type == "command_execution":
                    command = item.get("command")
                    if isinstance(command, list):
                        command = " ".join(str(part) for part in command)
                    completed_command_executions.append(
                        {
                            "event_index": event_count,
                            "item_id": item.get("id"),
                            "command": command if isinstance(command, str) else None,
                            "status": item.get("status"),
                            "exit_code": item.get("exit_code"),
                        }
                    )
            for node in recursive_nodes(event):
                if not isinstance(node, dict):
                    continue
                for key in (
                    "model",
                    "model_name",
                    "resolved_model",
                    "to_model",
                    "target_model",
                ):
                    value = node.get(key)
                    if isinstance(value, str) and value.startswith("gpt-"):
                        resolved_models.add(value)
                item_type = node.get("type")
                if item_type in TOOL_ITEM_TYPES:
                    item_id = node.get("id")
                    if isinstance(item_id, str):
                        tool_ids.add(item_id)
                    elif event_type == "item.completed":
                        tool_without_ids += 1
                command = node.get("command")
                if isinstance(command, str):
                    command_texts.append(command)
                elif isinstance(command, list):
                    command_texts.append(" ".join(str(part) for part in command))
    forbidden_command_markers = [
        marker
        for marker in (
            str(SOURCE_ROOT),
            str(HOST_MEMORY),
            str(LEVEL1_PRIOR_ROOT),
            str(LEVEL2_PRIOR_ROOT),
            str(HOST_AUTH),
        )
        if any(marker in command for command in command_texts)
    ]
    legacy_ag_shell_commands = [
        command
        for command in command_texts
        if re.search(r"(?:^|[;&|]\s*)ag\s+(?:move|submit)\b", command)
    ]
    for delivery in native_move_deliveries:
        delivery["subsequent_model_activity_observed"] = any(
            row["event_index"] > delivery["event_index"]
            and row.get("item_id") != delivery.get("item_id")
            for row in model_activity_events
        )
    for delivery in native_view_deliveries:
        delivery["subsequent_model_activity_observed"] = any(
            row["event_index"] > delivery["event_index"]
            and row.get("item_id") != delivery.get("item_id")
            for row in model_activity_events
        )
    return {
        "jsonl_events": event_count,
        "malformed_jsonl_lines": malformed,
        "thread_ids": sorted(thread_ids),
        "thread_started_count": len(thread_ids),
        "total_codex_tool_calls": len(tool_ids) + tool_without_ids,
        "tool_item_ids": sorted(tool_ids),
        "resolved_model_candidates": sorted(resolved_models),
        "model_events": model_events,
        "usage": usage,
        "forbidden_host_paths_in_model_commands": forbidden_command_markers,
        "completed_mcp_calls": completed_mcp_calls,
        "native_move_deliveries": native_move_deliveries,
        "native_view_deliveries": native_view_deliveries,
        "native_submit_receipts": native_submit_receipts,
        "completed_image_views": completed_image_views,
        "completed_command_executions": completed_command_executions,
        "command_texts": command_texts,
        "legacy_ag_shell_commands": legacy_ag_shell_commands,
        "context_compaction_events": compaction_events,
    }


def build_exec_argv(
    workspace: Path,
    generic_root: Path,
    token: str,
    level: int,
    initial_path: Path | None = None,
    prompt_override: str | None = None,
) -> list[str]:
    prompt = prompt_override if prompt_override is not None else prompt_for(level)
    shell_environment = scored_shell_environment(workspace, generic_root)
    shell_set = (
        "shell_environment_policy.set={"
        f"PATH={toml_quote(shell_environment['PATH'])},"
        f"HOME={toml_quote(shell_environment['HOME'])},"
        f"TMPDIR={toml_quote(shell_environment['TMPDIR'])},"
        f"TMPPREFIX={toml_quote(shell_environment['TMPPREFIX'])},"
        f"PYTHONDONTWRITEBYTECODE={toml_quote(shell_environment['PYTHONDONTWRITEBYTECODE'])},"
        f"LANG={toml_quote(shell_environment['LANG'])}}}"
    )
    mcp_server = generic_root / "bin/ag-mcp"
    mcp_config = (
        "mcp_servers.activeglimpse={"
        f"command={toml_quote(mcp_server)},"
        'env_vars=["AG_MCP_MAILBOX","AG_MCP_TOKEN",'
        '"AG_MCP_OBSERVATION_DIR","AG_MCP_WORKSPACE"],'
        'enabled=true,required=true,enabled_tools=["move","view_image","submit"],'
        'startup_timeout_sec=10,tool_timeout_sec=120,'
        'default_tools_approval_mode="auto"}'
    )
    argv = [
        str(CODEX),
        "exec",
        "--model",
        MODEL,
        "--config",
        f'model_reasoning_effort="{REASONING_EFFORT}"',
        "--config",
        'approval_policy="never"',
        "--config",
        "features.respect_system_proxy=true",
        "--config",
        'shell_environment_policy.inherit="none"',
        "--config",
        shell_set,
        "--config",
        mcp_config,
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--skip-git-repo-check",
        "--sandbox",
        "danger-full-access",
        "--json",
        "--color",
        "never",
        "--strict-config",
        "--cd",
        str(workspace),
        "--image",
        str(initial_path or workspace / "observations/0000.png"),
    ]
    add_config_args(argv, CLEAN_CONFIG)
    argv.append(prompt)
    return argv


def scored_shell_environment(
    workspace: Path,
    generic_root: Path,
) -> dict[str, str]:
    """Exact environment exposed to model-initiated shell commands."""
    agent_tmp = workspace / ".tmp"
    return {
        "HOME": str(workspace),
        "TMPDIR": str(agent_tmp),
        "TMPPREFIX": str(agent_tmp / "zsh"),
        "PATH": (
            f"{generic_root / 'bin'}:{HOMEBREW_ROOT / 'bin'}:"
            "/opt/anaconda3/bin:/usr/bin:/bin"
        ),
        "PYTHONDONTWRITEBYTECODE": "1",
        "LANG": "C.UTF-8",
    }


def scored_parent_environment(
    runtime: Path,
    proxy_url: str,
    workspace: Path,
    observation_dir: Path,
    token: str,
) -> dict[str, str]:
    environment = parent_environment(runtime, proxy_url)
    environment.update(
        {
            "AG_MCP_MAILBOX": str(workspace / ".ag-ipc"),
            "AG_MCP_TOKEN": token,
            "AG_MCP_OBSERVATION_DIR": str(observation_dir),
            "AG_MCP_WORKSPACE": str(workspace),
        }
    )
    return environment


def deterministic_local_toolchain_preflight(
    run_root: Path,
    generic_root: Path,
) -> dict[str, Any]:
    """Hard-gate MCP, native PNG, viewer, and ImageMagick without a model call."""
    workspace = Path("/private/tmp") / secrets.token_hex(16)
    observation_dir = workspace / secrets.token_hex(16)
    agent_tmp = workspace / ".tmp"
    runtime_stub: Path | None = None
    proxy: AuthenticatedConnectProxy | None = None
    try:
        RUNTIME_PARENT.mkdir(parents=True, exist_ok=True)
        runtime_stub = Path(
            tempfile.mkdtemp(
                prefix=f"mnistpro-{ACTIVE_ARM.lower()}-direct-",
                dir=RUNTIME_PARENT,
            )
        )
        observation_dir.mkdir(parents=True, mode=0o700)
        agent_tmp.mkdir(mode=0o700)
        (runtime_stub / "home").mkdir(mode=0o700)
        (runtime_stub / "codex-home").mkdir(mode=0o700)
        (runtime_stub / "tmp").mkdir(mode=0o700)
        proxy = AuthenticatedConnectProxy()
    except BaseException:
        if proxy is not None:
            proxy.stop()
        if workspace.exists():
            shutil.rmtree(workspace)
        if runtime_stub is not None and runtime_stub.exists():
            shutil.rmtree(runtime_stub)
        raise
    assert runtime_stub is not None and proxy is not None
    initial_path = observation_dir / f"{secrets.token_hex(16)}.png"
    composite_path = agent_tmp / f"{secrets.token_hex(16)}.png"
    archived_composite = run_root / "preflight/direct_toolchain_composite.png"
    rpc_trace = run_root / "preflight/direct_toolchain_rpc.jsonl"
    rpc_stderr_path = run_root / "preflight/direct_toolchain_rpc.stderr.log"
    magick_stdout_path = run_root / "preflight/direct_toolchain_magick.stdout.log"
    magick_stderr_path = run_root / "preflight/direct_toolchain_magick.stderr.log"
    process: subprocess.Popen[bytes] | None = None
    magick_process: subprocess.Popen[bytes] | None = None
    evaluator: HiddenEvaluator | None = None
    pump_thread: threading.Thread | None = None
    rpc_stderr_handle: Any = None
    stopping = threading.Event()
    try:
        initial = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (127, 127, 127))
        left = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (17, 83, 149))
        right = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (191, 53, 71))
        initial.save(initial_path, format="PNG")
        expected_composite_pixels = np.concatenate(
            [np.asarray(left), np.asarray(right)], axis=1
        )
        expected_composite_pixel_sha256 = sha256_bytes(
            expected_composite_pixels.tobytes()
        )

        token = secrets.token_hex(32)

        class DirectEnvironment:
            x = 0
            y = 0

            def __init__(self) -> None:
                self.calls = 0

            def step(
                self, action: str
            ) -> tuple[Image.Image, float, bool, dict[str, Any]]:
                _ = action
                panel = (left, right)[min(self.calls, 1)].copy()
                self.calls += 1
                self.x = STEP_SIZE * self.calls
                return panel, 0.0, False, {}

        evaluator = HiddenEvaluator(
            DirectEnvironment(),
            7,
            BUDGETS[1],
            workspace,
            token,
            observation_dir,
            feedback_enabled=ACTIVE_ARM == "A2",
        )
        evaluator.record_initial(initial_path)
        profile = external_profile(
            workspace, runtime_stub, generic_root, proxy.port, observation_dir
        )
        prefix = external_sandbox_prefix(profile)
        model_shell_environment = scored_shell_environment(workspace, generic_root)
        mcp_environment = scored_parent_environment(
            runtime_stub, proxy.url, workspace, observation_dir, token
        )
        if any(key.startswith("AG_MCP_") for key in model_shell_environment):
            raise RuntimeError("MCP capability secret leaked into model shell")

        def pump_evaluator() -> None:
            try:
                while not stopping.is_set():
                    assert evaluator is not None
                    evaluator.handle_pending()
                    time.sleep(0.005)
            except BaseException as error:
                pump_errors.append(f"{type(error).__name__}: {error}")
                stopping.set()

        pump_errors: list[str] = []
        pump_thread = threading.Thread(
            target=pump_evaluator,
            name=f"direct-mcp-pump-{ACTIVE_ARM}",
            daemon=True,
        )
        pump_thread.start()
        rpc_stderr_handle = rpc_stderr_path.open("wb")
        process = subprocess.Popen(
            prefix + [str(generic_root / "bin/ag-mcp")],
            cwd=workspace,
            env=mcp_environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=rpc_stderr_handle,
            start_new_session=True,
        )
        assert process.stdin is not None and process.stdout is not None
        responses: list[dict[str, Any]] = []

        def send_notification(method: str, params: dict[str, Any]) -> None:
            message = {"jsonrpc": "2.0", "method": method, "params": params}
            process.stdin.write(
                json.dumps(message, separators=(",", ":")).encode("utf-8") + b"\n"
            )
            process.stdin.flush()

        def rpc_call(
            request_id: int, method: str, params: dict[str, Any]
        ) -> dict[str, Any]:
            message = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            }
            process.stdin.write(
                json.dumps(message, separators=(",", ":")).encode("utf-8") + b"\n"
            )
            process.stdin.flush()
            deadline = time.monotonic() + 45
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(f"direct MCP RPC {request_id} timed out")
                ready, _, _ = select.select(
                    [process.stdout.fileno()], [], [], min(0.25, remaining)
                )
                if not ready:
                    if pump_errors:
                        raise RuntimeError(
                            "direct evaluator pump failed: " + pump_errors[0]
                        )
                    if process.poll() is not None:
                        raise RuntimeError("direct MCP process exited before response")
                    continue
                line = process.stdout.readline(8_000_001)
                if not line or len(line) > 8_000_000 or not line.endswith(b"\n"):
                    raise RuntimeError("direct MCP response line was invalid")
                response = json.loads(line)
                if (
                    not isinstance(response, dict)
                    or response.get("jsonrpc") != "2.0"
                    or response.get("id") != request_id
                ):
                    raise RuntimeError("direct MCP response ID changed")
                if "error" in response:
                    raise RuntimeError(f"direct MCP RPC error: {response['error']}")
                responses.append(response)
                return response

        initialize_response = rpc_call(
            1,
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "controller-preflight", "version": "1"},
            },
        )
        if initialize_response.get("result") != {
            "protocolVersion": "2025-06-18",
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "activeglimpse", "version": "1.0.0"},
        }:
            raise RuntimeError("direct MCP initialize contract changed")
        send_notification("notifications/initialized", {})
        tools_response = rpc_call(2, "tools/list", {})
        tools = tools_response["result"].get("tools")
        if [tool.get("name") for tool in tools or []] != [
            "move",
            "view_image",
            "submit",
        ]:
            raise RuntimeError("direct MCP tool catalog changed")
        expected_required = {
            "move": ["direction"],
            "view_image": ["path"],
            "submit": ["value"],
        }
        if any(
            not isinstance(tool.get("description"), str)
            or not isinstance(tool.get("inputSchema"), dict)
            or tool["inputSchema"].get("type") != "object"
            or tool["inputSchema"].get("required")
            != expected_required[tool["name"]]
            or tool["inputSchema"].get("additionalProperties") is not False
            for tool in tools
        ):
            raise RuntimeError("direct MCP tool schemas changed")
        tools_payload_sha256 = sha256_bytes(canonical_json_bytes(tools))
        if tools_payload_sha256 != PINNED_HASHES["ag_mcp_tools"]:
            raise RuntimeError("direct MCP canonical tool payload changed")

        move_responses = [
            rpc_call(
                request_id,
                "tools/call",
                {"name": "move", "arguments": {"direction": direction}},
            )
            for request_id, direction in ((3, "right"), (4, "down"))
        ]
        expected_moves = evaluator.observation_hashes[1:]
        move_paths: list[str] = []
        move_pngs: list[bytes] = []
        for response, expected in zip(move_responses, expected_moves):
            content = response["result"].get("content")
            if (
                response["result"].get("isError") is not False
                or not isinstance(content, list)
                or [block.get("type") for block in content] != ["text", "image"]
                or content[1].get("mimeType") != "image/png"
            ):
                raise RuntimeError("direct MCP move content shape changed")
            path = content[0].get("text")
            png = base64.b64decode(content[1].get("data", ""), validate=True)
            if path != expected["agent_visible_path"] or sha256_bytes(png) != expected["sha256"]:
                raise RuntimeError("direct MCP move image hash changed")
            move_paths.append(path)
            move_pngs.append(png)

        source_view_pngs: list[bytes] = []
        for request_id, path, expected_png in zip(
            (5, 6), move_paths, move_pngs
        ):
            response = rpc_call(
                request_id,
                "tools/call",
                {"name": "view_image", "arguments": {"path": path}},
            )
            content = response["result"].get("content")
            if (
                response["result"].get("isError") is not False
                or not isinstance(content, list)
                or [block.get("type") for block in content] != ["text", "image"]
                or content[0].get("text") != path
                or content[1].get("mimeType") != "image/png"
            ):
                raise RuntimeError("direct MCP source view shape changed")
            viewed_png = base64.b64decode(
                content[1].get("data", ""), validate=True
            )
            if viewed_png != expected_png:
                raise RuntimeError("direct MCP source view bytes changed")
            source_view_pngs.append(viewed_png)

        resolved_magick = shutil.which(
            "magick", path=model_shell_environment["PATH"]
        )
        if resolved_magick != str(IMAGEMAGICK):
            raise RuntimeError("scored shell PATH does not resolve pinned ImageMagick")
        magick_process = subprocess.Popen(
            prefix
            + [
                "/usr/bin/env",
                "magick",
                move_paths[0],
                move_paths[1],
                "+append",
                str(composite_path),
            ],
            cwd=workspace,
            env=model_shell_environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            magick_stdout, magick_stderr = magick_process.communicate(timeout=60)
        except Exception:
            terminate_process_group(magick_process)
            raise
        magick_returncode, magick_group_terminated = terminate_process_group(
            magick_process
        )
        if not magick_group_terminated:
            raise RuntimeError("direct ImageMagick process group survived")
        magick_process = None
        magick_stdout_path.write_bytes(magick_stdout)
        magick_stderr_path.write_bytes(magick_stderr)
        if (
            magick_returncode != 0
            or not magick_group_terminated
            or not composite_path.is_file()
        ):
            raise RuntimeError("direct ImageMagick preflight failed")
        composite_descriptor = os.open(
            composite_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        )
        try:
            composite_before = os.fstat(composite_descriptor)
            if (
                not stat.S_ISREG(composite_before.st_mode)
                or composite_before.st_uid != os.getuid()
                or composite_before.st_nlink != 1
                or not 1 <= composite_before.st_size <= 8_000_000
            ):
                raise RuntimeError("direct composite failed structural validation")
            composite_chunks: list[bytes] = []
            composite_remaining = composite_before.st_size
            while composite_remaining:
                chunk = os.read(
                    composite_descriptor, min(composite_remaining, 1024 * 1024)
                )
                if not chunk:
                    break
                composite_chunks.append(chunk)
                composite_remaining -= len(chunk)
            composite_after = os.fstat(composite_descriptor)
        finally:
            os.close(composite_descriptor)
        if (
            composite_remaining != 0
            or (
                composite_before.st_dev,
                composite_before.st_ino,
                composite_before.st_size,
                composite_before.st_mtime_ns,
            )
            != (
                composite_after.st_dev,
                composite_after.st_ino,
                composite_after.st_size,
                composite_after.st_mtime_ns,
            )
        ):
            raise RuntimeError("direct composite changed during read")
        composite_bytes = b"".join(composite_chunks)
        with Image.open(io.BytesIO(composite_bytes)) as composite:
            composite.load()
            composite_width = composite.width
            composite_height = composite.height
            composite_pixel_sha256 = sha256_bytes(
                np.asarray(composite.convert("RGB")).tobytes()
            )
        if (
            magick_stdout
            or composite_width != 2 * IMAGE_SIZE
            or composite_height != IMAGE_SIZE
            or composite_pixel_sha256 != expected_composite_pixel_sha256
        ):
            raise RuntimeError("ImageMagick PATH or composite parity changed")

        composite_response = rpc_call(
            7,
            "tools/call",
            {"name": "view_image", "arguments": {"path": str(composite_path)}},
        )
        composite_content = composite_response["result"].get("content")
        if (
            composite_response["result"].get("isError") is not False
            or not isinstance(composite_content, list)
            or [block.get("type") for block in composite_content]
            != ["text", "image"]
            or composite_content[0].get("text") != str(composite_path)
            or composite_content[1].get("mimeType") != "image/png"
        ):
            raise RuntimeError("direct MCP composite view shape changed")
        composite_view_png = base64.b64decode(
            composite_content[1].get("data", ""), validate=True
        )
        submit_response = rpc_call(
            8,
            "tools/call",
            {"name": "submit", "arguments": {"value": "7"}},
        )
        submit_content = submit_response["result"].get("content")
        expected_receipt = "CORRECT" if ACTIVE_ARM == "A2" else "submission accepted"
        if (
            submit_response["result"].get("isError") is not False
            or submit_content != [{"type": "text", "text": expected_receipt}]
        ):
            raise RuntimeError("direct MCP submit receipt changed")

        process.stdin.close()
        process.stdin = None
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            terminate_process_group(process)
            raise
        rpc_returncode, rpc_group_terminated = terminate_process_group(process)
        if not rpc_group_terminated:
            raise RuntimeError("direct MCP process group survived")
        process = None
        rpc_stderr_handle.close()
        rpc_stderr_handle = None
        stopping.set()
        pump_thread.join(timeout=5)
        if pump_thread.is_alive():
            raise RuntimeError("direct evaluator pump did not stop")
        evaluator.handle_pending()
        rpc_trace.write_bytes(
            b"".join(
                json.dumps(response, separators=(",", ":")).encode("utf-8")
                + b"\n"
                for response in responses
            )
        )
        rpc_stderr = rpc_stderr_path.read_bytes()
        counters = evaluator.counters()
        if (
            rpc_returncode != 0
            or pump_thread.is_alive()
            or pump_errors
            or composite_view_png != composite_bytes
            or counters["accepted_moves"] != 2
            or counters["native_image_delivery_receipts"] != 2
            or counters["submit_calls"] != 1
            or counters["correct"] != (True if ACTIVE_ARM == "A2" else None)
            or evaluator.transport_security_events
            or len(responses) != 8
            or [response.get("id") for response in responses] != list(range(1, 9))
            or token.encode("utf-8") in rpc_trace.read_bytes()
            or token.encode("utf-8") in rpc_stderr
            or proxy.snapshot()
        ):
            raise RuntimeError("direct MCP transport reconciliation failed")
        archived_composite.write_bytes(composite_bytes)
        return {
            "passed": True,
            "model_calls": 0,
            "exact_scored_seatbelt_profile": True,
            "profile_sha256": sha256_bytes(profile.encode("utf-8")),
            "tools_list": ["move", "view_image", "submit"],
            "tools_payload_sha256": tools_payload_sha256,
            "native_move_png_sha256": [sha256_bytes(png) for png in move_pngs],
            "controller_move_png_sha256": [
                expected["sha256"] for expected in expected_moves
            ],
            "source_move_images_reopened_byte_exact": 2,
            "viewed_composite_png_sha256": sha256_bytes(composite_view_png),
            "composite_pixel_sha256": composite_pixel_sha256,
            "expected_composite_pixel_sha256": expected_composite_pixel_sha256,
            "imagemagick_resolved_path": resolved_magick,
            "imagemagick_sha256": sha256_file(IMAGEMAGICK),
            "imagemagick_invoked_by_bare_path_name": True,
            "submit_receipt": expected_receipt,
            "token_serialized_to_stdio": False,
            "mcp_parent_environment_matches_scored_boundary": True,
            "model_shell_contains_mcp_secret": False,
            "process_groups_terminated": True,
            "archived_composite": str(archived_composite),
            "rpc_trace": str(rpc_trace),
        }
    finally:
        stopping.set()
        cleanup_groups_terminated = True
        if magick_process is not None:
            _, group_terminated = terminate_process_group(magick_process)
            cleanup_groups_terminated = (
                cleanup_groups_terminated and group_terminated
            )
        if process is not None:
            _, group_terminated = terminate_process_group(process)
            cleanup_groups_terminated = (
                cleanup_groups_terminated and group_terminated
            )
        if rpc_stderr_handle is not None:
            rpc_stderr_handle.close()
        if pump_thread is not None:
            pump_thread.join(timeout=5)
        pump_survived = bool(pump_thread is not None and pump_thread.is_alive())
        if (
            evaluator is not None
            and not pump_survived
            and cleanup_groups_terminated
        ):
            evaluator.close()
        proxy.stop()
        if pump_survived or not cleanup_groups_terminated:
            raise RuntimeError(
                "direct child or evaluator pump survived; controller-private "
                "state preserved"
            )
        shutil.rmtree(workspace)
        if (
            runtime_stub.parent != RUNTIME_PARENT
            or not runtime_stub.name.startswith(
                f"mnistpro-{ACTIVE_ARM.lower()}-direct-"
            )
        ):
            raise RuntimeError("direct runtime identity changed")
        shutil.rmtree(runtime_stub)


def a1_integrated_scored_shape_preflight(
    run_root: Path,
    generic_root: Path,
    minimal_auth: dict[str, Any],
    vault: AuthStateVault,
    required_access_expiry: float,
    module: Any,
    spec: EpisodeSpec,
    image_cache: dict[int, tuple[Image.Image, int]],
) -> dict[str, Any]:
    """Prove native MCP ImageContent delivery under the exact scored profile."""
    _ = (module, image_cache)
    workspace = Path("/private/tmp") / secrets.token_hex(16)
    observation_dir = workspace / secrets.token_hex(16)
    observation_dir.mkdir(parents=True, mode=0o700)
    (workspace / ".tmp").mkdir(mode=0o700)
    if ACTIVE_ARM in {"A1", "A2"}:
        write_initial_notes(workspace, b"")
    initial_name = secrets.token_hex(16) + ".png"
    initial_path = observation_dir / initial_name
    runtime = create_runtime(minimal_auth, "mnistpro-a1-probe-")
    proxy = AuthenticatedConnectProxy()
    evaluator: HiddenEvaluator | None = None
    auth_update: dict[str, Any] | None = None
    proxy_events: list[dict[str, Any]] = []
    result: dict[str, Any] = {}
    process: subprocess.Popen[bytes] | None = None
    timed_out = False
    trace_path = run_root / "preflight/native_image_delivery.jsonl"
    stderr_path = run_root / "preflight/native_image_delivery.stderr.log"
    try:
        digits = [secrets.choice("246789") for _ in range(9)]
        visual_nonce = "".join(digits)

        def redundant_code_image(segment: list[str]) -> Image.Image:
            """Render one high-contrast three-digit code twice for OCR redundancy."""
            code = " ".join(segment)
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), "white")
            draw = ImageDraw.Draw(image)
            font = ImageFont.load_default(size=74)
            for center_y in (58, 166):
                bounds = draw.textbbox((0, 0), code, font=font)
                width = bounds[2] - bounds[0]
                height = bounds[3] - bounds[1]
                draw.text(
                    ((IMAGE_SIZE - width) / 2, center_y - height / 2 - bounds[1]),
                    code,
                    fill="black",
                    font=font,
                )
            draw.rectangle((2, 2, IMAGE_SIZE - 3, IMAGE_SIZE - 3), outline="black", width=4)
            draw.line((4, IMAGE_SIZE // 2, IMAGE_SIZE - 5, IMAGE_SIZE // 2), fill="black", width=2)
            return image

        initial = redundant_code_image(digits[:3])
        moved_observations = [
            redundant_code_image(digits[3:6]),
            redundant_code_image(digits[6:9]),
        ]
        initial.save(initial_path, format="PNG")
        diagnostic_images = [initial, *moved_observations]
        archived_diagnostic_images: list[dict[str, Any]] = []
        for image_index, diagnostic_image in enumerate(diagnostic_images):
            archive_path = (
                run_root
                / "preflight"
                / f"native_image_delivery_observation_{image_index}.png"
            )
            diagnostic_image.save(archive_path, format="PNG")
            archived_diagnostic_images.append(
                {
                    "observation_index": image_index,
                    "path": str(archive_path),
                    "png_sha256": sha256_file(archive_path),
                    "pixel_sha256": sha256_bytes(
                        np.asarray(diagnostic_image.convert("RGB")).tobytes()
                    ),
                    "expected_segment": visual_nonce[
                        image_index * 3 : (image_index + 1) * 3
                    ],
                }
            )

        class VisualNonceEnvironment:
            x = 0
            y = 0
            move_index = 0

            def step(
                self, action: str
            ) -> tuple[Image.Image, float, bool, dict[str, Any]]:
                image = moved_observations[min(self.move_index, 1)].copy()
                self.move_index += 1
                self.x += STEP_SIZE
                return image, 0.0, False, {}

        token = secrets.token_hex(32)
        evaluator = HiddenEvaluator(
            VisualNonceEnvironment(),
            "unscored-native-image-diagnostic",
            BUDGETS[1],
            workspace,
            token,
            observation_dir,
        )
        evaluator.record_initial(initial_path)
        profile = external_profile(
            workspace, runtime, generic_root, proxy.port, observation_dir
        )
        prefix = external_sandbox_prefix(profile)
        outer_environment = scored_parent_environment(
            runtime, proxy.url, workspace, observation_dir, token
        )
        login = subprocess.run(
            prefix + [str(CODEX), "login", "status"],
            cwd=workspace,
            env=outer_environment,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=30,
        )
        login_text = (login.stdout + login.stderr).decode("utf-8", "replace").strip()
        if login.returncode != 0 or login_text != "Logged in using ChatGPT":
            raise RuntimeError(f"A1 integrated authentication failed: {login_text[:200]}")

        scored_argv = build_exec_argv(
            workspace, generic_root, token, 1, initial_path
        )
        config_values: list[str] = []
        for index, argument in enumerate(scored_argv[:-1]):
            if argument in {"-c", "--config"}:
                config_values.append(scored_argv[index + 1])
        sentinel_key = "zzzz_controller_preflight_unknown_sentinel"
        strict_argv = [
            *scored_argv[:-1],
            "--config",
            f"{sentinel_key}=true",
            scored_argv[-1],
        ]
        strict_process = subprocess.run(
            prefix + strict_argv,
            cwd=workspace,
            env=outer_environment,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=60,
        )
        strict_stderr = strict_process.stderr.decode("utf-8", "replace")
        if (
            strict_process.returncode == 0
            or f"unknown configuration field `{sentinel_key}`" not in strict_stderr
            or b'"type":"thread.started"' in strict_process.stdout
            or b'"type": "thread.started"' in strict_process.stdout
        ):
            raise RuntimeError("A1 strict configuration probe failed")

        debug_argv = [
            str(CODEX),
            "debug",
            "prompt-input",
            "--image",
            str(initial_path),
        ]
        add_config_args(debug_argv, config_values)
        debug_argv.append(prompt_for(1))
        rendered_process = subprocess.run(
            prefix + debug_argv,
            cwd=workspace,
            env=outer_environment,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=60,
        )
        if rendered_process.returncode != 0:
            raise RuntimeError(
                "A1 prompt render failed: "
                + rendered_process.stderr.decode("utf-8", "replace")[:500]
            )
        rendered = json.loads(rendered_process.stdout)
        content = rendered[0].get("content") if isinstance(rendered, list) and len(rendered) == 1 else None
        texts = [
            node.get("text")
            for node in content or []
            if isinstance(node, dict) and node.get("type") == "input_text"
        ]
        images = [
            node
            for node in content or []
            if isinstance(node, dict) and node.get("type") == "input_image"
        ]
        expected_texts = [
            f'<image name=[Image #1] path="{initial_path}">',
            "</image>",
            prompt_for(1),
        ]
        all_strings = "\n".join(
            node for node in recursive_nodes(rendered) if isinstance(node, str)
        )
        private_markers = [
            spec.key,
            f"episode_{spec.episode_id:03d}",
            f"episode-{spec.episode_id:03d}",
            f"eval_index={spec.indices[0]}",
            "100 episodes",
            "balanced",
            "round-robin",
        ]
        leaked_private = [marker for marker in private_markers if marker in all_strings]
        if texts != expected_texts or len(images) != 1 or leaked_private:
            raise RuntimeError(
                f"A1 rendered prompt mismatch or private leak: {leaked_private}"
            )
        diagnostic_prompt = (
            "This is a non-scored multi-turn native-image transport diagnostic. "
            "The attached initial image shows one three-digit code, printed twice "
            "identically for visual redundancy. Read the code once. Then call "
            "`activeglimpse.move` exactly twice, using direction `right` both times. "
            "Each move result contains one more redundantly printed three-digit code "
            "as native image content. Remember the earlier codes and concatenate all "
            "nine digits in observation order: initial, first move, second move. "
            "Every digit is one of 2, 4, 6, 7, 8, and 9. Do not "
            "run shell commands, do not call an image viewer, and do not open any "
            "auxiliary pathname. Finally call `activeglimpse.submit` exactly once "
            "with the nine digits in `value`. Do not call any other tool."
        )
        diagnostic_argv = build_exec_argv(
            workspace,
            generic_root,
            token,
            1,
            initial_path,
            prompt_override=diagnostic_prompt,
        )
        with trace_path.open("wb") as trace_handle, stderr_path.open("wb") as stderr_handle:
            process = subprocess.Popen(
                prefix + diagnostic_argv,
                cwd=workspace,
                env=outer_environment,
                stdin=subprocess.DEVNULL,
                stdout=trace_handle,
                stderr=stderr_handle,
                start_new_session=True,
            )
            deadline = time.monotonic() + 5 * 60
            while process.poll() is None:
                evaluator.handle_pending()
                if time.monotonic() >= deadline:
                    timed_out = True
                    break
                time.sleep(0.025)
            evaluator.handle_pending()
        returncode, process_group_terminated = terminate_process_group(process)
        process = None
        trace_audit = parse_jsonl_trace(trace_path)
        counters = evaluator.counters()
        notes_bytes, notes_meta = (
            acquire_notes_candidate(workspace)
            if ACTIVE_ARM in {"A1", "A2"}
            else (None, {"structural_pass": True})
        )
        move_deliveries = trace_audit["native_move_deliveries"]
        expected_moves = evaluator.observation_hashes[1:]
        normalized_submission = re.sub(
            r"\s+", "", evaluator.submitted_answer_raw or ""
        ).upper()
        expected_segments = [visual_nonce[index : index + 3] for index in range(0, 9, 3)]
        submitted_segments = [
            normalized_submission[index : index + 3] for index in range(0, 9, 3)
        ]
        segment_exact = [
            submitted == expected
            for submitted, expected in zip(submitted_segments, expected_segments)
        ]
        character_matches = sum(
            submitted == expected
            for submitted, expected in zip(normalized_submission, visual_nonce)
        )
        passed_move_deliveries = [
            delivery for delivery in move_deliveries if delivery.get("passed") is True
        ]
        move_transport_consistent = bool(
            len(passed_move_deliveries) == counters["accepted_moves"]
            and counters["native_image_delivery_receipts"]
            == counters["accepted_moves"]
            and len(expected_moves) == counters["accepted_moves"]
            and all(
                delivery["agent_visible_path"] == expected["agent_visible_path"]
                and delivery["image"].get("png_sha256") == expected["sha256"]
                and delivery["image"].get("pixel_sha256")
                == expected["pixel_sha256"]
                for delivery, expected in zip(
                    passed_move_deliveries, expected_moves
                )
            )
        )
        accepted_submit_calls = sum(
            row.get("operation") == "submit" and row.get("accepted") is True
            for row in evaluator.request_log
        )
        submit_transport_consistent = bool(
            len(
                [
                    receipt
                    for receipt in trace_audit["native_submit_receipts"]
                    if receipt.get("passed") is True
                ]
            )
            == accepted_submit_calls
            and all(
                receipt.get("receipt_text") == "submission accepted"
                for receipt in trace_audit["native_submit_receipts"]
                if receipt.get("passed") is True
            )
        )
        terra_followed_diagnostic = bool(
            returncode == 0
            and not timed_out
            and counters["glimpses_exposed"] == 3
            and counters["accepted_moves"] == 2
            and counters["submit_calls"] == 1
            and trace_audit["total_codex_tool_calls"] == 3
            and not trace_audit["completed_command_executions"]
            and not trace_audit["native_view_deliveries"]
            and not trace_audit["legacy_ag_shell_commands"]
            and not trace_audit["context_compaction_events"]
            and (ACTIVE_ARM not in {"A1", "A2"} or notes_bytes == b"")
            and notes_meta["structural_pass"]
        )
        if (
            not process_group_terminated
            or counters["correct"] is not None
            or not move_transport_consistent
            or not submit_transport_consistent
            or trace_audit["malformed_jsonl_lines"]
            or evaluator.transport_security_events
        ):
            raise RuntimeError(
                "A1 native ImageContent diagnostic failed: "
                + json.dumps(
                    {
                        "returncode": returncode,
                        "counters": counters,
                        "normalized_submission": normalized_submission,
                        "expected_nonce": visual_nonce,
                        "trace": trace_audit,
                    },
                    default=str,
                )[:4000]
            )
        result = {
            "passed": True,
            "unscored": True,
            "protocol": IMAGE_DELIVERY_PROTOCOL,
            "profile_sha256": sha256_bytes(profile.encode("utf-8")),
            "codex_login_under_profile": login_text,
            "strict_config_sentinel_rejected_before_thread": True,
            "rendered_top_level_items": 1,
            "rendered_input_images": 1,
            "exact_arm_prompt": ACTIVE_ARM,
            "opaque_initial_image_name": bool(
                re.fullmatch(r"[0-9a-f]{32}\.png", initial_name)
            ),
            "opaque_move_image_name": True,
            "native_mcp_move_result": "one text path plus one image/png block",
            "deterministic_transport_gate": "direct_local_toolchain",
            "terra_followed_diagnostic": terra_followed_diagnostic,
            "behavioral_result_is_launch_gate": False,
            "returncode": returncode,
            "timed_out": timed_out,
            "terra_tool_calls_observed": trace_audit["total_codex_tool_calls"],
            "successive_native_move_images_used": counters["accepted_moves"],
            "native_png_sha256": [
                delivery["image"]["png_sha256"]
                for delivery in move_deliveries
                if delivery.get("passed") is True
            ],
            "native_pixel_sha256": [
                delivery["image"]["pixel_sha256"]
                for delivery in move_deliveries
                if delivery.get("passed") is True
            ],
            "controller_png_sha256": [row["sha256"] for row in expected_moves],
            "controller_pixel_sha256": [
                row["pixel_sha256"] for row in expected_moves
            ],
            "archived_diagnostic_images": archived_diagnostic_images,
            "behavioral_accessibility_assay": {
                "infrastructure_gate": False,
                "expected_nonce": visual_nonce,
                "normalized_submission": normalized_submission,
                "expected_segments": expected_segments,
                "submitted_segments": submitted_segments,
                "segment_exact": segment_exact,
                "character_matches": character_matches,
                "character_total": len(visual_nonce),
                "full_sequence_exact": normalized_submission == visual_nonce,
                "initial_segment_exact_after_two_moves": bool(segment_exact[0]),
                "interpretation": (
                    "behavioral evidence of visual accessibility and retention; "
                    "not a transport-integrity condition"
                ),
            },
            "shell_or_viewer_calls": (
                len(trace_audit["completed_command_executions"])
                + len(trace_audit["native_view_deliveries"])
            ),
            "context_compaction_events": len(
                trace_audit["context_compaction_events"]
            ),
            "mcp_move_round_trip": (
                "passed" if counters["accepted_moves"] else "not_exercised_by_terra"
            ),
            "mcp_submit_round_trip": (
                "passed" if counters["submit_calls"] else "not_exercised_by_terra"
            ),
            "live_correctness_value": None,
            "notes_unchanged_or_not_applicable": bool(
                ACTIVE_ARM not in {"A1", "A2"} or notes_bytes == b""
            ),
            "agent_visible_private_markers": [],
        }
    finally:
        if process is not None:
            terminate_process_group(process)
        if evaluator is not None:
            evaluator.close()
        proxy.stop()
        proxy_events = proxy.snapshot()
        try:
            auth_update = adopt_runtime_auth(
                runtime, minimal_auth, vault, required_access_expiry
            )
        except Exception as error:
            raise RuntimeError(
                f"A1 integrated runtime preserved after auth adoption failure: {runtime}"
            ) from error
        else:
            remove_runtime(runtime)
        shutil.rmtree(workspace)
    result["proxy_events"] = proxy_events
    result["auth_state_update"] = auth_update
    return result


def a1_viewer_fusion_preflight(
    run_root: Path,
    generic_root: Path,
    minimal_auth: dict[str, Any],
    vault: AuthStateVault,
    required_access_expiry: float,
) -> dict[str, Any]:
    """Prove absolute-path viewing and private scratch-image creation."""
    workspace = Path("/private/tmp") / secrets.token_hex(16)
    observation_dir = workspace / secrets.token_hex(16)
    observation_dir.mkdir(parents=True, mode=0o700)
    agent_tmp = workspace / ".tmp"
    agent_tmp.mkdir(mode=0o700)
    if ACTIVE_ARM in {"A1", "A2"}:
        write_initial_notes(workspace, b"")
    initial_path = observation_dir / f"{secrets.token_hex(16)}.png"
    composite_path = agent_tmp / f"{secrets.token_hex(16)}.png"
    runtime = create_runtime(minimal_auth, "mnistpro-a1-probe-")
    proxy = AuthenticatedConnectProxy()
    evaluator: HiddenEvaluator | None = None
    auth_update: dict[str, Any] | None = None
    proxy_events: list[dict[str, Any]] = []
    process: subprocess.Popen[bytes] | None = None
    trace_path = run_root / "preflight/viewer_fusion.jsonl"
    stderr_path = run_root / "preflight/viewer_fusion.stderr.log"
    archived_composite = run_root / "preflight/viewer_fusion_composite.png"
    result: dict[str, Any] = {}
    timed_out = False
    try:
        letters = list("RGBYMC")
        secrets.SystemRandom().shuffle(letters)
        visual_nonce = "".join(letters)
        palette = {
            "R": (255, 0, 0),
            "G": (0, 255, 0),
            "B": (0, 0, 255),
            "Y": (255, 255, 0),
            "M": (255, 0, 255),
            "C": (0, 255, 255),
        }

        def make_panel(panel_letters: list[str]) -> Image.Image:
            pixels = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
            for index, letter in enumerate(panel_letters):
                start = index * IMAGE_SIZE // len(panel_letters)
                end = (index + 1) * IMAGE_SIZE // len(panel_letters)
                pixels[:, start:end, :] = palette[letter]
            return Image.fromarray(pixels, mode="RGB")

        panels = [make_panel(letters[:3]), make_panel(letters[3:])]
        initial = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (128, 128, 128))
        initial.save(initial_path, format="PNG")
        expected_composite = np.concatenate(
            [np.asarray(panel.convert("RGB")) for panel in panels], axis=1
        )
        expected_composite_pixel_sha256 = sha256_bytes(
            expected_composite.tobytes()
        )

        class ViewerFusionEnvironment:
            x = 0
            y = 0

            def __init__(self) -> None:
                self.calls = 0

            def step(
                self, action: str
            ) -> tuple[Image.Image, float, bool, dict[str, Any]]:
                panel = panels[min(self.calls, len(panels) - 1)].copy()
                self.calls += 1
                self.x = STEP_SIZE * self.calls
                return panel, 0.0, False, {}

        token = secrets.token_hex(32)
        evaluator = HiddenEvaluator(
            ViewerFusionEnvironment(),
            "unscored-viewer-fusion-diagnostic",
            BUDGETS[1],
            workspace,
            token,
            observation_dir,
        )
        evaluator.record_initial(initial_path)
        profile = external_profile(
            workspace, runtime, generic_root, proxy.port, observation_dir
        )
        outer_environment = scored_parent_environment(
            runtime, proxy.url, workspace, observation_dir, token
        )
        diagnostic_prompt = (
            "This is a non-scored image-viewer and private-scratch diagnostic. "
            "Call `activeglimpse.move` with direction `right`. Its text block is "
            "an absolute PNG path; call `activeglimpse.view_image` on that exact "
            "path. Then call "
            "`activeglimpse.move` with direction `down` and view its absolute "
            "PNG path with `activeglimpse.view_image`. Use a shell command with "
            "ImageMagick's `magick` executable "
            "to read those two files in move order, concatenate them "
            "horizontally without rescaling, and save the result exactly at "
            f"`{composite_path}`. Call `activeglimpse.view_image` on that exact "
            "composite path. This diagnostic fails unless all three explicit "
            "view-image calls complete. Read its six vertical color bands from left to right "
            "using only R, G, B, Y, M, and C. Finally call "
            "`activeglimpse.submit` exactly once with the six letters in `value`. "
            "Do not edit NOTES.md and do not access any path outside this "
            "workspace or its private temporary directory."
        )
        argv = external_sandbox_prefix(profile) + build_exec_argv(
            workspace,
            generic_root,
            token,
            1,
            initial_path,
            prompt_override=diagnostic_prompt,
        )
        with trace_path.open("wb") as trace_handle, stderr_path.open(
            "wb"
        ) as stderr_handle:
            process = subprocess.Popen(
                argv,
                cwd=workspace,
                env=outer_environment,
                stdin=subprocess.DEVNULL,
                stdout=trace_handle,
                stderr=stderr_handle,
                start_new_session=True,
            )
            deadline = time.monotonic() + 7 * 60
            while process.poll() is None:
                evaluator.handle_pending()
                if time.monotonic() >= deadline:
                    timed_out = True
                    break
                time.sleep(0.025)
            evaluator.handle_pending()
        returncode, process_group_terminated = terminate_process_group(process)
        process = None
        trace_audit = parse_jsonl_trace(trace_path)
        counters = evaluator.counters()
        notes_bytes, notes_meta = (
            acquire_notes_candidate(workspace)
            if ACTIVE_ARM in {"A1", "A2"}
            else (None, {"structural_pass": True})
        )
        expected_moves = evaluator.observation_hashes[1:]
        move_deliveries = trace_audit["native_move_deliveries"]
        normalized_submission = re.sub(
            r"\s+", "", evaluator.submitted_answer_raw or ""
        ).upper()
        stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")

        composite_bytes: bytes | None = None
        composite_metadata: dict[str, Any] = {}
        try:
            descriptor = os.open(composite_path, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                before = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(before.st_mode)
                    or before.st_uid != os.getuid()
                    or before.st_nlink != 1
                    or before.st_size > 8_000_000
                ):
                    raise RuntimeError("unsafe viewer/fusion composite")
                chunks: list[bytes] = []
                while True:
                    chunk = os.read(descriptor, 1024 * 1024)
                    if not chunk:
                        break
                    chunks.append(chunk)
                after = os.fstat(descriptor)
                if (before.st_dev, before.st_ino, before.st_size) != (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                ):
                    raise RuntimeError("viewer/fusion composite changed while read")
                composite_bytes = b"".join(chunks)
            finally:
                os.close(descriptor)
            with Image.open(io.BytesIO(composite_bytes)) as composite:
                composite.load()
                composite_metadata = {
                    "format": composite.format,
                    "width": composite.width,
                    "height": composite.height,
                    "png_sha256": sha256_bytes(composite_bytes),
                    "pixel_sha256": sha256_bytes(
                        np.asarray(composite.convert("RGB")).tobytes()
                    ),
                }
        except (OSError, RuntimeError):
            composite_bytes = None

        view_calls = trace_audit["native_view_deliveries"]
        viewed_paths = {
            path
            for call in view_calls
            for path in (call.get("agent_visible_path"),)
            if isinstance(path, str)
        }
        source_paths = [row["agent_visible_path"] for row in expected_moves]
        command_blob = "\n".join(trace_audit["command_texts"])
        imagemagick_path_invocation = bool(
            re.search(r"(?<![/A-Za-z0-9_.-])magick(?:\s|$)", command_blob)
        )
        move_transport_consistent = bool(
            len(
                [delivery for delivery in move_deliveries if delivery.get("passed") is True]
            )
            == len(expected_moves)
            == counters["accepted_moves"]
            == counters["native_image_delivery_receipts"]
            and [
                delivery["agent_visible_path"]
                for delivery in move_deliveries
                if delivery.get("passed") is True
            ]
            == source_paths
            and all(
                delivery["image"].get("png_sha256") == expected["sha256"]
                and delivery["image"].get("pixel_sha256")
                == expected["pixel_sha256"]
                for delivery, expected in zip(
                    (
                        row
                        for row in move_deliveries
                        if row.get("passed") is True
                    ),
                    expected_moves,
                )
            )
        )
        accepted_submit_calls = sum(
            row.get("operation") == "submit" and row.get("accepted") is True
            for row in evaluator.request_log
        )
        submit_transport_consistent = bool(
            len(
                [
                    receipt
                    for receipt in trace_audit["native_submit_receipts"]
                    if receipt.get("passed") is True
                ]
            )
            == accepted_submit_calls
            and all(
                receipt.get("receipt_text") == "submission accepted"
                for receipt in trace_audit["native_submit_receipts"]
                if receipt.get("passed") is True
            )
        )
        expected_view_paths = (
            source_paths + [str(composite_path)] if len(source_paths) == 2 else []
        )
        expected_view_hashes = {
            **{
                row["agent_visible_path"]: row["sha256"]
                for row in expected_moves
            },
            str(composite_path): composite_metadata.get("png_sha256"),
        }
        viewer_match = bool(
            len(expected_view_paths) == 3
            and
            len(view_calls) == 3
            and all(call.get("passed") is True for call in view_calls)
            and [call.get("agent_visible_path") for call in view_calls]
            == expected_view_paths
            and set(expected_view_paths) == viewed_paths
            and all(
                isinstance(call.get("agent_visible_path"), str)
                and call["agent_visible_path"] in expected_view_hashes
                and call.get("image", {}).get("png_sha256")
                == expected_view_hashes[call["agent_visible_path"]]
                for call in view_calls
            )
            and "unable to locate image" not in stderr_text.lower()
            and "view_image.detail" not in stderr_text.lower()
        )
        command_match = bool(
            len(source_paths) == 2
            and
            all(path in command_blob for path in source_paths)
            and str(composite_path) in command_blob
            and imagemagick_path_invocation
            and trace_audit["completed_command_executions"]
        )
        composite_match = bool(
            composite_bytes is not None
            and composite_metadata.get("format") == "PNG"
            and composite_metadata.get("width") == 2 * IMAGE_SIZE
            and composite_metadata.get("height") == IMAGE_SIZE
            and composite_metadata.get("pixel_sha256")
            == expected_composite_pixel_sha256
        )
        terra_followed_diagnostic = bool(
            returncode == 0
            and not timed_out
            and counters["accepted_moves"] == 2
            and counters["submit_calls"] == 1
            and move_transport_consistent
            and submit_transport_consistent
            and viewer_match
            and command_match
            and composite_match
            and not trace_audit["legacy_ag_shell_commands"]
            and not trace_audit["context_compaction_events"]
            and (ACTIVE_ARM not in {"A1", "A2"} or notes_bytes == b"")
            and notes_meta["structural_pass"]
        )
        if (
            not process_group_terminated
            or counters["correct"] is not None
            or not move_transport_consistent
            or not submit_transport_consistent
            or trace_audit["malformed_jsonl_lines"]
            or evaluator.transport_security_events
        ):
            raise RuntimeError(
                "A1 image-viewer/fusion diagnostic failed: "
                + json.dumps(
                    {
                        "returncode": returncode,
                        "counters": counters,
                        "normalized_submission": normalized_submission,
                        "expected_nonce": visual_nonce,
                        "move_transport_consistent": move_transport_consistent,
                        "submit_transport_consistent": submit_transport_consistent,
                        "viewer_match": viewer_match,
                        "command_match": command_match,
                        "composite_match": composite_match,
                        "view_calls": view_calls,
                        "commands": trace_audit[
                            "completed_command_executions"
                        ],
                    },
                    default=str,
                )[:5000]
            )
        if composite_bytes is not None:
            archived_composite.write_bytes(composite_bytes)
        result = {
            "passed": True,
            "unscored": True,
            "protocol": IMAGE_DELIVERY_PROTOCOL,
            "deterministic_capability_gate": "deterministic_local_toolchain",
            "terra_followed_diagnostic": terra_followed_diagnostic,
            "behavioral_result_is_launch_gate": False,
            "returncode": returncode,
            "timed_out": timed_out,
            "private_scratch_root": ".tmp inside the fresh opaque workspace",
            "shared_tmp_tree_readable": False,
            "native_move_results_hash_matched": sum(
                1 for delivery in move_deliveries if delivery.get("passed") is True
            ),
            "absolute_source_paths_viewed": sum(
                1 for path in source_paths if path in viewed_paths
            ),
            "generated_composite_viewed": str(composite_path) in viewed_paths,
            "completed_image_view_calls": len(view_calls),
            "completed_command_executions": len(
                trace_audit["completed_command_executions"]
            ),
            "imagemagick_resolved_from_model_shell_path": (
                imagemagick_path_invocation
            ),
            "behavioral_visual_readout": {
                "infrastructure_gate": False,
                "expected_nonce": visual_nonce,
                "normalized_submission": normalized_submission,
                "full_sequence_exact": normalized_submission == visual_nonce,
                "interpretation": (
                    "behavioral readout only; path resolution, byte delivery, "
                    "assembly, and viewing are verified independently by hashes"
                ),
            },
            "composite_width": composite_metadata.get("width"),
            "composite_height": composite_metadata.get("height"),
            "composite_png_sha256": composite_metadata.get("png_sha256"),
            "composite_pixel_sha256": composite_metadata.get("pixel_sha256"),
            "expected_composite_pixel_sha256": expected_composite_pixel_sha256,
            "archived_composite": (
                str(archived_composite) if composite_bytes is not None else None
            ),
            "trace": str(trace_path),
            "stderr": str(stderr_path),
            "context_compaction_events_observed": len(
                trace_audit["context_compaction_events"]
            ),
        }
    finally:
        if process is not None:
            terminate_process_group(process)
        if evaluator is not None:
            evaluator.close()
        proxy.stop()
        proxy_events = proxy.snapshot()
        try:
            auth_update = adopt_runtime_auth(
                runtime, minimal_auth, vault, required_access_expiry
            )
        except Exception as error:
            raise RuntimeError(
                "A1 viewer/fusion runtime preserved after auth adoption failure: "
                f"{runtime}"
            ) from error
        else:
            remove_runtime(runtime)
        shutil.rmtree(workspace)
    result["proxy_events"] = proxy_events
    result["auth_state_update"] = auth_update
    return result


def run_one_arm_episode(
    spec: EpisodeSpec,
    execution_ordinal: int,
    approved_notes: bytes,
    module: Any,
    image_cache: dict[int, tuple[Image.Image, int]],
    run_root: Path,
    generic_root: Path,
    minimal_auth: dict[str, Any],
    vault: AuthStateVault,
    required_access_expiry: float,
    prompt_audit: dict[str, Any],
    journal: Journal,
    timeout_seconds: int,
) -> tuple[dict[str, Any], bytes]:
    """Run one fresh Terra session and commit only the arm-authorized state."""
    started = utc_now()
    notes_enabled = ACTIVE_ARM in {"A1", "A2"}
    collision_paths = [
        run_root / "attempts" / f"{spec.key}.json",
        run_root / "traces" / f"{spec.key}.jsonl",
        run_root / "traces" / f"{spec.key}.stderr.log",
        run_root / "slot_commits" / f"slot_{execution_ordinal + 1:03d}.json",
    ]
    if notes_enabled:
        collision_paths.extend(
            (
                run_root
                / "notes_history"
                / f"ep_{execution_ordinal + 1:03d}.md",
                run_root
                / "notes_candidates"
                / f"ep_{execution_ordinal + 1:03d}.raw",
                run_root
                / "leakage_audits"
                / f"ep_{execution_ordinal + 1:03d}.json",
            )
        )
        versions_path = run_root / "notes_versions.jsonl"
        version_lines = (
            versions_path.read_bytes().splitlines() if versions_path.is_file() else []
        )
        if len(version_lines) != execution_ordinal:
            raise RuntimeError(
                "notes version journal ordinal collision before fresh launch"
            )
        if version_lines:
            try:
                last_version = json.loads(version_lines[-1])
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise RuntimeError("notes version journal is malformed") from error
            if last_version.get("execution_ordinal_zero_based") != execution_ordinal - 1:
                raise RuntimeError(
                    "notes version journal is not a contiguous exposure prefix"
                )
    collided = [
        str(path)
        for path in collision_paths
        if path.exists() or path.is_symlink()
    ]
    if collided:
        raise RuntimeError(
            "fresh-slot artifact collision before model launch: " + repr(collided)
        )
    workspace_uuid = secrets.token_hex(16)
    workspace = Path("/private/tmp") / workspace_uuid
    archive_workspace = run_root / "workspaces" / workspace_uuid
    observation_dir = workspace / secrets.token_hex(16)
    observation_dir.mkdir(parents=True, mode=0o700)
    agent_tmp = workspace / ".tmp"
    agent_tmp.mkdir(mode=0o700)
    notes_path = write_initial_notes(workspace, approved_notes) if notes_enabled else None
    initial_name = secrets.token_hex(16) + ".png"
    initial_path = observation_dir / initial_name
    live_trace = Path("/private/tmp") / f"{secrets.token_hex(16)}.jsonl"
    live_stderr = Path("/private/tmp") / f"{secrets.token_hex(16)}.log"
    trace_path = run_root / "traces" / f"{spec.key}.jsonl"
    stderr_path = run_root / "traces" / f"{spec.key}.stderr.log"
    isolation_log = run_root / "preflight/episodes" / f"slot_{execution_ordinal:03d}.log"
    attempt_path = run_root / "attempts" / f"{spec.key}.json"
    active_process_record = run_root / "ACTIVE_MODEL_PROCESS.json"
    if active_process_record.exists() or active_process_record.is_symlink():
        raise RuntimeError("stale active model-process record exists before launch")
    runtime: Path | None = None
    proxy: AuthenticatedConnectProxy | None = None
    process: subprocess.Popen[bytes] | None = None
    evaluator: HiddenEvaluator | None = None
    model_launches = 0
    process_pid: int | None = None
    process_group_terminated = True
    timed_out = False
    returncode: int | None = None
    isolation: dict[str, Any] = {"passed": False}
    exception_text: str | None = None
    prelaunch_files: list[dict[str, Any]] = []
    runtime_inventory: list[str] = []
    auth_state_update: dict[str, Any] | None = None
    proxy_events: list[dict[str, Any]] = []
    scored_profile: str | None = None
    prompt_hash = sha256_bytes(prompt_for(1).encode("utf-8"))
    active_surface_hash: str | None = None

    journal.append(
        {
            "event": "episode_preparing",
            "execution_ordinal": execution_ordinal,
            "level": 1,
            "episode_id": spec.episode_id,
            "workspace_uuid": workspace_uuid,
            "arm": ACTIVE_ARM,
            "incoming_notes_sha256": (
                sha256_bytes(approved_notes) if notes_enabled else None
            ),
        }
    )
    try:
        environment, hidden_label, initial = make_episode_environment(
            module, spec, image_cache
        )
        initial.save(initial_path, format="PNG")
        with Image.open(initial_path) as check_image:
            initial_metadata = dict(check_image.info)
        if initial_metadata:
            raise RuntimeError(f"initial PNG contains metadata: {sorted(initial_metadata)}")
        prelaunch_files = inventory_files(workspace)
        expected_paths = sorted(
            (["NOTES.md"] if notes_enabled else [])
            + [str(initial_path.relative_to(workspace))]
        )
        if [row["path"] for row in prelaunch_files] != expected_paths:
            raise RuntimeError(f"unexpected {ACTIVE_ARM} prelaunch files: {prelaunch_files}")
        if notes_path is not None and sha256_file(notes_path) != sha256_bytes(approved_notes):
            raise RuntimeError("incoming NOTES.md differs from the approved state")

        isolation = episode_isolation_probe(
            workspace,
            generic_root,
            minimal_auth,
            isolation_log,
            observation_dir,
        )
        if not isolation["passed"]:
            raise RuntimeError("episode isolation preflight failed")
        after_probe_files = inventory_files(workspace)
        if [row["path"] for row in after_probe_files] != expected_paths:
            raise RuntimeError("isolation probe left an agent-visible file")

        access_expiry = jwt_expiry(minimal_auth["tokens"]["access_token"])
        if access_expiry is None or access_expiry <= (
            time.time()
            + timeout_seconds
            + 15 * 60
        ):
            raise RuntimeError(
                f"access token lifetime is insufficient for this {ACTIVE_ARM} slot"
            )
        runtime = create_runtime(
            minimal_auth, f"mnistpro-{ACTIVE_ARM.lower()}-runtime-"
        )
        initial_home_files = [
            str(path.relative_to(runtime / "codex-home"))
            for path in (runtime / "codex-home").rglob("*")
            if path.is_file()
        ]
        if initial_home_files != ["auth.json"]:
            raise RuntimeError(
                f"fresh CODEX_HOME contained unexpected files: {initial_home_files}"
            )
        token = secrets.token_hex(32)
        evaluator = HiddenEvaluator(
            environment,
            hidden_label,
            BUDGETS[1],
            workspace,
            token,
            observation_dir,
            feedback_enabled=ACTIVE_ARM == "A2",
        )
        evaluator.record_initial(initial_path)
        exec_argv = build_exec_argv(
            workspace, generic_root, token, 1, initial_path
        )
        proxy = AuthenticatedConnectProxy()
        scored_profile = external_profile(
            workspace, runtime, generic_root, proxy.port, observation_dir
        )
        argv = external_sandbox_prefix(scored_profile) + exec_argv
        if prompt_hash != prompt_audit["prompt_sha256"]:
            raise RuntimeError(
                f"{ACTIVE_ARM} prompt hash differs from the clean prompt preflight"
            )
        visible_surface = {
            "cwd": str(workspace),
            "image_wrapper_path": str(initial_path),
            "workspace_leaf": workspace.name,
            "observation_directory_leaf": observation_dir.name,
            "initial_image_basename": initial_path.name,
            "prelaunch_paths": expected_paths,
            "prompt_sha256": prompt_hash,
        }
        visible_strings = "\n".join(
            [str(value) for value in visible_surface.values()] + exec_argv
        )
        leaked_markers = [
            marker
            for marker in (
                spec.key,
                f"episode_{spec.episode_id}",
                f"episode-{spec.episode_id}",
                "notes_history",
                "schedule_manifest",
                "round-robin",
                "balanced classes",
                str(run_root),
            )
            if marker in visible_strings
        ]
        if (
            leaked_markers
            or not re.fullmatch(r"[0-9a-f]{32}", workspace.name)
            or not re.fullmatch(r"[0-9a-f]{32}", observation_dir.name)
            or not re.fullmatch(r"[0-9a-f]{32}\.png", initial_path.name)
        ):
            raise RuntimeError(f"agent-visible naming leak: {leaked_markers}")
        active_surface_hash = sha256_bytes(canonical_json_bytes(visible_surface))

        journal.append(
            {
                "event": "episode_launching",
                "execution_ordinal": execution_ordinal,
                "level": 1,
                "episode_id": spec.episode_id,
                "workspace_uuid": workspace_uuid,
                "prompt_sha256": prompt_hash,
            }
        )
        with live_trace.open("wb") as trace_handle, live_stderr.open("wb") as stderr_handle:
            process = subprocess.Popen(
                argv,
                stdout=trace_handle,
                stderr=stderr_handle,
                stdin=subprocess.DEVNULL,
                env=scored_parent_environment(
                    runtime, proxy.url, workspace, observation_dir, token
                ),
                cwd=workspace,
                start_new_session=True,
            )
            model_launches = 1
            process_pid = process.pid
            write_json(
                active_process_record,
                {
                    "active": True,
                    "arm": ACTIVE_ARM,
                    "suite_id": ACTIVE_SUITE_ID,
                    "controller_pid": os.getpid(),
                    "child_pid": process.pid,
                    "child_pgid": process.pid,
                    "execution_ordinal": execution_ordinal,
                    "workspace_uuid": workspace_uuid,
                    "created_at": utc_now(),
                },
            )
            deadline = time.monotonic() + timeout_seconds
            while True:
                evaluator.handle_pending()
                returncode = process.poll()
                if returncode is not None:
                    evaluator.handle_pending()
                    break
                if time.monotonic() >= deadline:
                    timed_out = True
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        returncode = process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        returncode = process.wait(timeout=15)
                    evaluator.handle_pending()
                    break
                time.sleep(0.025)
    except Exception:
        exception_text = traceback.format_exc()
    finally:
        if process is not None:
            returncode, process_group_terminated = terminate_process_group(process)
        if process_group_terminated and active_process_record.exists():
            active_process_record.unlink()
        if proxy is not None:
            proxy.stop()
            proxy_events = proxy.snapshot()
        if evaluator is not None:
            evaluator.close()
        if runtime is not None and runtime.exists():
            runtime_inventory = [
                str(path.relative_to(runtime / "codex-home"))
                for path in (runtime / "codex-home").rglob("*")
                if path.is_file() and path.name != "auth.json"
            ]
            try:
                auth_state_update = adopt_runtime_auth(
                    runtime, minimal_auth, vault, required_access_expiry
                )
            except Exception as error:
                raise RuntimeError(
                    f"episode runtime preserved after auth adoption failure: {runtime}"
                ) from error
            else:
                remove_runtime(runtime)

    if not process_group_terminated:
        raise RuntimeError("scored process group survived; NOTES.md cannot be trusted")
    if not live_trace.exists():
        live_trace.touch()
    if not live_stderr.exists():
        live_stderr.write_text(exception_text or "", encoding="utf-8")
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(live_trace), str(trace_path))
    shutil.move(str(live_stderr), str(stderr_path))

    if notes_enabled:
        next_approved_notes, notes_result = observe_and_snapshot_notes(
            workspace,
            approved_notes,
            run_root,
            execution_ordinal,
        )
        if notes_result["previous_state_sha256"] != sha256_bytes(approved_notes):
            raise RuntimeError("notes transition input hash changed")
        if (
            notes_result["history_snapshot_sha256"]
            != notes_result["next_state_sha256"]
        ):
            raise RuntimeError("notes snapshot does not equal carried state")
    else:
        next_approved_notes = b""
        notes_result = {
            "applicable": False,
            "content_validator_applied": False,
            "history_snapshot": None,
            "history_snapshot_sha256": None,
            "next_state_sha256": None,
            "state_eligible": True,
        }

    if evaluator is not None:
        evaluator.finalize_correctness()
    trace_audit = parse_jsonl_trace(trace_path)
    final_files = inventory_files(workspace)
    evaluator_paths = (
        {row["path"] for row in evaluator.observation_hashes} | evaluator.response_paths
        if evaluator is not None
        else set()
    )
    transport_request_paths = (
        evaluator.interface_request_paths if evaluator is not None else set()
    )
    note_paths = {
        row["path"]
        for row in final_files
        if notes_enabled and row["path"] == "NOTES.md"
    }
    autonomous_files = [
        row
        for row in final_files
        if row["path"] not in evaluator_paths
        and row["path"] not in transport_request_paths
        and row["path"] not in note_paths
    ]
    spatial = inspect_spatial_artifacts(workspace, autonomous_files, 1)
    counters = evaluator.counters() if evaluator is not None else {
        "glimpses_exposed": 0,
        "move_requests": 0,
        "accepted_moves": 0,
        "effective_moves": 0,
        "effective_revisits": 0,
        "wall_bumps": 0,
        "revisit_rate": None,
        "wall_bump_rate": None,
        "position_history": [],
        "unique_positions": 0,
        "submit_calls": 0,
        "submitted_answer_raw": None,
        "submitted_answer": None,
        "correct": None,
        "budget_exhausted": False,
        "budget_rejections": 0,
        "invalid_requests": 0,
    }
    resolved_models = set(trace_audit["resolved_model_candidates"])
    resolved_model_consistent = not resolved_models or resolved_models == {MODEL}
    denied_network_attempts = [
        event
        for event in proxy_events
        if event.get("reason") in {"proxy_auth_required", "target_blocked"}
    ]
    successful_forbidden_network = any(
        event.get("allowed") is True and event.get("authorized") is not True
        for event in proxy_events
    )
    transport_security_events = (
        evaluator.transport_security_events if evaluator is not None else []
    )
    image_paths = [
        row["agent_visible_path"] for row in evaluator.observation_hashes
    ] if evaluator else []
    opaque_image_paths = bool(image_paths) and len(image_paths) == len(set(image_paths)) and all(
        re.fullmatch(
            rf"{re.escape(str(observation_dir))}/[0-9a-f]{{32}}\.png", path
        )
        for path in image_paths
    )
    expected_move_images = {
        row["agent_visible_path"]: row
        for row in (evaluator.observation_hashes[1:] if evaluator else [])
    }
    successful_native_deliveries = [
        row for row in trace_audit["native_move_deliveries"] if row["passed"]
    ]
    observed_native_by_path = {
        row.get("agent_visible_path"): row for row in successful_native_deliveries
    }
    native_delivery_hashes_match = bool(
        evaluator is not None
        and len(successful_native_deliveries) == evaluator.accepted_moves
        and len(evaluator.delivery_receipts) == evaluator.accepted_moves
        and set(observed_native_by_path) == set(expected_move_images)
        and all(
            observed_native_by_path[path]["image"].get("png_sha256")
            == expected["sha256"]
            and observed_native_by_path[path]["image"].get("pixel_sha256")
            == expected["pixel_sha256"]
            and observed_native_by_path[path]["subsequent_model_activity_observed"]
            for path, expected in expected_move_images.items()
        )
    )
    expected_submit_receipt = (
        ("CORRECT" if counters["correct"] else "INCORRECT")
        if ACTIVE_ARM == "A2" and counters["correct"] is not None
        else "submission accepted"
    )
    observed_submit_receipts = [
        row.get("receipt_text") for row in trace_audit["native_submit_receipts"]
    ]
    feedback_interface_matches_arm = bool(
        counters["submit_calls"] != 1
        or observed_submit_receipts == [expected_submit_receipt]
    )
    notes_snapshot_exact = bool(
        not notes_enabled
        or (
            notes_result.get("state_eligible") is True
            and notes_result.get("history_snapshot_sha256")
            == notes_result.get("next_state_sha256")
        )
    )
    validity_conditions = {
        "external_filesystem_isolation_passed": bool(isolation.get("passed")),
        "exactly_one_scored_codex_launch": model_launches == 1,
        "exactly_one_fresh_thread": trace_audit["thread_started_count"] == 1,
        "requested_model_exact": MODEL == "gpt-5.6-terra",
        "resolved_model_consistent_if_reported": resolved_model_consistent,
        "reasoning_effort_exact": REASONING_EFFORT == "medium",
        "exact_arm_prompt_and_one_initial_image": (
            prompt_hash == prompt_audit["prompt_sha256"]
            and not prompt_audit["forbidden_markers"]
        ),
        "opaque_agent_visible_image_paths": opaque_image_paths,
        "guaranteed_native_image_delivery": native_delivery_hashes_match,
        "native_delivery_receipts_match_accepted_moves": bool(
            evaluator is not None
            and len(evaluator.delivery_receipts) == evaluator.accepted_moves
        ),
        "no_legacy_path_only_ag_commands": not trace_audit[
            "legacy_ag_shell_commands"
        ],
        "ephemeral_no_resume": True,
        "no_forbidden_access_observed": (
            not trace_audit["forbidden_host_paths_in_model_commands"]
            and not successful_forbidden_network
            and not transport_security_events
        ),
        "model_tool_general_network_disabled": True,
        "exactly_one_submission": counters["submit_calls"] == 1,
        "exactly_one_native_submit_receipt": bool(
            len(trace_audit["native_submit_receipts"]) == 1
            and trace_audit["native_submit_receipts"][0]["passed"]
        ),
        "native_submit_transport_consistent": bool(
            len(trace_audit["native_submit_receipts"])
            == counters["submit_calls"]
            and all(
                receipt["passed"]
                for receipt in trace_audit["native_submit_receipts"]
            )
        ),
        "authoritative_hidden_evaluator_counters": evaluator is not None,
        "process_not_timed_out": not timed_out,
        "codex_exit_successful": returncode == 0,
        "jsonl_well_formed": trace_audit["malformed_jsonl_lines"] == 0,
        "scored_process_group_terminated": process_group_terminated,
        "controller_exception_absent": exception_text is None,
        "notes_snapshot_exact_or_not_applicable": notes_snapshot_exact,
        "feedback_interface_matches_arm": feedback_interface_matches_arm,
        "correctness_timing_matches_arm": bool(
            evaluator is not None
            and evaluator.feedback_enabled == (ACTIVE_ARM == "A2")
        ),
        "concurrent_auth_state_unchanged": bool(
            auth_state_update is not None
            and auth_state_update.get("credential_material_changed") is False
        ),
    }
    valid = all(validity_conditions.values())
    shutil.move(str(workspace), str(archive_workspace))
    report = {
        "attempt_key": spec.key,
        "execution_ordinal": execution_ordinal,
        "validity": "VALID" if valid else "INVALID",
        "validity_conditions": validity_conditions,
        "configuration": {
            "level": 1,
            "arm": ACTIVE_ARM,
            "episode_id": spec.episode_id,
            "seed": SEED,
            "num_eval_sets": NUM_EVAL_SETS,
            "num_digits": 1,
            "image_size": IMAGE_SIZE,
            "box_size": BOX_SIZE,
            "step_size": STEP_SIZE,
            "observation_budget": BUDGETS[1],
            "requested_model": MODEL,
            "resolved_model": (
                trace_audit["resolved_model_candidates"][0]
                if len(trace_audit["resolved_model_candidates"]) == 1
                else None
            ),
            "reasoning_effort": REASONING_EFFORT,
            "fresh_process_session_codex_home_workspace": True,
        },
        "schedule": {
            "controller_only_indices": list(spec.indices),
            "source_manifest_sha256": PINNED_HASHES["level1_summary"],
        },
        "prompt": {
            "sha256": prompt_hash,
            "utf8_bytes": len(prompt_for(1).encode("utf-8")),
            "original_cleanroom_prompt_sha256": sha256_bytes(
                baseline_prompt_for(1).encode("utf-8")
            ),
            "native_image_common_prompt_sha256": sha256_bytes(
                native_image_prompt_for(1).encode("utf-8")
            ),
            "transport_prompt_amendment": IMAGE_DELIVERY_PROTOCOL,
            "notes_instruction_sha256": (
                sha256_bytes(notes_instruction_for(ACTIVE_ARM).encode("utf-8"))
                if notes_instruction_for(ACTIVE_ARM) is not None else None
            ),
            "global_clean_prompt_audit": prompt_audit,
        },
        "agent_visible_surface": {
            "sha256": active_surface_hash,
            "workspace_leaf_format": "32 lowercase hexadecimal characters",
            "observation_directory_format": "32 lowercase hexadecimal characters",
            "controller_created_image_basename_format": "32 lowercase hexadecimal characters plus .png",
            "source_episode_markers_detected": [],
            "initial_png_metadata": {},
            "live_trace_and_stderr_basenames_opaque": True,
            "image_delivery_protocol": IMAGE_DELIVERY_PROTOCOL,
        },
        "persistent_notes": notes_result,
        "isolation": {
            **isolation,
            "outer_sandbox": "macOS Seatbelt via sandbox-exec",
            "fresh_codex_home_initial_files": ["auth.json"],
            "fresh_codex_home_runtime_non_auth_files": runtime_inventory,
            "auth_state_update_after_episode": auth_state_update,
            "isolation_violations": transport_security_events,
            "denied_or_forbidden_model_path_attempts": trace_audit[
                "forbidden_host_paths_in_model_commands"
            ],
            "network_proxy_events": proxy_events,
            "denied_network_attempts": denied_network_attempts,
            "proxy_credentials_in_model_shell_environment": False,
            "scored_profile_sha256": (
                sha256_bytes(scored_profile.encode("utf-8"))
                if scored_profile is not None
                else None
            ),
        },
        "process": {
            "scored_codex_launches": model_launches,
            "outer_launcher_pid": process_pid,
            "returncode": returncode,
            "timed_out": timed_out,
            "process_group_terminated": process_group_terminated,
            "started_at": started,
            "completed_at": utc_now(),
        },
        "counters": {
            **counters,
            "total_codex_tool_calls": trace_audit["total_codex_tool_calls"],
        },
        "answer": counters["submitted_answer"],
        "correctness": counters["correct"],
        "ground_truth_label": evaluator.hidden_label if evaluator is not None else None,
        "correctness_timing": (
            "one-bit verdict computed at accepted submit before post-submit model activity"
            if ACTIVE_ARM == "A2"
            else "computed controller-side after process reaping and any notes snapshot"
        ),
        "submit_receipt_expected": expected_submit_receipt,
        "submit_receipts_observed": observed_submit_receipts,
        "observation_hashes": evaluator.observation_hashes if evaluator is not None else [],
        "native_image_delivery": {
            "protocol": IMAGE_DELIVERY_PROTOCOL,
            "controller_delivery_receipts": (
                evaluator.delivery_receipts if evaluator is not None else []
            ),
            "trace_native_move_deliveries": trace_audit[
                "native_move_deliveries"
            ],
            "hashes_match": native_delivery_hashes_match,
            "context_compaction_events": trace_audit[
                "context_compaction_events"
            ],
        },
        "files": {
            "prelaunch": prelaunch_files,
            "terra_autonomous_files": autonomous_files,
            "notes_state_files": sorted(note_paths),
            "interface_transport_request_files": sorted(transport_request_paths),
            "evaluator_created_files": sorted(evaluator_paths),
            "final_workspace_file_count": len(final_files),
        },
        "spatial_artifact_observation": spatial,
        "trace_audit": trace_audit,
        "paths": {
            "workspace": str(archive_workspace),
            "jsonl_trace": str(trace_path),
            "stderr": str(stderr_path),
            "attempt_report": str(attempt_path),
        },
        "hidden_evaluator_request_log": evaluator.request_log if evaluator else [],
        "hidden_evaluator_transport_security_events": transport_security_events,
        "controller_exception": exception_text,
    }
    write_json_exclusive(attempt_path, report)
    write_json_exclusive(
        run_root / "slot_commits" / f"slot_{execution_ordinal + 1:03d}.json",
        {
            "schema_version": 1,
            "commit_kind": "fresh_scored_attempt",
            "arm": ACTIVE_ARM,
            "suite_id": ACTIVE_SUITE_ID,
            "execution_ordinal": execution_ordinal,
            "episode_id": spec.episode_id,
            "attempt_key": spec.key,
            "attempt_sha256": sha256_file(attempt_path),
            "destination_relative_path": str(attempt_path.relative_to(run_root)),
            "fresh_process_session_codex_home_workspace": True,
            "retry_of_prior_slot": False,
            "validity": report["validity"],
        },
    )
    journal.append(
        {
            "event": "episode_committed",
            "execution_ordinal": execution_ordinal,
            "level": 1,
            "episode_id": spec.episode_id,
            "workspace_uuid": workspace_uuid,
            "validity": report["validity"],
            "arm": ACTIVE_ARM,
            "notes_state_eligible": notes_result.get("state_eligible"),
            "notes_state_sha256": notes_result.get("next_state_sha256"),
            "attempt_report": str(attempt_path),
        }
    )
    print(
        f"COMMITTED arm={ACTIVE_ARM} slot={execution_ordinal + 1:03d} "
        f"validity={report['validity']} notes_state_eligible="
        f"{notes_result.get('state_eligible')}",
        flush=True,
    )
    return report, next_approved_notes


class ArmProcessLock:
    """Nonblocking global per-arm exclusion held for the controller lifetime."""

    def __init__(self, suite_id: str, arm: str):
        lock_parent = THREE_ARM_PRIVATE_ROOT / "locks"
        lock_parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(lock_parent, 0o700)
        self.path = lock_parent / f"active-{arm.lower()}.lock"
        self.handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self.handle.close()
            raise RuntimeError(
                f"another {arm} controller holds the global arm lock"
            ) from error
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(
            json.dumps(
                {"pid": os.getpid(), "arm": arm, "suite_id": suite_id},
                separators=(",", ":"),
            )
            + "\n"
        )
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def close(self) -> None:
        if self.handle.closed:
            return
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()


def active_three_arm_process_preflight() -> dict[str, Any]:
    """Reject legacy MNIST controllers while allowing matched sibling arms."""
    result = subprocess.run(
        ("/bin/ps", "-axo", "pid=,ppid=,pgid=,command="),
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    legacy_markers = (
        "work/mnist_pro_arm_a1/controller.py",
        "work/mnist_pro_arm_a2/controller.py",
        "work/mnist_pro_cleanroom/controller.py",
        "/private/tmp/mnistpro-arm-pair-live/",
        "AG_TOKEN=",
    )
    offenders: list[dict[str, Any]] = []
    sibling_arms: list[str] = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) != 4:
            continue
        pid_text, ppid_text, pgid_text, command = parts
        pid = int(pid_text)
        if pid == os.getpid():
            continue
        if "work/mnist_pro_three_arm/controller.py" in command:
            match = re.search(r"--arm\s+(A[012])\b", command)
            if match:
                sibling_arms.append(match.group(1))
            continue
        matched = [marker for marker in legacy_markers if marker in command]
        if matched:
            offenders.append(
                {
                    "pid": pid,
                    "ppid": int(ppid_text),
                    "pgid": int(pgid_text),
                    "matched_markers": matched,
                }
            )
    if offenders:
        raise RuntimeError(f"legacy MNIST process remains: {offenders}")
    return {
        "passed": True,
        "legacy_processes": [],
        "matched_sibling_arms_observed": sorted(set(sibling_arms)),
        "checked_at": utc_now(),
    }


def arm_feedback_interface_preflight() -> dict[str, Any]:
    """Exercise the exact hidden-evaluator submit responses without a model call."""
    rows: list[dict[str, Any]] = []
    for hidden_label in (7, 2):
        workspace = Path("/private/tmp") / secrets.token_hex(16)
        observation_dir = workspace / secrets.token_hex(16)
        observation_dir.mkdir(parents=True, mode=0o700)
        token = secrets.token_hex(32)

        class StaticEnvironment:
            x = 0
            y = 0

            def step(self, action: str) -> tuple[Image.Image, float, bool, dict[str, Any]]:
                _ = action
                return Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE)), 0.0, False, {}

        evaluator = HiddenEvaluator(
            StaticEnvironment(),
            hidden_label,
            BUDGETS[1],
            workspace,
            token,
            observation_dir,
            feedback_enabled=ACTIVE_ARM == "A2",
        )
        try:
            request_id = secrets.token_hex(16)
            response = evaluator.process_request(
                sign_episode_payload(
                    token,
                    {
                        "protocol": 1,
                        "request_id": request_id,
                        "operation": "submit",
                        "value": "7",
                    },
                ),
                request_id,
            )
            live_correctness = evaluator.correct
            evaluator.finalize_correctness()
            rows.append(
                {
                    "hidden_case": "matching" if hidden_label == 7 else "nonmatching",
                    "receipt": response.get("message"),
                    "live_correctness": live_correctness,
                    "final_correctness": evaluator.correct,
                }
            )
        finally:
            evaluator.close()
            shutil.rmtree(workspace)
    expected_receipts = (
        ["CORRECT", "INCORRECT"]
        if ACTIVE_ARM == "A2"
        else ["submission accepted", "submission accepted"]
    )
    if [row["receipt"] for row in rows] != expected_receipts:
        raise RuntimeError(f"{ACTIVE_ARM} submit receipt contract changed: {rows}")
    if ACTIVE_ARM == "A2":
        if [row["live_correctness"] for row in rows] != [True, False]:
            raise RuntimeError("A2 verdict was not computed at submit")
    elif any(row["live_correctness"] is not None for row in rows):
        raise RuntimeError(f"{ACTIVE_ARM} leaked live correctness")
    return {
        "passed": True,
        "arm": ACTIVE_ARM,
        "receipts": expected_receipts,
        "true_label_exposed": False,
        "confidence_or_distance_exposed": False,
        "cases": rows,
    }


def notes_persistence_preflight(run_root: Path) -> dict[str, Any]:
    """Prove that valid UTF-8 notes are carried exactly without a size cap."""
    if ACTIVE_ARM == "A0":
        return {"passed": True, "applicable": False, "notes_file_created": False}
    cases = {
        "answer_and_verdict": b"Previous answer 7 was CORRECT in episode_042.\n",
        "large_notes": (("large persistent note \u03bb\n" * 8192)).encode("utf-8"),
        "invalid_utf8": b"valid prefix\n\xff\xfe",
    }
    results: dict[str, Any] = {}
    for case_name, candidate in cases.items():
        workspace = Path("/private/tmp") / secrets.token_hex(16)
        case_root = run_root / "preflight/notes_persistence" / case_name
        workspace.mkdir(mode=0o700)
        write_initial_notes(workspace, candidate)
        try:
            next_state, result = observe_and_snapshot_notes(
                workspace, b"prior", case_root, 0
            )
        finally:
            shutil.rmtree(workspace)
        results[case_name] = {
            "state_eligible": result["state_eligible"],
            "carried_verbatim": result["carried_verbatim"],
            "candidate_sha256": result["candidate_sha256"],
            "next_state_sha256": result["next_state_sha256"],
            "size_cap_enforced": result["size_cap_enforced"],
            "unicode_codepoints": result["unicode_codepoints"],
            "utf8_bytes": result["utf8_bytes"],
            "raw_candidate_archived": bool(
                result["raw_structurally_invalid_candidate"]
            ),
            "observational_detector_rule_ids": result[
                "observational_leakage_test"
            ]["detector_rule_ids"],
            "next_state_equals_candidate": next_state == candidate,
        }
    if not (
        results["answer_and_verdict"]["next_state_equals_candidate"]
        and results["answer_and_verdict"]["state_eligible"]
        and results["large_notes"]["next_state_equals_candidate"]
        and results["large_notes"]["state_eligible"]
        and results["large_notes"]["size_cap_enforced"] is False
        and results["large_notes"]["utf8_bytes"] > 8000
        and results["large_notes"]["unicode_codepoints"] > 2000
        and not results["invalid_utf8"]["state_eligible"]
        and not results["invalid_utf8"]["next_state_equals_candidate"]
        and results["invalid_utf8"]["raw_candidate_archived"]
    ):
        raise RuntimeError(f"content-neutral notes persistence preflight failed: {results}")
    return {
        "passed": True,
        "applicable": True,
        "semantic_or_answer_content_gate": False,
        "size_cap_enforced": False,
        "large_notes_carried_verbatim": True,
        "invalid_utf8_policy": "archive raw candidate and retain prior state",
        "cases": results,
    }


def a2_post_submit_notes_preflight(
    run_root: Path,
    generic_root: Path,
    minimal_auth: dict[str, Any],
    vault: AuthStateVault,
    required_access_expiry: float,
) -> dict[str, Any]:
    """Prove both A2 verdicts can condition a later NOTES.md edit."""
    if ACTIVE_ARM != "A2":
        return {"passed": True, "applicable": False, "model_launches": 0}
    cases: list[dict[str, Any]] = []
    for case_name, hidden_label, expected_receipt in (
        ("matching", 7, "CORRECT"),
        ("nonmatching", 2, "INCORRECT"),
    ):
        workspace = Path("/private/tmp") / secrets.token_hex(16)
        observation_dir = workspace / secrets.token_hex(16)
        observation_dir.mkdir(parents=True, mode=0o700)
        (workspace / ".tmp").mkdir(mode=0o700)
        write_initial_notes(workspace, b"")
        initial_path = observation_dir / f"{secrets.token_hex(16)}.png"
        Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (128, 128, 128)).save(
            initial_path, format="PNG"
        )
        runtime = create_runtime(minimal_auth, "mnistpro-a2-probe-")
        proxy = AuthenticatedConnectProxy()
        evaluator: HiddenEvaluator | None = None
        process: subprocess.Popen[bytes] | None = None
        trace_path = run_root / f"preflight/a2_feedback_{case_name}.jsonl"
        stderr_path = run_root / f"preflight/a2_feedback_{case_name}.stderr.log"
        correct_marker = f"correct-{secrets.token_hex(12)}"
        incorrect_marker = f"incorrect-{secrets.token_hex(12)}"
        expected_marker = (
            correct_marker if expected_receipt == "CORRECT" else incorrect_marker
        )
        auth_update: dict[str, Any] | None = None
        proxy_events: list[dict[str, Any]] = []
        case_result: dict[str, Any] | None = None
        timed_out = False
        try:
            class StaticEnvironment:
                x = 0
                y = 0

                def step(
                    self, action: str
                ) -> tuple[Image.Image, float, bool, dict[str, Any]]:
                    _ = action
                    return Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE)), 0.0, False, {}

            token = secrets.token_hex(32)
            evaluator = HiddenEvaluator(
                StaticEnvironment(),
                hidden_label,
                BUDGETS[1],
                workspace,
                token,
                observation_dir,
                feedback_enabled=True,
            )
            evaluator.record_initial(initial_path)
            prompt = (
                "This is an unscored A2 feedback-transport diagnostic. Call "
                "`activeglimpse.submit` exactly once with `value` equal to 7. Read "
                "the tool's text receipt. After the receipt, run one shell command "
                "that writes exactly the following marker with no trailing newline "
                "to NOTES.md: if the receipt is CORRECT, write `"
                + correct_marker
                + "`; if it is INCORRECT, write `"
                + incorrect_marker
                + "`. Do not inspect the attached image, do not call move or "
                "view_image, and do not write either marker before the submit tool "
                "has returned. Then stop."
            )
            profile = external_profile(
                workspace, runtime, generic_root, proxy.port, observation_dir
            )
            argv = external_sandbox_prefix(profile) + build_exec_argv(
                workspace,
                generic_root,
                token,
                1,
                initial_path,
                prompt_override=prompt,
            )
            with trace_path.open("wb") as trace_handle, stderr_path.open(
                "wb"
            ) as stderr_handle:
                process = subprocess.Popen(
                    argv,
                    cwd=workspace,
                    env=scored_parent_environment(
                        runtime, proxy.url, workspace, observation_dir, token
                    ),
                    stdin=subprocess.DEVNULL,
                    stdout=trace_handle,
                    stderr=stderr_handle,
                    start_new_session=True,
                )
                deadline = time.monotonic() + 5 * 60
                while process.poll() is None:
                    evaluator.handle_pending()
                    if time.monotonic() >= deadline:
                        timed_out = True
                        break
                    time.sleep(0.025)
                evaluator.handle_pending()
            returncode, group_terminated = terminate_process_group(process)
            process = None
            trace = parse_jsonl_trace(trace_path)
            candidate, acquisition = acquire_notes_candidate(workspace)
            submit_receipts = trace["native_submit_receipts"]
            submit_event = (
                submit_receipts[0]["event_index"] if len(submit_receipts) == 1 else None
            )
            all_commands = trace["completed_command_executions"]
            post_submit_commands = [
                row
                for row in all_commands
                if submit_event is not None and row["event_index"] > submit_event
            ]
            notes_writing_commands = [
                row
                for row in post_submit_commands
                if isinstance(row.get("command"), str)
                and "NOTES.md" in row["command"]
                and expected_marker in row["command"]
                and row.get("status") == "completed"
                and row.get("exit_code") == 0
            ]
            pre_submit_commands = [
                row
                for row in all_commands
                if submit_event is None or row["event_index"] < submit_event
            ]
            behavioral_conditioning_observed = bool(
                returncode == 0
                and not timed_out
                and group_terminated
                and acquisition.get("structural_pass")
                and candidate == expected_marker.encode("utf-8")
                and evaluator.correct == (expected_receipt == "CORRECT")
                and len(submit_receipts) == 1
                and submit_receipts[0].get("passed") is True
                and submit_receipts[0].get("receipt_text") == expected_receipt
                and len(all_commands) == 1
                and len(notes_writing_commands) == 1
                and not pre_submit_commands
                and not trace["native_move_deliveries"]
                and not trace["native_view_deliveries"]
                and not trace["malformed_jsonl_lines"]
            )
            accepted_submit_calls = sum(
                row.get("operation") == "submit" and row.get("accepted") is True
                for row in evaluator.request_log
            )
            passed_submit_receipts = [
                receipt
                for receipt in submit_receipts
                if receipt.get("passed") is True
            ]
            submit_transport_consistent = bool(
                len(passed_submit_receipts) == accepted_submit_calls
                and all(
                    receipt.get("receipt_text") == expected_receipt
                    for receipt in passed_submit_receipts
                )
            )
            if (
                not group_terminated
                or trace["malformed_jsonl_lines"]
                or evaluator.transport_security_events
                or not submit_transport_consistent
            ):
                raise RuntimeError(
                    f"A2 post-submit diagnostic infrastructure failed for {case_name}: "
                    + json.dumps(
                        {
                            "returncode": returncode,
                            "group_terminated": group_terminated,
                            "receipt": submit_receipts,
                            "candidate_sha256": (
                                sha256_bytes(candidate) if candidate is not None else None
                            ),
                            "expected_sha256": sha256_bytes(
                                expected_marker.encode("utf-8")
                            ),
                            "all_commands": all_commands,
                            "notes_writing_commands": notes_writing_commands,
                            "pre_submit_commands": pre_submit_commands,
                        },
                        default=str,
                    )[:3000]
                )
            case_result = {
                    "case": case_name,
                    "receipt": expected_receipt,
                    "infrastructure_gate_passed": True,
                    "behavioral_conditioning_observed": behavioral_conditioning_observed,
                    "behavioral_result_is_launch_gate": False,
                    "timed_out": timed_out,
                    "returncode": returncode,
                    "expected_marker_sha256": sha256_bytes(
                        expected_marker.encode("utf-8")
                    ),
                    "candidate_sha256": (
                        sha256_bytes(candidate) if candidate is not None else None
                    ),
                    "notes_exact": candidate == expected_marker.encode("utf-8"),
                    "post_submit_command_observed": bool(notes_writing_commands),
                    "no_pre_submit_command_observed": not pre_submit_commands,
                    "submit_calls_observed": evaluator.submit_calls,
                    "submit_transport_consistent": submit_transport_consistent,
                    "thread_started_count": trace["thread_started_count"],
                    "context_compaction_events_observed": len(
                        trace["context_compaction_events"]
                    ),
                    "trace": str(trace_path),
                    "stderr": str(stderr_path),
                }
        finally:
            if process is not None:
                terminate_process_group(process)
            if evaluator is not None:
                evaluator.close()
            proxy.stop()
            proxy_events = proxy.snapshot()
            try:
                auth_update = adopt_runtime_auth(
                    runtime, minimal_auth, vault, required_access_expiry
                )
            except Exception as error:
                raise RuntimeError(
                    f"A2 feedback runtime preserved after auth failure: {runtime}"
                ) from error
            else:
                remove_runtime(runtime)
            shutil.rmtree(workspace)
        if case_result is None:
            raise RuntimeError(f"A2 {case_name} diagnostic produced no result")
        case_result["auth_state_update"] = auth_update
        case_result["proxy_events"] = proxy_events
        cases.append(case_result)
    return {
        "passed": True,
        "applicable": True,
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "model_launches": 2,
        "both_verdict_branches_tested": True,
        "post_submit_notes_update_observed": all(
            row["behavioral_conditioning_observed"] for row in cases
        ),
        "behavioral_conditioning_is_launch_gate": False,
        "true_label_exposed": False,
        "cases": cases,
    }


def write_arm_aggregate_outputs(
    run_root: Path,
    reports: list[dict[str, Any]],
    preflight: dict[str, Any],
    shuffled_specs: list[EpisodeSpec],
) -> dict[str, Any]:
    ordered = sorted(reports, key=lambda row: row["execution_ordinal"])
    ordinals_complete = [row["execution_ordinal"] for row in ordered] == list(range(100))
    schedule_matches = [row["configuration"]["episode_id"] for row in ordered] == [
        spec.episode_id for spec in shuffled_specs
    ]
    continuation = preflight.get("continuation", no_continuation_record())
    continuation_integrity = continuation.get("applicable") is False
    if continuation.get("applicable") is True:
        sentinel_relative = continuation.get("import_complete_sentinel")
        sentinel_path = (
            run_root / sentinel_relative
            if isinstance(sentinel_relative, str)
            else run_root / "__missing_continuation_sentinel__"
        )
        continuation_integrity = bool(
            continuation.get("prefix_validated") is True
            and continuation.get("import_complete") is True
            and sentinel_path.is_file()
            and not sentinel_path.is_symlink()
            and sentinel_path.stat().st_nlink == 1
            and sentinel_path.resolve().is_relative_to(run_root.resolve())
            and sha256_file(sentinel_path)
            == continuation.get("import_complete_sentinel_sha256")
        )
    allowed_exception_by_ordinal = {
        row["execution_ordinal"]: row
        for row in continuation.get("allowed_terminal_exceptions", [])
    }

    def critical_report_passed(report: dict[str, Any]) -> bool:
        ordinary_pass = all(
            report["validity_conditions"].get(condition) is True
            for condition in CRITICAL_ARM_INFRASTRUCTURE_CONDITIONS
        )
        if ordinary_pass:
            return True
        exception = allowed_exception_by_ordinal.get(report["execution_ordinal"])
        if exception is None:
            return False
        attempt_path = run_root / "attempts" / f"{report['attempt_key']}.json"
        return bool(
            report["execution_ordinal"]
            < int(continuation.get("carried_attempts", 0))
            and attempt_path.is_file()
            and sha256_file(attempt_path) == exception.get("attempt_sha256")
        )

    critical_integrity = all(critical_report_passed(report) for report in ordered)
    all_slots_protocol_valid = all(
        all(
            report["validity_conditions"].get(condition) is True
            for condition in CRITICAL_ARM_INFRASTRUCTURE_CONDITIONS
        )
        for report in ordered
    )
    notes_enabled = ACTIVE_ARM in {"A1", "A2"}
    snapshots = sorted((run_root / "notes_history").glob("ep_*.md"))
    expected_snapshot_names = [f"ep_{position:03d}.md" for position in range(1, 101)]
    notes_complete = (
        [path.name for path in snapshots] == expected_snapshot_names
        if notes_enabled else not snapshots
    )
    notes_chain_valid = True
    expected_previous = sha256_bytes(b"")
    migrations_by_ordinal: dict[int, list[dict[str, Any]]] = {}
    for migration in continuation.get("state_migrations", []):
        migrations_by_ordinal.setdefault(
            int(migration["after_execution_ordinal"]), []
        ).append(migration)
    if notes_enabled:
        for row in ordered:
            note = row["persistent_notes"]
            if (
                note.get("previous_state_sha256") != expected_previous
                or note.get("history_snapshot_sha256")
                != note.get("next_state_sha256")
            ):
                notes_chain_valid = False
            expected_previous = note.get("next_state_sha256")
            for migration in migrations_by_ordinal.get(row["execution_ordinal"], []):
                artifact = run_root / migration["artifact_relative_path"]
                if (
                    migration.get("from_state_sha256") != expected_previous
                    or migration.get("to_state_sha256")
                    != migration.get("artifact_sha256")
                    or not artifact.is_file()
                    or sha256_file(artifact) != migration.get("to_state_sha256")
                    or migration.get("no_model_rerun") is not True
                    or migration.get("model_invoked_for_migration") is not False
                ):
                    notes_chain_valid = False
                expected_previous = migration.get("to_state_sha256")
    if set(migrations_by_ordinal) - {row["execution_ordinal"] for row in ordered}:
        notes_chain_valid = False

    version_lines = []
    versions_path = run_root / "notes_versions.jsonl"
    if notes_enabled and versions_path.is_file():
        version_lines = versions_path.read_bytes().splitlines()
    notes_versions_complete = (
        len(version_lines) == EXPECTED_EPISODES if notes_enabled else not versions_path.exists()
    )
    slot_commits = sorted((run_root / "slot_commits").glob("slot_*.json"))
    expected_slot_commit_names = [
        f"slot_{position:03d}.json" for position in range(1, EXPECTED_EPISODES + 1)
    ]
    slot_commits_complete = [path.name for path in slot_commits] == expected_slot_commit_names
    if slot_commits_complete:
        for report, commit_path in zip(ordered, slot_commits):
            commit = json.loads(commit_path.read_bytes())
            attempt_path = run_root / "attempts" / f"{report['attempt_key']}.json"
            if (
                commit.get("execution_ordinal") != report["execution_ordinal"]
                or commit.get("attempt_key") != report["attempt_key"]
                or not attempt_path.is_file()
                or commit.get("attempt_sha256") != sha256_file(attempt_path)
            ):
                slot_commits_complete = False
                break
    valid_rows = [row for row in ordered if row["validity"] == "VALID"]
    correct_valid = sum(row["correctness"] is True for row in valid_rows)
    thread_ids = [
        row["trace_audit"]["thread_ids"][0]
        for row in ordered
        if len(row["trace_audit"].get("thread_ids", [])) == 1
    ]
    fresh_threads_complete = bool(
        len(thread_ids) == EXPECTED_EPISODES
        and len(set(thread_ids)) == EXPECTED_EPISODES
    )

    quartiles: list[dict[str, Any]] = []
    for quartile in range(4):
        rows = ordered[quartile * 25 : (quartile + 1) * 25]
        valid_quartile = [row for row in rows if row["validity"] == "VALID"]
        def mean_metric(name: str) -> float | None:
            values = [row["counters"].get(name) for row in valid_quartile]
            numeric = [float(value) for value in values if value is not None]
            return sum(numeric) / len(numeric) if numeric else None
        effective_moves_sum = sum(
            int(row["counters"]["effective_moves"]) for row in valid_quartile
        )
        effective_revisits_sum = sum(
            int(row["counters"]["effective_revisits"]) for row in valid_quartile
        )
        accepted_moves_sum = sum(
            int(row["counters"]["accepted_moves"]) for row in valid_quartile
        )
        wall_bumps_sum = sum(
            int(row["counters"]["wall_bumps"]) for row in valid_quartile
        )
        correct_quartile = sum(
            row["correctness"] is True for row in valid_quartile
        )
        quartiles.append(
            {
                "quartile": quartile + 1,
                "positions": [quartile * 25 + 1, (quartile + 1) * 25],
                "attempted": len(rows),
                "valid": len(valid_quartile),
                "accuracy_valid": (
                    correct_quartile / len(valid_quartile)
                    if valid_quartile else None
                ),
                "accuracy_scheduled_sensitivity": correct_quartile / len(rows),
                "mean_effective_moves": mean_metric("effective_moves"),
                "pooled_revisit_rate": (
                    effective_revisits_sum / effective_moves_sum
                    if effective_moves_sum else None
                ),
                "pooled_wall_bump_rate": (
                    wall_bumps_sum / accepted_moves_sum
                    if accepted_moves_sum else None
                ),
                "mean_episode_revisit_rate_sensitivity": mean_metric(
                    "revisit_rate"
                ),
                "mean_episode_wall_bump_rate_sensitivity": mean_metric(
                    "wall_bump_rate"
                ),
                "rate_numerators_and_denominators": {
                    "effective_revisits": effective_revisits_sum,
                    "effective_moves": effective_moves_sum,
                    "wall_bumps": wall_bumps_sum,
                    "accepted_moves": accepted_moves_sum,
                },
            }
        )

    schedule_exhausted = bool(
        preflight.get("passed")
        and len(ordered) == EXPECTED_EPISODES
        and continuation_integrity
        and ordinals_complete
        and schedule_matches
        and critical_integrity
        and notes_complete
        and notes_chain_valid
        and notes_versions_complete
        and slot_commits_complete
        and fresh_threads_complete
    )
    summary = {
        "arm": ACTIVE_ARM,
        "suite_id": ACTIVE_SUITE_ID,
        "schedule_exhausted": schedule_exhausted,
        "schedule_complete": schedule_exhausted,
        "all_slots_protocol_valid": all_slots_protocol_valid,
        "continuation_integrity": continuation_integrity,
        "attempted": len(ordered),
        "imported_attempts": int(continuation.get("carried_attempts", 0)),
        "fresh_attempts": len(ordered)
        - int(continuation.get("carried_attempts", 0)),
        "retries": 0,
        "valid": len(valid_rows),
        "invalid": len(ordered) - len(valid_rows),
        "correct_valid": correct_valid,
        "accuracy_valid": correct_valid / len(valid_rows) if valid_rows else None,
        "accuracy_scheduled_sensitivity": (
            correct_valid / len(ordered) if ordered else None
        ),
        "quartiles": quartiles,
        "primary_metrics": ["effective_moves", "revisit_rate", "wall_bump_rate"],
        "secondary_metric": "accuracy_valid",
        "fresh_session_evidence": {
            "thread_ids_observed": len(thread_ids),
            "unique_thread_ids": len(set(thread_ids)),
            "exactly_100_unique": fresh_threads_complete,
        },
        "notes": {
            "enabled": notes_enabled,
            "history_snapshots": len(snapshots),
            "expected_snapshot_names_complete": notes_complete,
            "hash_chain_valid": notes_chain_valid,
            "version_records": len(version_lines),
            "version_records_complete": notes_versions_complete,
            "final_state_sha256": expected_previous if notes_enabled else None,
            "content_gate_applied": False,
            "observational_audits": len(
                list((run_root / "leakage_audits").glob("ep_*.json"))
            ),
        },
        "continuation": continuation,
        "slot_commits": {
            "observed": len(slot_commits),
            "complete_and_hash_matched": slot_commits_complete,
        },
        "native_image_delivery": {
            "accepted_moves": sum(row["counters"]["accepted_moves"] for row in ordered),
            "authenticated_receipts": sum(
                len(row["native_image_delivery"]["controller_delivery_receipts"])
                for row in ordered
            ),
            "hash_matched_results": sum(
                len([
                    delivery
                    for delivery in row["native_image_delivery"]["trace_native_move_deliveries"]
                    if delivery.get("passed") is True
                ])
                for row in ordered
            ),
            "context_compaction_events_observed": sum(
                len(row["native_image_delivery"]["context_compaction_events"])
                for row in ordered
            ),
        },
        "preflight": preflight,
        "completed_at": utc_now(),
    }
    write_json(run_root / "summary.json", summary)
    with (run_root / "results.jsonl").open("wb") as handle:
        for report in ordered:
            handle.write(canonical_json_bytes(report) + b"\n")
    columns = [
        "arm", "execution_position", "episode_id", "validity", "answer",
        "ground_truth_label", "correctness", "glimpses_exposed", "effective_moves",
        "effective_revisits", "revisit_rate", "wall_bumps", "wall_bump_rate",
        "submit_calls", "total_codex_tool_calls", "notes_changed",
        "notes_state_sha256", "workspace", "trace",
    ]
    with (run_root / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in ordered:
            notes = row["persistent_notes"]
            writer.writerow(
                {
                    "arm": ACTIVE_ARM,
                    "execution_position": row["execution_ordinal"] + 1,
                    "episode_id": row["configuration"]["episode_id"],
                    "validity": row["validity"],
                    "answer": row["answer"],
                    "ground_truth_label": row["ground_truth_label"],
                    "correctness": row["correctness"],
                    "glimpses_exposed": row["counters"]["glimpses_exposed"],
                    "effective_moves": row["counters"]["effective_moves"],
                    "effective_revisits": row["counters"]["effective_revisits"],
                    "revisit_rate": row["counters"]["revisit_rate"],
                    "wall_bumps": row["counters"]["wall_bumps"],
                    "wall_bump_rate": row["counters"]["wall_bump_rate"],
                    "submit_calls": row["counters"]["submit_calls"],
                    "total_codex_tool_calls": row["counters"]["total_codex_tool_calls"],
                    "notes_changed": notes.get("changed"),
                    "notes_state_sha256": notes.get("next_state_sha256"),
                    "workspace": row["paths"]["workspace"],
                    "trace": row["paths"]["jsonl_trace"],
                }
            )
    report_md = f"""# MNIST-PRO matched arm {ACTIVE_ARM}

Decision: **{'EXHAUSTED' if schedule_exhausted else 'INCOMPLETE'}**. This arm ran the frozen shuffled Level-1 schedule with the native-image MCP interface.

- Scored slots: {len(ordered)}
- Valid slots: {len(valid_rows)}
- Accuracy on valid slots: {summary['accuracy_valid']}
- Persistent notes enabled: {notes_enabled}
- Immutable notes snapshots: {len(snapshots)}
- Notes content gate applied: no

All successful moves were reconciled against the hidden evaluator by PNG-byte and RGB-pixel hashes. The image-viewing tool could reopen exposed images and workspace-created composites. A2 alone received a bare `CORRECT` or `INCORRECT` receipt; A0 and A1 received `submission accepted`. Notes diagnostics were observational and never altered carried content. Context-compaction events, if any, are reported descriptively rather than used to select trajectories.
"""
    (run_root / "audit_report.md").write_text(report_md, encoding="utf-8")
    return summary


def main() -> int:
    global ACTIVE_ARM, ACTIVE_SUITE_ID, RUNTIME_PARENT, AUTH_VAULT_PARENT
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("A0", "A1", "A2"), required=True)
    parser.add_argument("--suite-id", required=True)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--auth-source", type=Path, required=True)
    parser.add_argument("--barrier-file", type=Path)
    parser.add_argument("--continue-from-arm-root", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{7,63}", args.suite_id):
        parser.error("--suite-id must be 8-64 lowercase letters, digits, or hyphens")
    ACTIVE_ARM = args.arm
    ACTIVE_SUITE_ID = args.suite_id
    arm_private_root = THREE_ARM_PRIVATE_ROOT / args.suite_id / args.arm.lower()
    RUNTIME_PARENT = arm_private_root / "runtimes"
    AUTH_VAULT_PARENT = arm_private_root / "auth_vaults"
    if args.workers != 1:
        parser.error("--workers must be exactly 1 for matched sequential exposure")
    if args.timeout_seconds < 60:
        parser.error("--timeout-seconds must be at least 60")
    if not args.preflight_only and args.barrier_file is None:
        parser.error("scored runs require --barrier-file from the suite launcher")
    if args.preflight_only and args.continue_from_arm_root is not None:
        parser.error("--continue-from-arm-root is only valid for a scored run")

    auth_source = args.auth_source.resolve()
    expected_auth_parent = (
        THREE_ARM_PRIVATE_ROOT / "suite_auth" / ACTIVE_SUITE_ID
    ).resolve()
    if auth_source.parent != expected_auth_parent or auth_source.name != "auth.json":
        parser.error("--auth-source is not the suite-private authentication snapshot")

    previous_signal_handlers: dict[int, Any] = {}

    def request_abort(signum: int, frame: Any) -> None:
        _ = frame
        raise ControllerAbort(f"controller received signal {signum}")

    for controller_signal in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        previous_signal_handlers[controller_signal] = signal.getsignal(controller_signal)
        signal.signal(controller_signal, request_abort)

    run_root = args.run_root.resolve()
    if run_root.exists():
        raise RuntimeError(f"run root already exists: {run_root}")
    run_root.mkdir(parents=True, exist_ok=False)
    for directory in (
        run_root / "attempts",
        run_root / "traces",
        run_root / "workspaces",
        run_root / "notes_history",
        run_root / "notes_candidates",
        run_root / "leakage_audits",
        run_root / "state_migrations",
        run_root / "slot_commits",
        run_root / "preflight/episodes",
        run_root / "controller_assets",
    ):
        directory.mkdir(parents=True, exist_ok=False)
    controller_snapshot = run_root / "controller_assets/controller_snapshot.py"
    ag_mcp_snapshot = run_root / "controller_assets/ag_mcp_server.py"
    shutil.copy2(Path(__file__).resolve(), controller_snapshot)
    shutil.copy2(AG_MCP_SOURCE, ag_mcp_snapshot)
    shutil.copy2(
        A1_NOTES_DOCUMENT,
        run_root / "controller_assets/notes_instruction_A1_no_feedback.md",
    )
    shutil.copy2(
        A2_NOTES_DOCUMENT,
        run_root / "controller_assets/notes_instruction_A2_with_feedback.md",
    )
    write_json(
        run_root / "controller_assets/arm_configuration.json",
        {
            "arm": ACTIVE_ARM,
            "suite_id": ACTIVE_SUITE_ID,
            "notes_enabled": ACTIVE_ARM in {"A1", "A2"},
            "feedback_enabled": ACTIVE_ARM == "A2",
            "notes_size_cap_enforced": False,
            "prompt_sha256": sha256_bytes(prompt_for(1).encode("utf-8")),
            "notes_instruction_sha256": (
                sha256_bytes(notes_instruction_for(ACTIVE_ARM).encode("utf-8"))
                if notes_instruction_for(ACTIVE_ARM) is not None else None
            ),
        },
    )
    journal = Journal(run_root / "progress.jsonl")
    journal.append(
        {
            "event": "controller_started",
            "arm": ACTIVE_ARM,
            "suite_id": ACTIVE_SUITE_ID,
            "run_root": str(run_root),
        }
    )
    print(f"RUN_ROOT {run_root}", flush=True)
    print(f"PREFLIGHT validating pinned Level 1 {ACTIVE_ARM} protocol", flush=True)

    generic_root: Path | None = None
    generic_bin: Path | None = None
    minimal_auth: dict[str, Any] = {}
    vault: AuthStateVault | None = None
    arm_lock: ArmProcessLock | None = None
    controller_succeeded = False
    try:
        arm_lock = ArmProcessLock(ACTIVE_SUITE_ID, ACTIVE_ARM)
        active_process_check = active_three_arm_process_preflight()
        pinned = validate_pinned_inputs()
        level1, level2 = project_schedules()
        specs = build_specs(level1, level2)
        source_ids = [spec.episode_id for spec in specs]
        shuffle_seed = int(FROZEN_SHUFFLE_SEED_HEX, 16)
        shuffled_specs = list(specs)
        random.Random(shuffle_seed).shuffle(shuffled_specs)
        shuffled_ids = [spec.episode_id for spec in shuffled_specs]
        shuffled_hash = sha256_bytes(canonical_json_bytes(shuffled_ids))
        if (
            len(shuffled_specs) != EXPECTED_EPISODES
            or set(shuffled_ids) != set(range(100))
            or shuffled_ids == source_ids
            or shuffled_hash != FROZEN_PERMUTATION_SHA256
        ):
            raise RuntimeError("frozen Level 1 shuffle changed")
        continuation_reports: list[dict[str, Any]] = []
        continuation_approved_notes = b""
        continuation = no_continuation_record()
        if args.continue_from_arm_root is not None:
            (
                continuation_reports,
                continuation_approved_notes,
                continuation,
            ) = prepare_arm_continuation(
                args.continue_from_arm_root,
                run_root,
                shuffled_specs,
            )
            journal.append(
                {
                    "event": "continuation_prefix_committed",
                    "source_suite_id": continuation["source_suite_id"],
                    "carried_attempts": continuation["carried_attempts"],
                    "next_execution_ordinal": continuation[
                        "next_execution_ordinal"
                    ],
                    "state_migrations": continuation["state_migrations"],
                }
            )
        schedule_payload = {
            "source": {
                "manifest": str(LEVEL1_SUMMARY),
                "manifest_sha256": PINNED_HASHES["level1_summary"],
                "projection_sha256": PINNED_HASHES["level1_projection"],
                "rows": level1,
            },
            "shuffle": {
                "algorithm": "Python random.Random MT19937 shuffle (Fisher-Yates)",
                "independent_controller_seed_hex": f"{shuffle_seed:064x}",
                "execution_episode_ids": shuffled_ids,
                "execution_permutation_sha256": shuffled_hash,
                "visible_name_rng": "independent OS cryptographic randomness",
                "frozen_before_image_protocol_revision": True,
            },
            "configuration": {
                "arm": ACTIVE_ARM,
                "suite_id": ACTIVE_SUITE_ID,
                "levels": [1],
                "episode_ids": list(range(100)),
                "total_scored_episodes": EXPECTED_EPISODES,
                "seed": SEED,
                "num_eval_sets": NUM_EVAL_SETS,
                "image_size": IMAGE_SIZE,
                "box_size": BOX_SIZE,
                "step_size": STEP_SIZE,
                "observation_budget": BUDGETS[1],
                "image_delivery_protocol": IMAGE_DELIVERY_PROTOCOL,
            },
            "continuation": {
                key: continuation[key]
                for key in (
                    "applicable",
                    "source_suite_id",
                    "carried_attempts",
                    "next_execution_ordinal",
                    "state_migrations",
                )
            },
        }
        write_json(run_root / "schedule_manifest.json", schedule_payload)

        module = load_environment_module()
        image_cache = preload_mnist(specs)
        parity = validate_all_initial_observations(module, specs, image_cache)
        parity_public = {
            "episodes_checked": parity["episodes_checked"],
            "pixel_mismatches": parity["pixel_mismatches"],
            "raw_rgb_hashes_sha256": sha256_bytes(
                canonical_json_bytes(parity["raw_rgb_hashes"])
            ),
        }
        generic_root = Path("/private/tmp") / secrets.token_hex(16)
        generic_root.mkdir(mode=0o700)
        generic_bin = generic_root / "bin"
        generic_bin.mkdir()
        ag_mcp_runtime = generic_bin / "ag-mcp"
        shutil.copy2(ag_mcp_snapshot, ag_mcp_runtime)
        if sha256_file(ag_mcp_runtime) != PINNED_HASHES["ag_mcp_server"]:
            raise RuntimeError("runtime MCP bridge differs from pinned snapshot")
        ag_mcp_runtime.chmod(0o555)
        generic_bin.chmod(0o555)
        generic_root.chmod(0o555)

        minimal_auth, initial_auth_metadata = read_minimal_auth(auth_source)
        access_expiry = jwt_expiry(minimal_auth["tokens"]["access_token"])
        required_access_expiry = (
            time.time()
            + EXPECTED_EPISODES * args.timeout_seconds
            + 12 * 60 * 60
        )
        if access_expiry is None or access_expiry <= required_access_expiry:
            raise RuntimeError(
                "access token lifetime does not cover the concurrent arm horizon"
            )
        vault = AuthStateVault(ACTIVE_ARM)
        vault.checkpoint(minimal_auth)
        auth_check = auth_preflight(minimal_auth, vault, required_access_expiry)

        probe_dir = Path("/private/tmp") / secrets.token_hex(16)
        probe_obs_dir = probe_dir / secrets.token_hex(16)
        probe_obs_dir.mkdir(parents=True, mode=0o700)
        probe_observation = probe_obs_dir / f"{secrets.token_hex(16)}.png"
        _, _, prompt_image = make_episode_environment(
            module, shuffled_specs[0], image_cache
        )
        prompt_image.save(probe_observation, format="PNG")
        try:
            prompt_audit = prompt_preflight(1, probe_observation)
        finally:
            shutil.rmtree(probe_dir)
        if prompt_audit["prompt_sha256"] != sha256_bytes(
            prompt_for(1, ACTIVE_ARM).encode("utf-8")
        ):
            raise RuntimeError(f"native-image {ACTIVE_ARM} prompt composition changed")

        deterministic_toolchain_check = deterministic_local_toolchain_preflight(
            run_root, generic_root
        )
        network_check = network_isolation_probe(generic_root, minimal_auth, run_root)
        integrated_check = a1_integrated_scored_shape_preflight(
            run_root,
            generic_root,
            minimal_auth,
            vault,
            required_access_expiry,
            module,
            shuffled_specs[0],
            image_cache,
        )
        viewer_fusion_check = a1_viewer_fusion_preflight(
            run_root,
            generic_root,
            minimal_auth,
            vault,
            required_access_expiry,
        )
        feedback_interface_check = arm_feedback_interface_preflight()
        feedback_behavioral_check = a2_post_submit_notes_preflight(
            run_root,
            generic_root,
            minimal_auth,
            vault,
            required_access_expiry,
        )
        remote_catalog_check = remote_model_catalog_preflight(
            generic_root,
            minimal_auth,
            run_root,
            vault,
            required_access_expiry,
        )
        notes_persistence_check = notes_persistence_preflight(run_root)
        post_preflight_auth_metadata = auth_metadata(minimal_auth)
        refreshed_expiry = post_preflight_auth_metadata["access_token_exp_unix"]
        if refreshed_expiry is None or refreshed_expiry <= required_access_expiry:
            raise RuntimeError(
                f"access token lifetime no longer covers {ACTIVE_ARM} horizon"
            )
        preflight = {
            "passed": True,
            "pinned_inputs": pinned,
            "configuration": schedule_payload["configuration"],
            "schedule": {
                "permutation_complete": True,
                "permutation_nontrivial": True,
                "permutation_sha256": schedule_payload["shuffle"][
                    "execution_permutation_sha256"
                ],
                "controller_only_during_execution": True,
                "identical_across_arms": True,
            },
            "process_exclusivity": active_process_check,
            "prompt": {
                **prompt_audit,
                "original_cleanroom_prompt_sha256": sha256_bytes(
                    baseline_prompt_for(1).encode("utf-8")
                ),
                "native_image_common_prompt_sha256": sha256_bytes(
                    native_image_prompt_for(1).encode("utf-8")
                ),
                "verbatim_notes_instruction_sha256": (
                    sha256_bytes(notes_instruction_for(ACTIVE_ARM).encode("utf-8"))
                    if notes_instruction_for(ACTIVE_ARM) is not None else None
                ),
                "effective_notes_instruction_sha256": (
                    sha256_bytes(notes_instruction_for(ACTIVE_ARM).encode("utf-8"))
                    if notes_instruction_for(ACTIVE_ARM) is not None else None
                ),
                "source_notes_instruction_block_sha256": (
                    pinned.get(f"{ACTIVE_ARM.lower()}_source_notes_block_sha256")
                    if ACTIVE_ARM in {"A1", "A2"} else None
                ),
                "notes_protocol_amendment": {
                    "source_document_preserved_unchanged": True,
                    "size_cap_enforced": False,
                    "effective_instruction_differs_from_source_only_by_size_policy": (
                        ACTIVE_ARM in {"A1", "A2"}
                    ),
                },
                "composition": (
                    "native-image common prompt only"
                    if ACTIVE_ARM == "A0"
                    else "native-image common prompt + exactly two LF + effective no-cap notes instruction"
                ),
                "protocol_revision": IMAGE_DELIVERY_PROTOCOL,
            },
            "initial_observation_parity": parity_public,
            "deterministic_local_toolchain": deterministic_toolchain_check,
            "network_isolation": network_check,
            "integrated_scored_shape": integrated_check,
            "viewer_and_private_scratch": viewer_fusion_check,
            "feedback_interface": feedback_interface_check,
            "feedback_post_submit_behavioral": feedback_behavioral_check,
            "remote_model_catalog": remote_catalog_check,
            "notes_persistence": notes_persistence_check,
            "auth": {
                "initial": initial_auth_metadata,
                "login_status": auth_check,
                "post_preflight": post_preflight_auth_metadata,
                "credential_rotation_permitted": False,
                "durable_controller_only_vault": True,
            },
            "generic_ag_mcp_sha256": sha256_file(ag_mcp_runtime),
            "controller_snapshot_sha256": sha256_file(controller_snapshot),
            "ag_mcp_snapshot_sha256": sha256_file(ag_mcp_snapshot),
            "continuation": continuation,
        }
        write_json(run_root / "preflight.json", preflight)
        journal.append({"event": "global_preflight_passed"})
        print(
            f"PREFLIGHT passed for {ACTIVE_ARM}: 100/100 Level 1 observations "
            "matched; the model-free native-image/viewer/ImageMagick gate and "
            "feedback/persistence checks passed",
            flush=True,
        )
        if args.preflight_only:
            journal.append({"event": "controller_preflight_only_completed"})
            print("PREFLIGHT_ONLY_FINISHED", flush=True)
            controller_succeeded = True
            return 0

        assert args.barrier_file is not None
        barrier_path = args.barrier_file.resolve()
        barrier_deadline = time.monotonic() + 60 * 60
        write_json(
            run_root / "BARRIER_READY.json",
            {
                "ready": True,
                "arm": ACTIVE_ARM,
                "suite_id": ACTIVE_SUITE_ID,
                "controller_pid": os.getpid(),
                "controller_sha256": sha256_file(controller_snapshot),
                "ag_mcp_sha256": sha256_file(ag_mcp_snapshot),
                "published_at": utc_now(),
            },
        )
        journal.append(
            {
                "event": "preflight_barrier_waiting",
                "arm": ACTIVE_ARM,
                "barrier_path_sha256": sha256_bytes(str(barrier_path).encode("utf-8")),
            }
        )
        while not barrier_path.is_file():
            if time.monotonic() >= barrier_deadline:
                raise RuntimeError("suite preflight barrier timed out")
            time.sleep(0.1)
        barrier = json.loads(barrier_path.read_text(encoding="utf-8"))
        if (
            barrier.get("released") is not True
            or barrier.get("suite_id") != ACTIVE_SUITE_ID
            or barrier.get("schedule_sha256") != FROZEN_PERMUTATION_SHA256
            or barrier.get("controller_sha256") != sha256_file(controller_snapshot)
            or barrier.get("ag_mcp_sha256") != sha256_file(ag_mcp_snapshot)
        ):
            raise RuntimeError("suite preflight barrier payload mismatch")
        journal.append({"event": "preflight_barrier_released", "arm": ACTIVE_ARM})

        reports = list(continuation_reports)
        approved_notes = continuation_approved_notes
        for ordinal in range(continuation["next_execution_ordinal"], EXPECTED_EPISODES):
            spec = shuffled_specs[ordinal]
            report, approved_notes = run_one_arm_episode(
                spec,
                ordinal,
                approved_notes,
                module,
                image_cache,
                run_root,
                generic_root,
                minimal_auth,
                vault,
                required_access_expiry,
                prompt_audit,
                journal,
                args.timeout_seconds,
            )
            reports.append(report)
            if not report["process"]["process_group_terminated"]:
                raise RuntimeError("process group survived; refusing schedule advance")
            failed_infrastructure = [
                condition
                for condition in CRITICAL_ARM_INFRASTRUCTURE_CONDITIONS
                if report["validity_conditions"].get(condition) is not True
            ]
            if failed_infrastructure:
                journal.append(
                    {
                        "event": "critical_infrastructure_failure",
                        "execution_ordinal": ordinal,
                        "level": 1,
                        "episode_id": spec.episode_id,
                        "failed_conditions": failed_infrastructure,
                        "action": "schedule_halted_before_next_slot",
                    }
                )
                raise RuntimeError(
                    f"critical {ACTIVE_ARM} infrastructure condition failed; refusing "
                    f"schedule advance: {failed_infrastructure}"
                )

        summary = write_arm_aggregate_outputs(
            run_root, reports, preflight, shuffled_specs
        )
        if not summary["schedule_exhausted"]:
            raise RuntimeError(
                f"Arm {ACTIVE_ARM} schedule or authorized state chain is incomplete"
            )
        journal.append(
            {
                "event": "controller_completed",
                "arm": ACTIVE_ARM,
                "attempted": summary["attempted"],
                "valid": summary["valid"],
                "invalid": summary["invalid"],
            }
        )
        print(
            f"FINISHED arm={ACTIVE_ARM} attempted={summary['attempted']} "
            f"valid={summary['valid']} "
            f"invalid={summary['invalid']}",
            flush=True,
        )
        controller_succeeded = True
        return 0
    except Exception:
        failure = traceback.format_exc()
        (run_root / "CONTROLLER_FAILURE.txt").write_text(failure, encoding="utf-8")
        journal.append({"event": "controller_failed", "traceback": failure})
        print(failure, file=sys.stderr, flush=True)
        return 1
    finally:
        for controller_signal in previous_signal_handlers:
            signal.signal(controller_signal, signal.SIG_IGN)
        minimal_auth.clear()
        if vault is not None:
            if controller_succeeded:
                vault.remove()
            elif vault.path.exists():
                print(
                    f"AUTH_STATE_VAULT_PRESERVED {vault.path}",
                    file=sys.stderr,
                    flush=True,
                )
        if generic_root is not None and generic_root.exists():
            generic_root.chmod(0o755)
            if generic_bin is not None and generic_bin.exists():
                generic_bin.chmod(0o755)
            shutil.rmtree(generic_root)
        if arm_lock is not None:
            arm_lock.close()
        for controller_signal, previous_handler in previous_signal_handlers.items():
            signal.signal(controller_signal, previous_handler)


if __name__ == "__main__":
    raise SystemExit(main())
