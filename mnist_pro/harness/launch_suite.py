#!/opt/anaconda3/bin/python
"""Launch and supervise A0/A1/A2 x Level 1/Level 2 concurrently.

The launcher is controller-only.  It freezes the controller and the two saved
schedule manifests before starting six isolated controller process groups.  It
never calls Gemini itself and never places labels, schedules, or credentials in
an agent workspace.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import stat
import subprocess
import time
import traceback
from typing import Any


BASE = Path(
    "/Users/poriasoujanya/Documents/Codex/2026-08-25/"
    "files-pasted-by-the-user-act"
)
SOURCE_DIR = BASE / "work/mnist_pro_antigravity_six_arm"
DEFAULT_CONTROLLER = SOURCE_DIR / "controller.py"
OUTPUTS = BASE / "outputs"
PRIVATE = SOURCE_DIR / "controller_private"

LEVEL1_SUMMARY = Path(
    "/Users/poriasoujanya/Documents/activeglimpse-main/main_table_logs/"
    "gemini-3.7-flash_img224_box64_step32_maxsteps36_seed42_"
    "DefaultVisionAgent_hist-1_evalsets10_20260824_105544/"
    "results_summary.json"
)
LEVEL2_SUMMARY = Path(
    "/Users/poriasoujanya/Documents/activeglimpse-main/main_table_logs/"
    "multidigit_2_gemini-3.7-flash_img224_box64_step32_maxsteps78_"
    "seed42_MultiDigitDefaultVisionAgent_hist-1_evalsets10_"
    "20260824_111931/results_summary.json"
)
SOURCE_MANIFESTS = {1: LEVEL1_SUMMARY, 2: LEVEL2_SUMMARY}
SOURCE_MANIFEST_SHA256 = {
    1: "48362448e87a819e162bc42ac9104493ad8b3405bf65c1cec52418e88b21c208",
    2: "d1778e114011974e2416bd45ef2dba15047c8a065596e173b6e3b6daf620c465",
}
A1_NOTES_DOCUMENT = Path(
    "/Users/poriasoujanya/Downloads/notes_instruction_A1_no_feedback.md"
)
A2_NOTES_DOCUMENT = Path(
    "/Users/poriasoujanya/Downloads/notes_instruction_A2_with_feedback.md"
)
SOURCE_NOTES_SHA256 = {
    "A1": "f0f0e273d96d9dbc8c42c106115a15b4750a5a019e81e19112d39d14430a811e",
    "A2": "5cdccfe342ca2ebdd1749248d1a8d2056cb4e43d6e71515bb03467c94d769e94",
}

ARMS = ("A0", "A1", "A2")
LEVELS = (1, 2)
JOBS = tuple((level, arm) for level in LEVELS for arm in ARMS)
EXPECTED_EPISODES_PER_JOB = 100
TOTAL_SCORED_EPISODES = len(JOBS) * EXPECTED_EPISODES_PER_JOB
PYTHON = Path("/opt/anaconda3/bin/python")


class SuiteAbort(RuntimeError):
    """Raised by a signal handler so cleanup runs before exit."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def job_name(level: int, arm: str) -> str:
    return f"L{level}_{arm}"


def job_root(suite_root: Path, level: int, arm: str) -> Path:
    return suite_root / f"L{level}" / arm


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def validate_pinned_inputs(controller: Path) -> dict[str, str]:
    if not controller.is_file():
        raise RuntimeError(f"controller does not exist: {controller}")
    hashes: dict[str, str] = {"controller": sha256_file(controller)}
    for level, source in SOURCE_MANIFESTS.items():
        if not source.is_file():
            raise RuntimeError(f"Level {level} schedule source is missing: {source}")
        observed = sha256_file(source)
        if observed != SOURCE_MANIFEST_SHA256[level]:
            raise RuntimeError(f"Level {level} saved schedule manifest changed")
        hashes[f"level_{level}_source_manifest"] = observed
    for arm, source in {
        "A1": A1_NOTES_DOCUMENT,
        "A2": A2_NOTES_DOCUMENT,
    }.items():
        if not source.is_file() or sha256_file(source) != SOURCE_NOTES_SHA256[arm]:
            raise RuntimeError(f"{arm} source notes document changed")
        hashes[f"{arm.lower()}_source_notes_document"] = SOURCE_NOTES_SHA256[arm]
    return hashes


def validate_credential_reference(reference: str, *, resolve: bool) -> dict[str, str]:
    """Validate a reference without ever reading or returning the key value."""
    if reference == "keychain":
        if resolve:
            result = subprocess.run(
                (
                    "/usr/bin/security",
                    "find-generic-password",
                    "-s",
                    "gemini",
                    "-a",
                    "antigravity",
                ),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    "macOS Keychain has no service gemini/account antigravity item"
                )
        return {
            "kind": "macos_keychain",
            "service": "gemini",
            "account": "antigravity",
        }
    if reference.startswith("env:"):
        name = reference[4:]
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise RuntimeError("unsafe credential environment-variable name")
        if resolve and not os.environ.get(name):
            raise RuntimeError(f"credential environment variable is unset: {name}")
        return {"kind": "environment", "name": name}

    raw_path = reference[5:] if reference.startswith("file:") else reference
    path = Path(raw_path)
    if not path.is_absolute():
        raise RuntimeError(
            "credential source must be keychain, env:NAME, or an absolute file path"
        )
    if resolve:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.getuid()
                or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or not 1 <= metadata.st_size <= 65_536
            ):
                raise RuntimeError("credential file failed structural checks")
        finally:
            os.close(descriptor)
    return {"kind": "owner_only_file"}


def process_group_alive(pgid: int) -> bool:
    if pgid <= 1 or pgid == os.getpgrp():
        return False
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def terminate_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 20
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
    if process.poll() is None or process_group_alive(process.pid):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"process group {process.pid} survived cleanup") from error


class HeldLock:
    def __init__(self, path: Path, identity: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(path.parent, 0o700)
        self.path = path
        self.handle = path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self.handle.close()
            raise RuntimeError(f"active evaluation lock exists: {path.name}") from error
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(json.dumps(identity, sort_keys=True) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def update(self, identity: dict[str, Any]) -> None:
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(json.dumps(identity, sort_keys=True) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def close(self) -> None:
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()


def validate_offline_contracts(
    suite_root: Path,
    suite_id: str,
    controller_sha256: str,
) -> dict[str, Any]:
    contracts: dict[str, dict[str, Any]] = {}
    schedule_hashes: dict[int, set[str]] = {1: set(), 2: set()}
    for level, arm in JOBS:
        name = job_name(level, arm)
        path = job_root(suite_root, level, arm) / "preflight/offline_contract.json"
        record = read_json(path)
        expected = {
            "passed": True,
            "arm": arm,
            "level": level,
            "suite_id": suite_id,
            "controller_sha256": controller_sha256,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256[level],
            "notes_enabled": arm in {"A1", "A2"},
            "feedback_enabled": arm == "A2",
            "notes_size_cap_enforced": False,
            "expected_episode_count": EXPECTED_EPISODES_PER_JOB,
            "scored_api_calls": 0,
        }
        mismatches = {
            key: {"expected": value, "observed": record.get(key)}
            for key, value in expected.items()
            if record.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"{name} offline contract mismatch: {mismatches}")
        schedule_hash = record.get("schedule_sha256")
        if not isinstance(schedule_hash, str) or not re.fullmatch(
            r"[0-9a-f]{64}", schedule_hash
        ):
            raise RuntimeError(f"{name} lacks a valid schedule hash")
        schedule_hashes[level].add(schedule_hash)
        contracts[name] = record
    for level, hashes in schedule_hashes.items():
        if len(hashes) != 1:
            raise RuntimeError(f"Level {level} arm schedules are not identical")
    return {
        "passed": True,
        "jobs": list(contracts),
        "schedule_sha256": {
            f"L{level}": next(iter(hashes))
            for level, hashes in schedule_hashes.items()
        },
        "all_six_paid_calls_before_release": 0,
        "validated_at": utc_now(),
    }


def validate_ready_markers(
    suite_root: Path,
    suite_id: str,
    controller_sha256: str,
    processes: dict[str, subprocess.Popen[bytes]],
) -> None:
    for level, arm in JOBS:
        name = job_name(level, arm)
        record = read_json(job_root(suite_root, level, arm) / "BARRIER_READY.json")
        expected = {
            "ready": True,
            "arm": arm,
            "level": level,
            "suite_id": suite_id,
            "controller_pid": processes[name].pid,
            "controller_sha256": controller_sha256,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256[level],
            "scored_api_calls": 0,
        }
        if any(record.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"{name} common-barrier identity mismatch")


def notes_snapshots(root: Path) -> list[Path]:
    history = root / "notes_history"
    return sorted(history.glob("ep_*.md")) if history.is_dir() else []


def validate_job_summary(root: Path, level: int, arm: str) -> dict[str, Any]:
    complete = read_json(root / "ARM_COMPLETE.json")
    summary = read_json(root / "summary.json")
    if complete.get("complete") is not True:
        raise RuntimeError(f"{job_name(level, arm)} lacks a successful terminal marker")
    expected = {
        "arm": arm,
        "level": level,
        "attempted": EXPECTED_EPISODES_PER_JOB,
        "schedule_complete": True,
    }
    if any(summary.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"{job_name(level, arm)} summary is incomplete or mismatched")
    snapshots = notes_snapshots(root)
    expected_names = [f"ep_{index:03d}.md" for index in range(1, 101)]
    if arm in {"A1", "A2"}:
        if [path.name for path in snapshots] != expected_names:
            raise RuntimeError(f"{job_name(level, arm)} notes history is not exactly versioned")
    elif snapshots:
        raise RuntimeError(f"{job_name(level, arm)} unexpectedly produced persistent notes")
    return summary


def write_suite_summary(suite_root: Path) -> dict[str, Any]:
    jobs: dict[str, dict[str, Any]] = {}
    for level, arm in JOBS:
        jobs[job_name(level, arm)] = validate_job_summary(
            job_root(suite_root, level, arm), level, arm
        )
    summary = {
        "suite_complete": True,
        "schedule_exhausted": True,
        "total_scored_episodes": TOTAL_SCORED_EPISODES,
        "retried_episodes": 0,
        "concurrent_controllers": len(JOBS),
        "jobs": jobs,
        "notes_version_counts": {
            name: (100 if name.endswith(("A1", "A2")) else 0)
            for name in jobs
        },
        "total_notes_snapshots": 400,
        "completed_at": utc_now(),
    }
    write_json(suite_root / "suite_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-root", type=Path)
    parser.add_argument("--controller", type=Path, default=DEFAULT_CONTROLLER)
    parser.add_argument("--credential-source", required=True)
    parser.add_argument("--episode-timeout-seconds", type=int, default=30 * 60)
    parser.add_argument("--barrier-timeout-seconds", type=int, default=30 * 60)
    parser.add_argument("--monitor-interval-seconds", type=float, default=2.0)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="run six offline contracts concurrently and make no model/API calls",
    )
    args = parser.parse_args(argv)
    if args.episode_timeout_seconds < 60:
        parser.error("--episode-timeout-seconds must be at least 60")
    if args.barrier_timeout_seconds < 10:
        parser.error("--barrier-timeout-seconds must be at least 10")
    if not 0.05 <= args.monitor_interval_seconds <= 60:
        parser.error("--monitor-interval-seconds must be between 0.05 and 60")

    # Validate every external input before creating the immutable suite root.
    # In particular, a scored launch cannot create a half-initialized run and
    # only then discover that its credential reference or saved schedule is bad.
    controller = args.controller.resolve()
    pinned = validate_pinned_inputs(controller)
    credential_descriptor = validate_credential_reference(
        args.credential_source, resolve=not args.preflight_only
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nonce = secrets.token_hex(5)
    suite_id = f"ag-l12-{nonce}"
    suite_root = (
        args.suite_root
        or OUTPUTS / f"mnist_pro_antigravity_six_arm_{timestamp}_{nonce}"
    ).resolve()
    if suite_root.exists():
        raise RuntimeError(f"suite root already exists: {suite_root}")
    suite_root.mkdir(parents=True, exist_ok=False)
    (suite_root / "controller_assets/source_manifests").mkdir(parents=True)
    (suite_root / "controller_assets/source_notes").mkdir(parents=True)
    (suite_root / "controller_logs").mkdir()
    for level, arm in JOBS:
        job_root(suite_root, level, arm).mkdir(parents=True)

    frozen_controller = suite_root / "controller_assets/controller.py"
    shutil.copy2(controller, frozen_controller)
    for level, source in SOURCE_MANIFESTS.items():
        shutil.copy2(
            source,
            suite_root
            / f"controller_assets/source_manifests/level_{level}_results_summary.json",
        )
    shutil.copy2(
        A1_NOTES_DOCUMENT,
        suite_root / "controller_assets/source_notes/A1_source_document.md",
    )
    shutil.copy2(
        A2_NOTES_DOCUMENT,
        suite_root / "controller_assets/source_notes/A2_source_document.md",
    )
    write_json(
        suite_root / "controller_assets/source_notes/no_cap_amendment.json",
        {
            "applies_to": ["A1", "A2"],
            "source_documents_retained_for_provenance": True,
            "effective_replacement": "There is no size cap.",
            "size_cap_enforced": False,
        },
    )
    frozen_controller_sha256 = sha256_file(frozen_controller)
    if frozen_controller_sha256 != pinned["controller"]:
        raise RuntimeError("controller snapshot changed while being frozen")

    locks: list[HeldLock] = []
    processes: dict[str, subprocess.Popen[bytes]] = {}
    log_handles: list[Any] = []
    succeeded = False
    prior_handlers: dict[int, Any] = {}
    barrier_path = suite_root / "PREFLIGHT_RELEASE.json"

    def request_abort(signum: int, frame: Any) -> None:
        _ = frame
        raise SuiteAbort(f"suite supervisor received signal {signum}")

    try:
        suite_lock = HeldLock(
            PRIVATE / "locks/active_antigravity_six_arm_suite.lock",
            {"pid": os.getpid(), "suite_id": suite_id, "suite_root": str(suite_root)},
        )
        locks.append(suite_lock)
        for level, arm in JOBS:
            name = job_name(level, arm)
            locks.append(
                HeldLock(
                    PRIVATE / f"locks/active_{name.lower()}.lock",
                    {"pid": os.getpid(), "suite_id": suite_id, "job": name},
                )
            )
        for suite_signal in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            prior_handlers[suite_signal] = signal.getsignal(suite_signal)
            signal.signal(suite_signal, request_abort)

        manifest: dict[str, Any] = {
            "suite_id": suite_id,
            "suite_root": str(suite_root),
            "started_at": utc_now(),
            "harness": "Antigravity agent harness (Gemini 3.7 Flash)",
            "controller_role": "orchestration_only",
            "levels": list(LEVELS),
            "arms": list(ARMS),
            "jobs": [job_name(level, arm) for level, arm in JOBS],
            "concurrent_controllers": len(JOBS),
            "episodes_per_job": EXPECTED_EPISODES_PER_JOB,
            "total_scored_episodes": TOTAL_SCORED_EPISODES,
            "workers_per_job": 1,
            "retry_policy": "none",
            "credential_source": credential_descriptor,
            "controller_sha256": frozen_controller_sha256,
            "source_manifest_sha256": {
                f"L{level}": digest
                for level, digest in SOURCE_MANIFEST_SHA256.items()
            },
            "source_notes_document_sha256": SOURCE_NOTES_SHA256,
            "notes_size_cap_enforced": False,
            "notes_snapshots_expected_per_notes_job": 100,
            "total_notes_snapshots_expected": 400,
            "preflight_only": args.preflight_only,
        }
        write_json(suite_root / "launch_manifest.json", manifest)
        print(f"SUITE_ROOT {suite_root}", flush=True)
        print(f"SUITE_ID {suite_id}", flush=True)

        for level, arm in JOBS:
            name = job_name(level, arm)
            stdout = (suite_root / f"controller_logs/{name}.stdout.log").open("wb")
            stderr = (suite_root / f"controller_logs/{name}.stderr.log").open("wb")
            log_handles.extend((stdout, stderr))
            command = [
                str(PYTHON),
                str(frozen_controller),
                "--arm",
                arm,
                "--level",
                str(level),
                "--suite-id",
                suite_id,
                "--run-root",
                str(job_root(suite_root, level, arm)),
                "--credential-source",
                args.credential_source,
                "--barrier-file",
                str(barrier_path),
                "--episode-timeout-seconds",
                str(args.episode_timeout_seconds),
            ]
            if args.preflight_only:
                command.append("--preflight-only")
            process = subprocess.Popen(
                command,
                cwd=BASE,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            processes[name] = process
        manifest["controller_processes"] = {
            name: {
                "pid": process.pid,
                "pgid": process.pid,
                "run_root": str(
                    job_root(suite_root, int(name[1]), name.split("_", 1)[1])
                ),
            }
            for name, process in processes.items()
        }
        write_json(suite_root / "launch_manifest.json", manifest)
        print(
            "CONTROLLERS_LAUNCHED "
            + " ".join(f"{name}={process.pid}" for name, process in processes.items()),
            flush=True,
        )

        deadline = time.monotonic() + args.barrier_timeout_seconds
        ready: set[str] = set()
        while len(ready) < len(JOBS):
            for level, arm in JOBS:
                name = job_name(level, arm)
                process = processes[name]
                offline = job_root(suite_root, level, arm) / "preflight/offline_contract.json"
                barrier_ready = job_root(suite_root, level, arm) / "BARRIER_READY.json"
                if args.preflight_only:
                    if offline.is_file():
                        ready.add(name)
                    if process.poll() is not None and not offline.is_file():
                        raise RuntimeError(
                            f"{name} exited before offline preflight: {process.returncode}"
                        )
                else:
                    if offline.is_file() and barrier_ready.is_file():
                        ready.add(name)
                    if process.poll() is not None:
                        raise RuntimeError(
                            f"{name} exited before common barrier: {process.returncode}"
                        )
            if time.monotonic() >= deadline:
                raise RuntimeError("six-arm offline preflight barrier timed out")
            if len(ready) < len(JOBS):
                time.sleep(args.monitor_interval_seconds)

        cross_arm = validate_offline_contracts(
            suite_root, suite_id, frozen_controller_sha256
        )
        write_json(suite_root / "cross_arm_preflight.json", cross_arm)
        if args.preflight_only:
            for name, process in processes.items():
                try:
                    returncode = process.wait(timeout=30)
                except subprocess.TimeoutExpired as error:
                    raise RuntimeError(f"{name} preflight process did not exit") from error
                if returncode != 0:
                    raise RuntimeError(f"{name} preflight exited {returncode}")
            write_json(
                suite_root / "PREFLIGHT_SUITE_COMPLETE.json",
                {
                    "passed": True,
                    "jobs": len(JOBS),
                    "scored_api_calls": 0,
                    "paid_calls": 0,
                    "completed_at": utc_now(),
                },
            )
            print("PREFLIGHT_SUITE_FINISHED scored_api_calls=0", flush=True)
            succeeded = True
            return 0

        validate_ready_markers(
            suite_root, suite_id, frozen_controller_sha256, processes
        )
        write_json(
            barrier_path,
            {
                "released": True,
                "suite_id": suite_id,
                "controller_sha256": frozen_controller_sha256,
                "jobs": [job_name(level, arm) for level, arm in JOBS],
                "schedule_sha256": cross_arm["schedule_sha256"],
                "released_at": utc_now(),
            },
        )
        print("PREFLIGHT_BARRIER_RELEASED jobs=6", flush=True)

        while True:
            unfinished = [
                name for name, process in processes.items() if process.poll() is None
            ]
            failures: list[dict[str, Any]] = []
            progress: dict[str, Any] = {}
            for level, arm in JOBS:
                name = job_name(level, arm)
                root = job_root(suite_root, level, arm)
                process = processes[name]
                attempted = 0
                progress_path = root / "progress.jsonl"
                if progress_path.is_file():
                    with progress_path.open("rb") as handle:
                        attempted = sum(1 for line in handle if line.strip())
                progress[name] = {
                    "attempt_records": attempted,
                    "controller_alive": process.poll() is None,
                    "arm_complete": (root / "ARM_COMPLETE.json").is_file(),
                }
                failure_marker = root / "CONTROLLER_FAILURE.txt"
                if failure_marker.is_file():
                    failures.append({"job": name, "marker": str(failure_marker)})
                elif process.poll() is not None and process.returncode != 0:
                    failures.append({"job": name, "returncode": process.returncode})
                elif process.poll() is not None and not (root / "ARM_COMPLETE.json").is_file():
                    failures.append(
                        {"job": name, "error": "controller vanished without terminal marker"}
                    )
            write_json(
                suite_root / "suite_progress.json",
                {
                    "updated_at": utc_now(),
                    "jobs": progress,
                    "unfinished_controllers": unfinished,
                    "failures": failures,
                },
            )
            if failures:
                raise RuntimeError(f"controller failure; no retry permitted: {failures}")
            if not unfinished:
                break
            time.sleep(args.monitor_interval_seconds)

        suite_summary = write_suite_summary(suite_root)
        write_json(
            suite_root / "SUITE_COMPLETE.json",
            {
                "suite_complete": True,
                "total_scored_episodes": TOTAL_SCORED_EPISODES,
                "retried_episodes": 0,
                "jobs": list(suite_summary["jobs"]),
                "completed_at": utc_now(),
            },
        )
        print(f"SUITE_FINISHED total_scored_episodes={TOTAL_SCORED_EPISODES}", flush=True)
        succeeded = True
        return 0
    except Exception:
        failure = traceback.format_exc()
        (suite_root / "SUITE_FAILURE.txt").write_text(failure, encoding="utf-8")
        print(failure, flush=True)
        return 1
    finally:
        for suite_signal in prior_handlers:
            signal.signal(suite_signal, signal.SIG_IGN)
        cleanup_errors: list[str] = []
        if not succeeded:
            for name, process in processes.items():
                try:
                    terminate_group(process)
                except Exception as error:
                    cleanup_errors.append(f"{name}: {error}")
        for handle in log_handles:
            try:
                handle.close()
            except Exception as error:
                cleanup_errors.append(f"log handle: {error}")
        for lock in reversed(locks):
            try:
                lock.close()
            except Exception as error:
                cleanup_errors.append(f"lock {lock.path.name}: {error}")
        for suite_signal, prior in prior_handlers.items():
            signal.signal(suite_signal, prior)
        if cleanup_errors:
            write_json(
                suite_root / "CLEANUP_FAILURE.json",
                {"errors": cleanup_errors, "recorded_at": utc_now()},
            )
            if succeeded:
                raise RuntimeError(f"suite cleanup was incomplete: {cleanup_errors}")


if __name__ == "__main__":
    raise SystemExit(main())
