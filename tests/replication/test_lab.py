"""Phase 2 exit criteria as tests (SPEC §8, §11): the lab launches, and the Playground trains
n = 5, m = 2 in ≤ 10 s on CPU. Slow: they start a server and train real models.
"""

import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from superposition import lab

APP = Path(__file__).resolve().parents[2] / "app"
TARGET_SECONDS = 10.0  # SPEC §8: "n = 5, m = 2 trains in ≤ 10 s on CPU"


@pytest.mark.slow
def test_default_playground_trains_within_target_on_cpu() -> None:
    config = lab.PlaygroundConfig()
    assert (config.n_features, config.n_hidden) == (5, 2)
    result = lab.train_playground(config, device="cpu")
    print(
        f"\nPlayground default (n=5, m=2, {config.steps} steps x {config.restarts} restarts): "
        f"trained in {result.train_seconds:.2f} s on CPU"
    )
    assert result.train_seconds <= TARGET_SECONDS
    # The defaults are the p. 2 pentagon setting (S = 0.9): all five features represented.
    assert lab.summary(result)["represented features"] == 5


@pytest.mark.slow
def test_playground_page_first_run_within_target(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole page, uncached: training plus every figure and table, on CPU."""
    monkeypatch.setattr(sys, "path", [str(APP), *sys.path])
    start = time.perf_counter()
    at = AppTest.from_file(str(APP / "pages" / "1_Playground.py"), default_timeout=120).run()
    elapsed = time.perf_counter() - start
    print(f"\nPlayground page first run (default config, uncached): {elapsed:.2f} s")
    assert not at.exception
    assert elapsed <= TARGET_SECONDS


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.mark.slow
def test_superposition_app_launches() -> None:
    """`superposition app` starts Streamlit and serves the Home page.

    The CLI runs Streamlit as a child process. Earlier versions of this test terminated only
    the CLI and piped the server's output to a reader that then went away, so each run left an
    orphaned server that could not even log its own shutdown. The command now runs in its own
    process group with its output discarded, and the whole group is stopped.
    """
    port = _free_port()
    command = [sys.executable, "-m", "superposition.cli", "app", "--headless", "--port", str(port)]
    process = subprocess.Popen(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True
    )
    try:
        deadline = time.monotonic() + 60
        health = None
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/_stcore/health", timeout=2
                ) as r:
                    health = r.read().decode()
                    break
            except OSError:
                time.sleep(0.5)
        assert health == "ok", "the Streamlit server did not become healthy within 60 s"
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as page:
            assert page.status == 200
    finally:
        _stop_process_group(process)


def _stop_process_group(leader: subprocess.Popen, timeout: float = 10.0) -> None:
    """SIGTERM every process in the leader's group, wait until all have exited, then SIGKILL."""
    pgid = leader.pid
    os.killpg(pgid, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        leader.poll()  # reap the leader, or the group never looks empty
        try:
            os.killpg(pgid, 0)  # raises ProcessLookupError once the group is empty
        except ProcessLookupError:
            return
        except PermissionError:  # macOS: only exited, not yet reaped members remain
            pass
        time.sleep(0.2)
    os.killpg(pgid, signal.SIGKILL)
    leader.wait(timeout=5)
