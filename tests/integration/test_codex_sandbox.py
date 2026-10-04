"""Exercise OS enforcement, without spending model tokens or reading target contents."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import pytest

from tech_tree_arena.runtime.codex_sandbox import command, environment
from tech_tree_arena.runtime.services import ServiceFactory
from tech_tree_arena.runtime.subprocess_actor import SubprocessActorFactory

pytestmark = pytest.mark.skipif(sys.platform not in {"darwin", "linux"}, reason="requires OS sandbox")
ROOT = Path(__file__).resolve().parents[2]


def test_kernel_blocks_data_symlinks_other_sessions_processes_and_network(tmp_path):
    # This uses the same outer policy as Codex, with a tiny adversarial Python
    # executable as the test subject. A successful open only produces a bool;
    # test failures never print host file contents or credentials.
    private = tmp_path / "private"
    private.mkdir()
    (private / "gold.json").write_text("canary")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "escape").symlink_to(private / "gold.json")
    (workspace / "allowed").write_text("public")
    script = r'''
import os, json, socket, subprocess
from pathlib import Path
def readable(p):
    try:
        with open(p, "rb") as f: f.read(1)
        return True
    except OSError: return False
paths = json.loads(os.environ["PROBE_PATHS"])
r = {name: readable(path) for name,path in paths.items()}
try: list(Path(os.environ["PROBE_DIRECTORY"]).iterdir()); r["list_private"] = True
except OSError: r["list_private"] = False
try: subprocess.run(["/bin/sh", "-c", "true"], check=True); r["shell"] = True
except OSError: r["shell"] = False
s=socket.socket(); s.settimeout(0.2)
try: s.connect(("127.0.0.1",int(os.environ["PROBE_PORT"]))); r["network_denied"] = False
except OSError: r["network_denied"] = True
r["api_key"] = "OPENAI_API_KEY" in os.environ
print(json.dumps(r))
'''
    env = environment(workspace)
    env["PROBE_PATHS"] = json.dumps({"allowed": str(workspace / "allowed"), "private": str(private / "gold.json"),
        "symlink": str(workspace / "escape"), "host_auth": str(Path.home() / ".codex/auth.json"),
        "repo_data": str(ROOT / "src/tech_tree_arena/data/target_packs/smoke/secret/gold/smoke-blue.json")})
    env["PROBE_DIRECTORY"] = str(private)
    executable = Path(sys.executable).resolve()
    # A live host listener makes connection refusal meaningful on Linux,
    # where isolation uses a separate network namespace instead of EACCES.
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        env["PROBE_PORT"] = str(listener.getsockname()[1])
        completed = subprocess.run(command(workspace, executable, ["-I", "-S", "-c", script], read_roots=(Path(sys.base_prefix),)),
                                   env=env, cwd=workspace, capture_output=True, text=True, timeout=20, check=True)
    assert json.loads(completed.stdout) == {"allowed": True, "private": False, "symlink": False,
        "host_auth": False, "repo_data": False, "list_private": False, "shell": False,
        "network_denied": True, "api_key": False}


def test_reference_generator_runs_inside_isolated_python_worker(tmp_path):
    factory = SubprocessActorFactory(ROOT / "submissions/reference_pair", "participant.generator:Generator",
        service_factory=ServiceFactory(seed=1, public_resources={"offline_smoke": True}),
        sandbox_generator=True, log_path=tmp_path / "generator.log")
    actor, _ = factory.create()
    try:
        assert actor.step(None).options
        workspace = Path(actor._sandbox_directory.name)
        assert not (workspace / "runtime/tech_tree_arena/data").exists()
        assert not (workspace / "runtime/tech_tree_arena/evaluation").exists()
    finally:
        actor.close()
    assert not workspace.exists()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux namespace and egress relay")
def test_linux_host_proc_other_session_and_egress_are_inaccessible(tmp_path):
    workspace = tmp_path / "worker-a"
    workspace.mkdir()
    other = tmp_path / "worker-b"
    other.mkdir()
    (other / "auth.json").write_text("other-session-canary")
    script = r'''
import json, socket, sys
from pathlib import Path
results = {}
for name, path in json.loads(sys.argv[1]).items():
    try: Path(path).read_bytes(); results[name] = True
    except OSError: results[name] = False
with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
    s.connect('/run/arena-egress.sock')
    s.sendall(b'CONNECT localhost:443 HTTP/1.1\r\n\r\n')
    results['proxy_rejects_host'] = b'403' in s.recv(1024)
with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
    s.connect('/run/arena-egress.sock')
    s.sendall(b'CONNECT example.com:443 HTTP/1.1\r\n\r\n')
    results['proxy_rejects_other_domains'] = b'403' in s.recv(1024)
print(json.dumps(results))
'''
    paths = {"other_session": str(other / "auth.json"), "host_proc": f"/proc/{os.getpid()}/environ"}
    result = subprocess.run(command(workspace, Path(sys.executable).resolve(), ["-I", "-S", "-c", script, json.dumps(paths)],
                                    read_roots=(Path(sys.base_prefix),)),
                            env=environment(workspace), cwd=workspace, capture_output=True, text=True, timeout=20, check=True)
    assert json.loads(result.stdout) == {"other_session": False, "host_proc": False,
        "proxy_rejects_host": True, "proxy_rejects_other_domains": True}


@pytest.mark.skipif(sys.platform != "linux", reason="Linux concurrent namespace isolation")
@pytest.mark.parametrize("worker_count", [10, 80])
def test_live_namespaces_cannot_read_each_other(tmp_path, worker_count):
    workspaces = [tmp_path / f"worker-{i}" for i in range(worker_count)]
    for i, workspace in enumerate(workspaces):
        workspace.mkdir()
        (workspace / "canary").write_text(str(i))
    script = r'''
import json, sys, time
from pathlib import Path
w = Path.cwd()
(w / 'ready').write_text('ready')
deadline = time.monotonic() + 55
while not (w / 'go').exists():
    if time.monotonic() > deadline: raise RuntimeError('barrier timed out')
    time.sleep(0.02)
try: Path(sys.argv[1]).read_text(); peer = True
except OSError: peer = False
print(json.dumps({'own': (w / 'canary').read_text(), 'peer': peer}))
'''
    processes = []
    try:
        for i, workspace in enumerate(workspaces):
            invocation = command(workspace, Path(sys.executable).resolve(),
                ["-I", "-S", "-c", script, str(workspaces[(i + 1) % 10] / "canary")],
                read_roots=(Path(sys.base_prefix),))
            processes.append(subprocess.Popen(invocation, cwd=workspace, env=environment(workspace),
                                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
        deadline = time.monotonic() + 40
        while not all((w / "ready").exists() for w in workspaces) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert all((w / "ready").exists() for w in workspaces), "All isolated processes must be alive together"
        for workspace in workspaces:
            (workspace / "go").write_text("go")
        for i, process in enumerate(processes):
            stdout, stderr = process.communicate(timeout=10)
            assert process.returncode == 0, stderr
            assert json.loads(stdout) == {"own": str(i), "peer": False}
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
