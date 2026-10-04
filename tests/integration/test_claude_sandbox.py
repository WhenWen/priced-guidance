"""Actual Linux namespace checks; no model, credentials or target content."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest

from tech_tree_arena.runtime.claude_sandbox import command, environment

pytestmark=pytest.mark.skipif(sys.platform!='linux',reason='Claude uses Linux namespaces')


@pytest.mark.parametrize('network',[False,True])
def test_private_data_and_host_services_are_inaccessible(tmp_path,monkeypatch,network):
    workspace=tmp_path/'worker';workspace.mkdir()
    hidden=tmp_path/'private-gold';hidden.write_text('synthetic forbidden canary')
    sibling=tmp_path/'other-worker';sibling.mkdir();(sibling/'session').write_text('synthetic other session')
    (workspace/'escape').symlink_to(hidden)
    (workspace/'own').write_text('allowed')
    monkeypatch.setenv('ANTHROPIC_API_KEY','synthetic-forbidden-key')
    with socket.socket() as listener:
        listener.bind(('127.0.0.1',0));listener.listen()
        script='''
from pathlib import Path
import os,socket,json
blocked=[]
for path in PATHS:
 try:Path(path).read_text();blocked.append(False)
 except OSError:blocked.append(True)
try:
 socket.create_connection(('127.0.0.1',PORT),timeout=1);local=False
except OSError:local=True
print(json.dumps({'blocked':blocked,'own':Path('/arena/own').read_text(),'cwd':os.getcwd(),'no_key':'ANTHROPIC_API_KEY' not in os.environ,'local_blocked':local,'shell_absent':not Path('/bin/sh').exists()}))
'''.replace('PATHS',repr([str(hidden),str(sibling/'session'),'/arena/escape',str(Path.home()/'.claude/.credentials.json')])).replace('PORT',str(listener.getsockname()[1]))
        env=environment(workspace)
        proc=subprocess.run(command(workspace,Path(sys.executable).resolve(),['-I','-S','-c',script],network=network,
            read_roots=(Path(sys.base_prefix).resolve(),)),env=env,capture_output=True,text=True,timeout=20)
    assert proc.returncode==0,proc.stderr
    result=json.loads(proc.stdout)
    assert all(result['blocked']) and result['no_key'] and result['local_blocked'] and result['shell_absent']
    assert result['own']=='allowed' and result['cwd']=='/arena'
