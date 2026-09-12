"""Open the existing native simulator with the qualified controller configured."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
from fly_brain.assets import write_json,sha256_file
from fly_brain.simulator import Simulator,configure_episode,DEFAULT_BINARY,ReBotClient
from run_mlx_ui import validate_deployment,ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--deployment',type=Path,default=ROOT/'configs/trained-runtime.json')
    p.add_argument('--check-only',action='store_true')
    a=p.parse_args();settings=json.loads(a.deployment.read_text());verified=validate_deployment(settings)
    if a.check_only:print(json.dumps(verified,indent=2));return
    parent=ROOT/'data/ui';parent.mkdir(parents=True,exist_ok=True)
    application=Path(DEFAULT_BINARY).expanduser().parents[2]
    current=parent/'current.json'
    if current.exists():
        previous=json.loads(current.read_text())
        if previous.get('checkpoint')==verified['checkpoint']:
            try:
                state=ReBotClient(previous['control_directory']).call('rebot_get_state')
            except (OSError,RuntimeError,ValueError):
                state=None
            if state and state.get('process_id')==previous.get('native_state',{}).get('process_id'):
                ReBotClient(previous['control_directory']).call('rebot_set_view')
                print(json.dumps({'status':'already_open','control_directory':previous['control_directory'],'checkpoint':verified['checkpoint']}),flush=True)
                return
    directory=Path(tempfile.mkdtemp(prefix='trained-',dir=parent));os.chmod(directory,0o700)
    if len(str(directory/'control.sock').encode())>=104:raise ValueError('UI control path is too long for a Unix socket')
    environment={'FLY_BRAIN_EXECUTABLE':str(ROOT/'scripts/fly-brain-mlx'),'FLY_BRAIN_CHECKPOINT':verified['checkpoint'],
                 'FLY_BRAIN_DATA_HOME':str(ROOT/'data'),'FLY_BRAIN_DEPLOYMENT':str(a.deployment.resolve()),
                 'REBOT_CONTROL_DIRECTORY':str(directory),'REBOT_MCP_NO_LAUNCH':'1'}
    command=['/usr/bin/open','-n','-a',str(application)]
    for key,value in environment.items():command+=['--env',key+'='+value]
    command+=['--stdout',str(directory/'native.log'),'--stderr',str(directory/'native-error.log'),'--args','--external-controller']
    subprocess.run(command,check=True)
    with Simulator(directory) as client:
        task=json.loads((ROOT/'configs/cube20-folded.json').read_text())
        task.update(input_mode='vision',timeout_seconds=180)
        actual=configure_episode(client,task)
        client.call('rebot_set_view')
        state=client.call('rebot_get_state')
        if state.get('fly_brain',{}).get('checkpoint')!=Path(verified['checkpoint']).name:
            raise RuntimeError('The native UI did not select the configured checkpoint')
        record={'control_directory':str(directory),'checkpoint':verified['checkpoint'],'task':actual,
                'deployment':str(a.deployment.resolve()),'launcher_source_sha256':sha256_file(__file__),
                'native_state':state}
        write_json(directory/'launch.json',record);write_json(parent/'current.json',record)
    print(json.dumps({'status':'ready','control_directory':str(directory),'checkpoint':verified['checkpoint'],
                      'next':'Use Run fly brain in the simulator window. Reset before another trial; cube placement can be edited in the UI.'}),flush=True)


if __name__=='__main__':main()
