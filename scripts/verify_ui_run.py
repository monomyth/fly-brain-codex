"""Exercise the real Run button and capture active simulator/brain UI evidence."""
import json
from pathlib import Path
import subprocess
import time
from fly_brain.assets import write_json
from fly_brain.simulator import ReBotClient

root=Path(__file__).resolve().parents[1];report=root/'reports/retain-grasp-20260911'
record=json.loads((root/'data/ui/current.json').read_text());client=ReBotClient(record['control_directory'])
state=client.call('rebot_get_state');previous=Path(state['fly_brain']['result_directory'])
write_json(report/'ui-second-result.json',{'result_directory':str(previous),'evaluation':json.loads((previous/'evaluation.json').read_text()),'window':state['window']})
client.call('rebot_reset_episode',seed=75002);client.wait_ready();client.call('rebot_set_view')
pid=state['process_id']
subprocess.run(['osascript','-e',f'tell application "System Events" to set frontmost of first application process whose unix id is {pid} to true'],check=True)
subprocess.run([str(root/'.cache/ui_button'),str(pid),'Run fly brain'],check=True)
captured=False;deadline=time.monotonic()+200
while time.monotonic()<deadline:
    state=client.call('rebot_get_state');brain=state['brain_activity'];runner=state['fly_brain']
    if runner['running'] and brain.get('episode_id') and brain.get('maximum_activity') is not None and not captured:
        write_json(report/'ui-live-state.json',{'brain_activity':brain,'fly_brain':runner,'window':state['window'],'experiment':state['experiment']})
        subprocess.run([str(Path.home()/'.codex/skills/xcode-debug/scripts/capture_app_window.sh'),'--app-name','ReBot Motion Lab Codex','--window-title','B601-DM Simulator - Codex','--output',str(report/'ui-live.png')],check=True)
        captured=True;print('Captured active brain UI',flush=True)
    if not runner['running']:
        result=Path(runner['result_directory']);evaluation=json.loads((result/'evaluation.json').read_text())
        write_json(report/'ui-third-result.json',{'result_directory':str(result),'evaluation':evaluation,'window':state['window'],'active_ui_captured':captured})
        print(json.dumps({'ui_status':runner['status'],'attempts':evaluation['attempts'],'successes':evaluation['successes'],'captured':captured}),flush=True)
        break
    time.sleep(.25)
else:raise TimeoutError('UI verification did not complete')
