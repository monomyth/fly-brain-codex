"""Open a persistent camera-review window; save preview frames, never a training episode."""
import base64
import json
import os
from pathlib import Path
import subprocess
import time
from fly_brain.simulator import ReBotClient, DEFAULT_BINARY, screen_locked

project=Path(__file__).resolve().parents[1]
out=project/'reports/camera-fov-review'
out.mkdir(parents=True,exist_ok=True)
control=out/'control'
control.mkdir(mode=0o700,exist_ok=True)
if screen_locked():
    raise SystemExit('Unlock the Mac for camera preview rendering.')
client=ReBotClient(control)
try:
    state=client.call('rebot_get_state')
except (OSError,ConnectionError):
    log=(out/'preview-native.log').open('a')
    process=subprocess.Popen([str(Path(DEFAULT_BINARY).expanduser())],
        env={**os.environ,'REBOT_CONTROL_DIRECTORY':str(control),'REBOT_MCP_NO_LAUNCH':'1'},
        stdout=log,stderr=log,start_new_session=True)
    log.close()
    end=time.monotonic()+45
    while True:
        if process.poll() is not None:raise RuntimeError('Preview app exited; inspect preview-native.log')
        try:
            state=client.call('rebot_get_state')
            if state.get('scene_ready'):break
        except (OSError,ConnectionError):pass
        if time.monotonic()>end:raise TimeoutError('Preview window did not initialize')
        time.sleep(.2)
if not state.get('experiment',{}).get('episode_id') or state['experiment'].get('phase')=='disabled':
    client.call('rebot_configure_task',task={'cube_xy_mm':[350,0],'cube_size_mm':20,
                'initial_joints_deg':[0]*6,'initial_gripper_mm':0,'input_mode':'state'})
    client.wait_ready()
client.call('rebot_set_view',camera='Front',tool_axes=False,trace=False)
obs=client.observe(images=True)
assert [im['name'] for im in obs['images']]==['Front','Gripper']
assert obs['camera_rig_revision']=='front336l-gripper305-rgb-v4'
(out/'folded-observation.json').write_text(json.dumps(obs,indent=2))
for frame in obs['images']:
    (out/(frame['name'].lower()+'-folded.jpg')).write_bytes(base64.b64decode(frame['jpeg_base64']))
(out/'preview-process.json').write_text(json.dumps({'pid':state['process_id'],'control_directory':str(control),'purpose':'camera approval preview; no data collection or training'},indent=2))
print(json.dumps({'pid':state['process_id'],'output':str(out),'cameras':[f['name'] for f in obs['images']]},indent=2))
