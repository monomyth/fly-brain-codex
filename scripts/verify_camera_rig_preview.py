"""Check live sensor extrinsics and previews; does not collect demonstrations or train."""
import base64
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation
from fly_brain.simulator import ReBotClient
from fly_brain.cameras import observation_contract, CURRENT_CAMERAS, CURRENT_RIG

out=Path(__file__).resolve().parents[1]/'reports/camera-fov-review'
info=json.loads((out/'preview-process.json').read_text())
client=ReBotClient(info['control_directory'])
report={'rig_revision':CURRENT_RIG,'checks':[],'training_started':False,'demonstrations_collected':False}

def verify(label):
    obs=client.observe(images=True)
    assert observation_contract(obs)==(CURRENT_CAMERAS,CURRENT_RIG)
    assert obs['image_state_skew_seconds']==0
    by_name={c['name']:c for c in obs['images']}
    for frame in obs['images']:
        summary=next(c for c in obs['camera_calibration'] if c['name']==frame['name'])
        assert np.array_equal(frame['world_from_camera'],summary['world_from_camera'])
        (out/f"{frame['name'].lower()}-{label}.jpg").write_bytes(base64.b64decode(frame['jpeg_base64']))
    for name,model in [('Front','336L'),('Gripper','305')]:
        camera=by_name[name];k=np.asarray(camera['intrinsics'])
        assert (camera['width'],camera['height'])==(320,200)
        assert model in camera['camera_model']
        hfov=np.degrees(2*np.arctan(camera['width']/(2*k[0,0])))
        vfov=np.degrees(2*np.arctan(camera['height']/(2*k[1,1])))
        assert abs(hfov-camera['horizontal_fov_deg'])<1e-8 and abs(vfov-68)<1e-8
        assert abs(hfov-94)<.5
        assert camera['nominal_rgb_fov_deg']=={'horizontal':94,'vertical':68,'tolerance':3}
    report['rgb_optics_verified']={name:{key:by_name[name][key] for key in ['camera_model','width','height','horizontal_fov_deg','vertical_fov_deg']} for name in ['Front','Gripper']}
    wrist=by_name['Gripper'];tool=obs['tool_pose']
    world_tool=np.eye(4);world_tool[:3,:3]=Rotation.from_quat(tool['quaternion_xyzw']).as_matrix();world_tool[:3,3]=np.array(tool['position_mm'])/1000
    actual=np.array(wrist['world_from_camera']);expected=world_tool@np.array(wrist['mount_from_camera'])
    assert np.max(np.abs(actual-expected))<1e-5
    report['checks'].append({'pose':label,'joints_deg':obs['joints_deg'],'gripper_mm':obs['gripper_mm'],
            'max_wrist_transform_error':float(np.abs(actual-expected).max()),'camera_telemetry_synchronized':True})
    (out/f'{label}-observation.json').write_text(json.dumps(obs,indent=2))
    return by_name

def wait_motion():
    end=time.monotonic()+15
    while time.monotonic()<end:
        state=client.call('rebot_get_state')
        if state['playback']=='stopped':return
        time.sleep(.1)
    raise TimeoutError('Preview joint motion did not finish')

try:
    first=verify('folded')
    client.call('rebot_set_view',camera='Gripper')
    client.call('rebot_move_joints',joints_deg=[20,0,0,0,0,15],gripper_mm=35)
    wait_motion();moved=verify('wrist-rotated')
    assert np.max(np.abs(np.array(first['Gripper']['world_from_camera'])-np.array(moved['Gripper']['world_from_camera'])))>.1
    assert np.array_equal(first['Front']['world_from_camera'],moved['Front']['world_from_camera'])
    for name in ['Orbit','Front','Top','Gripper']:
        client.call('rebot_set_view',camera=name)
    report['view_choices_verified']=['Orbit','Front','Top','Gripper']
finally:
    client.call('rebot_playback',action='stop')
    client.call('rebot_reset_episode');client.wait_ready()
    client.call('rebot_set_view',camera='Front')
report['success']=True
(out/'verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
