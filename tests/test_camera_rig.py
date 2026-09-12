from copy import deepcopy
import json
import numpy as np
import pytest
from fly_brain.cameras import CURRENT_CAMERAS, CURRENT_RIG, LEGACY_CAMERAS, LEGACY_RIG, observation_contract
from fly_brain.dataset import EpisodeWriter
from fly_brain.audit import audit_dataset
from fly_brain.visual_dopamine.core import RetinalEncoder
from test_visual_dopamine import circuit, images


def current_images():
    frames=images()
    for frame,name in zip(frames,CURRENT_CAMERAS):
        frame.update(name=name,rig_revision=CURRENT_RIG,width=32,height=32)
    return frames


def test_retinal_pair_is_explicit_and_old_models_reject_changed_front():
    encoder=RetinalEncoder(circuit())
    old=encoder.sample(images())
    with pytest.raises(ValueError,match='Front and Top'):
        encoder.sample(current_images())
    renamed=current_images();renamed[1]['name']='Top'
    with pytest.raises(ValueError,match='Camera viewpoint changed'):
        encoder.sample(renamed)
    encoder.configure_cameras(CURRENT_CAMERAS,CURRENT_RIG)
    assert np.array_equal(encoder.sample(current_images()),old)
    assert observation_contract({'images':current_images()})==(CURRENT_CAMERAS,CURRENT_RIG)
    with pytest.raises(ValueError):
        encoder.sample(images())


def test_new_episode_keeps_both_images_and_audit_detects_mixed_rigs(tmp_path):
    task={'cube_xy_mm':[350,0],'cube_size_mm':20}
    writer=EpisodeWriter(tmp_path,task,'teacher_assisted',2,{'images_each_decision':True})
    obs={'images':current_images(),'episode_id':'a','frame_id':1,'simulation_time':1.,'image_state_skew_seconds':0}
    writer.append(obs,{'joints_deg':[0]*6,'gripper_mm':0})
    changed={**obs,'images':images(),'frame_id':2,'simulation_time':2.}
    with pytest.raises(ValueError,match='rig changed'):
        writer.append(changed,{})
    directory=writer.finish({'success':True})
    assert (directory/'images/00000-Gripper.jpg').is_file()
    meta=json.loads((directory/'manifest.json').read_text())
    assert meta['camera_names']==list(CURRENT_CAMERAS) and meta['camera_rig_revision']==CURRENT_RIG
    assert audit_dataset(tmp_path)['valid']
    meta['camera_rig_revision']=LEGACY_RIG
    (directory/'manifest.json').write_text(json.dumps(meta))
    assert not audit_dataset(tmp_path)['valid']


def test_observation_metadata_cannot_disagree_with_images():
    obs={'images':current_images(),'camera_rig_revision':LEGACY_RIG}
    with pytest.raises(ValueError,match='revisions disagree'):
        observation_contract(obs)
    bad=current_images();bad[1]['name']='Front'
    with pytest.raises(ValueError,match='distinct'):
        observation_contract({'images':bad})


def test_gripper_image_uses_its_full_frame_without_a_world_crop():
    encoder=RetinalEncoder(circuit());encoder.configure_cameras(CURRENT_CAMERAS,CURRENT_RIG)
    encoder.configure({'kind':'fixed_workspace','lower_m':[.25,-.1,0],'upper_m':[.45,.1,.17],'cameras':['Front']})
    rgb=np.zeros((20,30,3),np.uint8)
    assert encoder.image_region(rgb,{'name':'Gripper'}) is rgb


@pytest.mark.parametrize('intervene',[False,True])
def test_absolute_teacher_goals_are_not_confused_with_bounded_delta_labels(tmp_path,intervene):
    from fly_brain.adapters import ActionAdapter
    writer=EpisodeWriter(tmp_path,{},'conventional',5,{'execution_mode':'absolute_targets'})
    obs={'episode_id':'absolute','frame_id':1,'simulation_time':1.,'joints_deg':[0]*6,'gripper_mm':0}
    target={'joints_deg':[0,-30,-30,0,0,0],'gripper_mm':90}
    action={'joints_deg':[0,-4,-4,0,0,0],'gripper_mm':0} if intervene else target
    intervention={'reason':'recorded exploration'} if intervene else None
    writer.append(obs,action,ActionAdapter(5).label(obs,target['joints_deg'],90),intervention=intervention,expert_target=target)
    directory=writer.finish({'success':True})
    assert audit_dataset(tmp_path)['valid']
    file=directory/'steps.jsonl';row=json.loads(file.read_text());row['expert_label'][0]=1
    file.write_text(json.dumps(row)+'\n')
    assert not audit_dataset(tmp_path)['valid']
