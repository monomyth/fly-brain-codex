import json
import numpy as np
import torch
from fly_brain.visual_dopamine.trajectory_training import load_demonstrations
from fly_brain.kinematics import ToolKinematics, demonstration_target_poses


def write_recovery(directory, verified=True):
    episode=directory/'episode-recovery';episode.mkdir()
    meta={'complete':True,'success':True,'teacher_target_recorded':True,
          'supervision_contract':'recorded-recovery-v1','provenance':'teacher_assisted',
          'task':{'cube_xy_mm':[350,0],'hold_seconds':5,'lift_clearance_mm':100},
          'result':{},'execution_mode':'absolute_targets'}
    target={'joints_deg':[0,-100,-70,40,0,0],'gripper_mm':90}
    observation={'episode_id':'a','simulation_time':0,'joints_deg':[0]*6,'gripper_mm':0,'images':[]}
    rows=[{'stage':'recovery_open','observation':observation,'expert_target':target,'evaluation':{}},
          {'stage':'hold','observation':{**observation,'simulation_time':.5},'expert_target':{**target,'gripper_mm':0},
           'evaluation':{'success':verified,'held':True,'episode_id':'a','hold_seconds':5,'clearance_mm':125,'tilt_deg':0}}]
    (episode/'manifest.json').write_text(json.dumps(meta))
    (episode/'steps.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))
    return target


def test_recovery_targets_survive_generic_goal_and_grip_relabeling(tmp_path):
    expected=write_recovery(tmp_path)
    examples,_=load_demonstrations(tmp_path,{'train':[[350,0]]},goal_supervision='aligned',grasp_tolerance_mm=20,grasp_goal_height_mm=27)
    assert len(examples)==2
    assert all(e['preserve_expert_target'] for e in examples)
    assert np.array_equal(examples[0]['target'],expected['joints_deg']+[1.])
    assert examples[1]['target'][6]==0


def test_recovery_requires_native_hold_even_if_manifest_claims_success(tmp_path):
    write_recovery(tmp_path,verified=False)
    examples,episodes=load_demonstrations(tmp_path,{'train':[[350,0]]},goal_supervision='aligned')
    assert examples==[] and episodes==[]


def test_recovery_cartesian_targets_are_not_recentered_or_raised():
    fk=ToolKinematics();q=torch.zeros((3,6),dtype=torch.float64)
    examples=[{'stage':stage,'goal_role':role,'reference_lift_clearance_mm':115,'preserve_expert_target':True}
              for stage,role in [('recovery_open','approach'),('close','grasp'),('hold','transport')]]
    before,rotation=fk(q)
    after,r=demonstration_target_poses(fk,q,examples,125,0,27,[350,0],[350,0])
    assert torch.equal(before,after)
    assert torch.equal(rotation,r)


def test_recovery_alignment_and_opening_receive_grasp_correction_weight():
    from fly_brain.visual_dopamine.feedback_training import demonstration_weights
    examples=[{'stage':s,'observation':{'joints_deg':[0,-100,-70,40,0,0],'gripper_mm':30}}
              for s in ['recovery_open','recovery_align','hold']]
    weights=demonstration_weights(examples,[0,1,2],grasp_frame_weight=8)
    assert np.allclose(weights/weights[-1],[8,8,1])


def test_rotated_cube_contact_can_start_teacher_lift_at_wider_aperture():
    import importlib.util
    from pathlib import Path
    from fly_brain.teacher import grasp_pose
    path=Path(__file__).resolve().parents[1]/'scripts/collect_policy_recovery.py'
    spec=importlib.util.spec_from_file_location('recovery_collector_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    class Client:
        def call(self,*args,**kwargs):return {'joints_deg':[0]*6}
    teacher=module.RecoveryTeacher(Client(),{'cube_size_mm':20})
    teacher.stage='close'
    pose=grasp_pose([350,0],20,0);pose['position_mm'][2]=27
    observation={'cube_pose':{'position_mm':[350,0,9],'quaternion_xyzw':[0,0,-.22923,.97337]},
                 'grasp_pose':pose,'gripper_mm':26.7,'finger_contacts':{'left':True,'right':True},'joints_deg':[0]*6}
    target=teacher.target(observation,{'clearance_mm':0,'tilt_deg':0})
    assert teacher.stage=='lift'
    assert target['gripper_mm']==0


def test_recovery_keeps_every_observation_in_the_policy_history(tmp_path):
    write_recovery(tmp_path)
    path=tmp_path/'episode-recovery/steps.jsonl'
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    middle={**rows[0],'observation':{**rows[0]['observation'],'simulation_time':.2}}
    path.write_text('\n'.join(json.dumps(row) for row in [rows[0],middle,rows[1]]))
    examples,_=load_demonstrations(tmp_path,{'train':[[350,0]]},goal_supervision='aligned',sample_hz=2)
    assert [e['observation']['simulation_time'] for e in examples]==[0,.2,.5]


def test_reactive_labels_do_not_retreat_when_already_aligned(tmp_path):
    from fly_brain.teacher import grasp_pose
    directory=tmp_path/'episode-reactive';directory.mkdir()
    task={'cube_xy_mm':[350,0],'cube_size_mm':20,'cube_yaw_deg':0,'hold_seconds':5,'lift_clearance_mm':100}
    a=[0,-109,-104,85,0,0];g=[0,-112,-70,47,0,0];lift=[0,-105,-98,78,0,0]
    pose=grasp_pose([350,0],20,0);pose['position_mm'][2]=27
    cube={'position_mm':[350,0,9],'quaternion_xyzw':[0,0,-.22923,.97337]}
    obs={'episode_id':'a','simulation_time':0,'joints_deg':g,'gripper_mm':90,'images':[],
         'grasp_pose':pose,'cube_pose':cube,'finger_contacts':{'left':False,'right':False}}
    rows=[]
    for i,(stage,q) in enumerate([('approach',a),('recovery_align',a),('descend',g),('hold',lift)]):
        evaluation={'cube_pose':cube,'clearance_mm':0}
        current={**obs,'simulation_time':i*.5}
        if stage=='hold':
            current={**current,'finger_contacts':{'left':True,'right':True}}
            evaluation.update(success=True,held=True,episode_id='a',hold_seconds=5,clearance_mm=125,tilt_deg=0)
        rows.append({'stage':stage,'observation':current,'expert_target':{'joints_deg':q,'gripper_mm':0 if stage=='hold' else 90},
                     'expert_path':{'approach':a,'descend':g,'lift':lift},'evaluation':evaluation})
    meta={'complete':True,'success':True,'teacher_target_recorded':True,'supervision_contract':'recorded-recovery-v1',
          'provenance':'teacher_assisted','task':task,'result':{},'execution_mode':'absolute_targets'}
    (directory/'manifest.json').write_text(json.dumps(meta))
    (directory/'steps.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    examples,_=load_demonstrations(tmp_path,{'train':[[350,0]]},goal_supervision='aligned',grasp_goal_height_mm=27,recovery_target_policy='reactive')
    for e in examples[:2]:
        assert e['goal_role']=='grasp' and not e['preserve_expert_target']
        assert np.array_equal(e['target'],g+[0])
        assert e['grasp_pose_override']['position_mm']==[350,0,27]
    assert examples[-1]['target'][6]==0


def test_reactive_recipe_keeps_escape_and_lateral_recovery_commands():
    from fly_brain.visual_dopamine.trajectory_training import retain_recovery_target
    cube={'position_mm':[350,0,9]}
    row={'stage':'recovery_align','observation':{'grasp_pose':{'position_mm':[365,0,30]},'cube_pose':cube},'evaluation':{'cube_pose':cube}}
    assert retain_recovery_target(row,'reactive')
    row['observation']['grasp_pose']['position_mm']=[350,0,30]
    assert not retain_recovery_target(row,'reactive')
    assert retain_recovery_target({**row,'stage':'recovery_open'},'reactive')


def test_reactive_grasp_pose_uses_current_cube_not_a_later_reference():
    from fly_brain.teacher import grasp_pose
    fk=ToolKinematics();q=torch.zeros((1,6),dtype=torch.float64)
    pose=grasp_pose([342,-8],20,0);pose['position_mm'][2]=27
    example={'stage':'approach','goal_role':'grasp','reference_lift_clearance_mm':115,'grasp_pose_override':pose}
    after,rotation=demonstration_target_poses(fk,q,[example],125,0,27,[350,0],[350,0])
    assert torch.allclose(after[0],torch.tensor([342,-8,27],dtype=torch.float64))
    assert torch.allclose(rotation@rotation.transpose(1,2),torch.eye(3,dtype=torch.float64)[None],atol=1e-10)


def test_goal_weighting_follows_corrected_targets_and_visited_states():
    from fly_brain.visual_dopamine.feedback_training import demonstration_weights
    observation={'joints_deg':[0,-100,-70,40,0,0],'gripper_mm':40}
    examples=[{'stage':'approach','goal_role':'grasp','policy_visited':True,'observation':observation},
              {'stage':'approach','goal_role':'approach','policy_visited':True,'observation':observation},
              {'stage':'hold','goal_role':'transport','observation':observation}]
    legacy=demonstration_weights(examples,[0,1,2],grasp_frame_weight=8)
    assert np.allclose(legacy,[1,1,1])
    weighted=demonstration_weights(examples,[0,1,2],grasp_frame_weight=8,sample_weight_policy='goal_role',policy_visited_weight=8)
    assert np.allclose(weighted/weighted[-1],[8,8,1])
    assert np.isclose(weighted.mean(),1)


def test_recovery_validation_groups_have_equal_selection_influence():
    from fly_brain.visual_dopamine.feedback_training import validation_score
    def metrics(distance):return {'goal_position_rmse_mm':distance,'goal_orientation_rmse_deg':0,'gripper_accuracy':1}
    stats={**metrics(3),'groups':{'legacy':metrics(2),'recovery':metrics(10)}}
    assert validation_score(stats,True)==1.2
    assert validation_score(metrics(3),True)==.6


def test_wide_workspace_teacher_lifts_above_grip_and_preserves_label():
    import importlib.util
    from pathlib import Path
    from fly_brain.teacher import grasp_pose
    from fly_brain.visual_dopamine.trajectory_training import retain_recovery_target
    path=Path(__file__).resolve().parents[1]/'scripts/collect_policy_recovery.py'
    spec=importlib.util.spec_from_file_location('wide_teacher_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    poses=[]
    class Client:
        def call(self,*args,**kw):poses.append(kw['pose']);return {'joints_deg':[0]*6}
    teacher=module.RecoveryTeacher(Client(),{'cube_size_mm':20},hold_in_place=True)
    pose=grasp_pose([330,60],20,0);pose['position_mm'][2]=27
    obs={'cube_pose':{'position_mm':[330,60,9],'quaternion_xyzw':[0,0,0,1]},
         'grasp_pose':pose,'gripper_mm':20,'finger_contacts':{'left':True,'right':True},'joints_deg':[0]*6}
    teacher.target(obs,{'clearance_mm':0,'tilt_deg':0})
    assert np.allclose(poses[-1]['position_mm'],[330,60,152])
    assert retain_recovery_target({'stage':'lift','intervention':{'hold_in_place':True}},'reactive')
    assert not retain_recovery_target({'stage':'lift','intervention':{}},'reactive')
