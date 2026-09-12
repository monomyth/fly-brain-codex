from types import SimpleNamespace
import numpy as np
import pytest
from scipy import sparse
from fly_brain.visual_dopamine.motor_circuit import MotorCircuit
from fly_brain.visual_dopamine.motor_core import MotorLearner,motor_action,stimulation
from fly_brain.visual_dopamine.motor_policy import MotorPolicy
from fly_brain.visual_dopamine.core import VisualCore
from test_visual_dopamine import images


def circuit():
    c=SimpleNamespace(ids=np.arange(58),retina=np.array([0,1]),retina_uv=np.array([[.25,.5],[.75,.5]],dtype=np.float32),retina_side=np.array([0,1]),
        kc=np.arange(2,9),mbon=np.arange(9,16),dan=np.arange(16,23),cb=np.arange(23,30),dn=np.arange(30,37),pm=np.arange(37,44),motor=np.arange(44,58),
        plastic_base=np.eye(7,dtype=np.float32),dopamine_weights=np.eye(7,dtype=np.float32),
        cb_from_mbon=sparse.eye(7,format='csr'),dn_from_mbon_cb=sparse.hstack([sparse.eye(7)*.5,sparse.eye(7)*.5],format='csr'),
        pm_from_dn=sparse.eye(7,format='csr'),gain=np.ones(7,dtype=np.float32))
    c.motor_from_dn_pm=sparse.csr_matrix(([v for _ in range(7) for v in [1.,-1.]],([i for i in range(14)],[x for i in range(7) for x in [i,i+7]])),shape=(14,14))
    c.readout=np.zeros((7,14),dtype=np.float32)
    for axis in range(7):c.readout[axis,2*axis]=1;c.readout[axis,2*axis+1]=-1
    c.sensory=sparse.csr_matrix(([-1.]*7,(np.arange(2,9),[0,1,0,1,0,1,0])),shape=(58,58))
    c.output=lambda mbon:MotorCircuit.output(c,mbon)
    return c


def test_all_antagonist_pools_have_independent_bounded_motor_channels():
    c=circuit();obs={'joints_deg':[0,-10,-10,0,0,0],'gripper_mm':40}
    for axis in range(7):
        for direction in [-1,1]:
            scores,motor=stimulation(c,axis,direction)
            assert np.count_nonzero(scores)==1 and np.sign(scores[axis])==direction
            action=motor_action(obs,scores,selection='all')
            delta=np.r_[np.array(action['joints_deg'])-obs['joints_deg'],action['gripper_mm']-obs['gripper_mm']]
            expected=np.zeros(7);expected[axis]=direction*(4 if axis==6 else 2)
            assert np.array_equal(delta,expected)
    saturated=motor_action({'joints_deg':[0]*6,'gripper_mm':90},np.ones(7),selection='all')
    assert saturated['joints_deg'][1:3]==[0,0] and saturated['gripper_mm']==90
    with pytest.raises(ValueError):motor_action(obs,[np.nan]*7)


def test_each_neural_layer_and_pam_gate_are_used():
    c=circuit();on=MotorLearner(c,seed=3);off=MotorLearner(c,seed=3)
    choices,scores=on.choose(np.ones(7),True);off.choose(np.ones(7),True)
    assert all(np.any(x) for x in on.route.values()) and scores.shape==(7,)
    old=on.weights.copy();on.reinforce(1);off.reinforce(1,dopamine_enabled=False)
    assert not np.array_equal(on.weights,old) and np.array_equal(off.weights,old)
    assert (on.weights[~on.mask]==0).all()
    c.motor_from_dn_pm*=0
    _,blocked=on.choose(np.ones(7))
    assert np.array_equal(blocked,np.zeros(7)) and np.all(on.last_action==2)


def test_seven_motor_policy_uses_images_without_cube_or_goal_labels():
    c=circuit();p=object.__new__(MotorPolicy);p.metadata={};p.circuit=c;p.core=VisualCore(c);p.learner=MotorLearner(c)
    p.core.kc_mean.fill(-1.5)
    p.output_mode='directions';p.visual_enabled=True;p.state=np.zeros((1,58,1),dtype=np.float32);p.axis=None;p.selection='all';p.step_degrees=2;p.step_mm=4
    obs={'images':images(),'episode_id':'test','frame_id':1,'joints_deg':[0,-10,-10,0,0,0],'gripper_mm':40}
    p.reset(obs);first=p.act(obs)
    changed={**obs,'cube_pose':{'position_mm':[999]*3},'expert_action':[99]*7,'reward':1,'stage':'lift','task_for_evaluator_only':{}}
    p.reset(changed);second=p.act(changed)
    assert first==second and first['joints_deg']!=obs['joints_deg']
    with pytest.raises(ValueError,match='frame'):p.act(changed)
    p.visual_enabled=False;p.reset(changed)
    assert p.act(changed)=={'joints_deg':obs['joints_deg'],'gripper_mm':40.}


def test_motor_checkpoint_round_trip_preserves_mapping_and_learning(tmp_path,monkeypatch):
    from fly_brain.visual_dopamine.motor_training import save_checkpoint
    from fly_brain.assets import sha256_file,artifact_manifest
    from fly_brain.checkpoint import export_checkpoint
    import json
    import fly_brain.visual_dopamine.motor_policy as policy_module
    c=circuit();c.manifest={'graph_id':'a'*64,'circuit_id':'b'*64};c.motor_manifest={'motor_circuit_id':'c'*64}
    monkeypatch.setattr(policy_module,'MotorCircuit',lambda *args,**kwargs:c)
    core=VisualCore(c);core.kc_mean.fill(-1.5);learner=MotorLearner(c)
    checkpoint=tmp_path/'initial';save_checkpoint(checkpoint,core,learner,{'seed':0,'dopamine_enabled':True})
    original=sha256_file(checkpoint/'model.npz')
    policy=MotorPolicy(checkpoint,tmp_path)
    observation={'images':images(),'episode_id':'test','frame_id':1,'joints_deg':[0,-10,-10,0,0,0],'gripper_mm':40}
    policy.reset(observation);policy.act(observation,explore=True)
    assert policy.feedback(1,learn=True)['weight_change_l1']>0
    learned=tmp_path/'learned';policy.save(learned,{'attempts':1})
    restored=MotorPolicy(learned,tmp_path)
    assert np.array_equal(policy.learner.weights,restored.learner.weights)
    assert policy.learner.rng.bit_generator.state==restored.learner.rng.bit_generator.state
    assert sha256_file(checkpoint/'model.npz')==original
    exported=tmp_path/'consumer';export_checkpoint(learned,exported,root=tmp_path)
    assert json.loads((exported/'checkpoint/manifest.json').read_text())['schema']=='seven-motor-checkpoint-v1'
    with np.load(learned/'model.npz',allow_pickle=False) as arrays:data={key:arrays[key].copy() for key in arrays.files}
    data['readout'][0,0]*=-1
    np.savez(learned/'model.npz',**data)
    metadata=json.loads((learned/'manifest.json').read_text());metadata.pop('files');artifact_manifest(learned,metadata)
    with pytest.raises(ValueError,match='muscle-pool'):MotorPolicy(learned,tmp_path)


def test_synaptic_credit_matches_independent_forward_finite_differences():
    c=circuit();learner=MotorLearner(c)
    features=np.linspace(-.2,.7,7,dtype=np.float32)
    learner.choose(features)
    for axis in [0,3,6]:
        analytic=learner._score_gradient(axis)[axis]*learner.rates[axis]
        original=learner.weights[axis,axis];epsilon=.001
        learner.weights[axis,axis]=original+epsilon
        upper=c.output(np.tanh(learner.weights@learner.rates))[0][axis]
        learner.weights[axis,axis]=original-epsilon
        lower=c.output(np.tanh(learner.weights@learner.rates))[0][axis]
        learner.weights[axis,axis]=original
        assert float(analytic)==pytest.approx(float((upper-lower)/(2*epsilon)),rel=.002,abs=.0001)


def test_motionless_hold_does_not_trip_stagnation_timeout():
    from fly_brain.visual_dopamine.motor_run import update_stagnation
    previous=None
    for now in np.arange(0,12,.5):
        previous=update_stagnation(previous,0,{'held':True,'hold_seconds':min(now,5)},now)
        assert previous is None
    previous=update_stagnation(None,0,{'held':False,'hold_seconds':0},10.)
    assert update_stagnation(previous,0,{'held':False,'hold_seconds':0},20.)==10.
    assert update_stagnation(previous,.1,{'held':False},21.) is None


def test_inhibitory_ascending_inputs_keep_their_sign_and_gradient():
    from fly_brain.visual_dopamine.trajectory_training import MotorTargetNetwork
    import torch
    c=circuit();c.input_signs=np.array([1,-1,1,-1,1,-1,1],dtype=np.float32)
    learner=MotorLearner(c);learner.configure_centered_response()
    x=np.linspace(-.7,.8,7,dtype=np.float32)
    _,scores=learner.choose(x)
    model=MotorTargetNetwork(c)
    predicted=model(torch.tensor(x[None])).detach().numpy()[0]
    assert np.allclose(scores,predicted,atol=1e-7)
    model.zero_grad();model(torch.tensor(x[None]))[0,1].backward()
    analytic=learner._score_gradient(1)[1]*learner.rates[1]*c.input_signs[1]
    assert float(model.weights.grad[1,1])==pytest.approx(float(analytic),rel=.001)
    learner.choose(x,explore=True,exploration_axis=1)
    old_weights=learner.weights.copy();old_offset=learner.excitability.copy()
    learner.reinforce(1,dopamine_enabled=False)
    assert np.array_equal(old_weights,learner.weights)
    assert np.array_equal(old_offset,learner.excitability)


def test_body_feedback_uses_encoders_and_contacts_without_task_geometry():
    from fly_brain.visual_dopamine.feedback import BodyEncoder
    c=SimpleNamespace(body_indices=np.arange(9),sensory_groups=[
        {'kind':'position','channel':0,'indices':[0,1,2]},
        {'kind':'position','channel':6,'indices':[3,4,5]},
        {'kind':'contact','channel':'L','indices':[6,7]},
        {'kind':'contact','channel':'R','indices':[8]}])
    encoder=BodyEncoder(c)
    obs={'joints_deg':[0]*6,'gripper_mm':40,'finger_contacts':{'left':False,'right':False}}
    a=encoder.sample(obs)
    assert np.array_equal(a,encoder.sample({**obs,'cube_pose':{'position_mm':[999]*3},'stage':'hold','reward':1}))
    b=encoder.sample({**obs,'joints_deg':[30,0,0,0,0,0],'finger_contacts':{'left':True,'right':False}})
    assert not np.array_equal(a[:3],b[:3])
    assert np.array_equal(b[6:],[1,1,0])
    with pytest.raises(ValueError):encoder.sample({**obs,'gripper_mm':float('nan')})


def test_expert_delta_preserves_coordinated_direction_and_hz():
    from fly_brain.visual_dopamine.trajectory_training import expert_delta
    assert np.array_equal(expert_delta([0,-90,-45,30,0,0],[0]*6,2),[0,-15,-7.5,5,0,0])


def test_batched_feedback_matches_independent_scipy_propagation():
    from fly_brain.visual_dopamine.feedback import FeedbackCore,TorchFeedbackEncoder
    c=circuit();c.visual_kc=c.kc[:5];c.ascending=c.kc[5:];c.body_indices=np.array([55,56]);c.sensory_groups=[]
    c.feedback_manifest={'recipe':{'version':2}}
    matrix=c.sensory.tolil();matrix[c.ascending[0],55]=.3;matrix[c.ascending[1],56]=-.2;c.sensory=matrix.tocsr()
    core=FeedbackCore(c)
    retinal=np.array([[.1,.2],[.7,.4],[.3,.5]],dtype=np.float32)
    body=np.array([[.9,.7],[.1,.3],[.2,.4]],dtype=np.float32)
    expected=np.array([core.response(a,b) for a,b in zip(retinal,body)])
    actual=TorchFeedbackEncoder(core,'cpu').responses(retinal,body,batch_size=2)
    assert np.allclose(actual,expected,atol=1e-7)


def test_learned_motor_synapses_round_trip_without_changing_anatomical_signs(tmp_path,monkeypatch):
    from fly_brain.visual_dopamine.motor_training import save_checkpoint
    import fly_brain.visual_dopamine.motor_policy as policy_module
    c=circuit();c.manifest={'graph_id':'a'*64,'circuit_id':'b'*64};c.motor_manifest={'motor_circuit_id':'c'*64}
    def load_circuit(*args,**kwargs):
        other=circuit();other.manifest=c.manifest;other.motor_manifest=c.motor_manifest;return other
    monkeypatch.setattr(policy_module,'MotorCircuit',load_circuit)
    core=VisualCore(c);learner=MotorLearner(c);learner.configure_centered_response()
    c.motor_from_dn_pm.data*=.7
    save_checkpoint(tmp_path/'model',core,learner,{'seed':0,'dopamine_enabled':True,'motor_synapse_plasticity':True})
    p=MotorPolicy(tmp_path/'model',tmp_path)
    assert np.array_equal(p.circuit.motor_from_dn_pm.data,c.motor_from_dn_pm.data)
    c.motor_from_dn_pm.data[0]*=-1
    save_checkpoint(tmp_path/'bad-sign',core,learner,{'seed':0,'dopamine_enabled':True,'motor_synapse_plasticity':True})
    with pytest.raises(ValueError,match='motor synapses'):MotorPolicy(tmp_path/'bad-sign',tmp_path)


def test_visual_student_cannot_receive_teacher_cube_pose():
    from fly_brain.collect import actor_observation
    from types import SimpleNamespace
    original={'input_mode':'state','cube_pose':{'position_mm':[350,0,9]},'joints_deg':[0]*6,'gripper_mm':20,'images':images()}
    visible=actor_observation(original,SimpleNamespace(mode='vision'))
    assert 'cube_pose' not in visible and visible['input_mode']=='vision'
    assert 'cube_pose' in original and visible['images']==original['images']


def test_teacher_levels_cube_when_camera_observation_omits_cube_pose():
    from fly_brain.teacher import Teacher
    from fly_brain.adapters import ActionAdapter
    requests=[]
    def solve(name,**arguments):
        requests.append(arguments);return {'joints_deg':[0]*6}
    teacher=object.__new__(Teacher);teacher.client=SimpleNamespace(call=solve)
    teacher.task={'cube_size_mm':20,'cube_xy_mm':[350,0],'cube_yaw_deg':0,'lift_clearance_mm':100,'hold_seconds':5}
    teacher.adapter=ActionAdapter(2);teacher.stage='hold';teacher.stage_started=0
    teacher.path={'lift':[0]*6};teacher.hold_joints=[0]*6;teacher.hold_succeeded=False;teacher.release_required=True;teacher.recovery_count=0
    obs={'simulation_time':1,'joints_deg':[0]*6,'gripper_mm':20,'finger_contacts':{'left':True,'right':True},
         'grasp_pose':{'position_mm':[350,0,51],'quaternion_xyzw':[0,0,0,1]}}
    evaluation={'cube_pose':{'position_mm':[350,0,39],'quaternion_xyzw':[0,0,0,1]},'clearance_mm':30,'success':False}
    teacher.act(obs,evaluation)
    assert requests[0]['pose']['position_mm'][2]==pytest.approx(136)
    assert 'cube_pose' not in obs



def test_absolute_neural_target_servo_converges_without_integrating_bias(tmp_path,monkeypatch):
    from fly_brain.visual_dopamine.motor_training import save_checkpoint
    import fly_brain.visual_dopamine.motor_policy as policy_module
    c=circuit();c.manifest={'graph_id':'a'*64,'circuit_id':'b'*64};c.motor_manifest={'motor_circuit_id':'c'*64}
    monkeypatch.setattr(policy_module,'MotorCircuit',lambda *a,**k:c)
    core=VisualCore(c);learner=MotorLearner(c);learner.configure_centered_response()
    save_checkpoint(tmp_path/'model',core,learner,{'seed':0,'dopamine_enabled':True,'architecture':{
        'kind':'malecns_motor_dopamine','updates':12,'output_mode':'joint_targets','output_channels':7,
        'joint_gain':1440.,'control_hz':2.,'position_servo':True,'command_scale':.5}})
    policy=MotorPolicy(tmp_path/'model',tmp_path)
    target=np.array([0,-100,-80,50,0,0],dtype=float)
    policy.learner.choose=lambda *a,**k:(np.zeros(7,dtype=int),np.r_[target/1440,1.])
    observation={'episode_id':'servo','frame_id':1,'images':images(),'joints_deg':[0,-10,-10,0,0,0],'gripper_mm':40}
    policy.reset(observation)
    for frame in range(1,80):
        observation['frame_id']=frame;before=np.array(observation['joints_deg']);action=policy.act(observation)
        assert np.max(np.abs(np.array(action['joints_deg'])-before))<=7.5+1e-8
        observation.update(action)
    assert np.allclose(observation['joints_deg'],target,atol=1e-5)


def test_fixed_workspace_retina_uses_camera_calibration_without_cube_pose():
    from fly_brain.visual_dopamine.core import RetinalEncoder
    encoder=RetinalEncoder(circuit())
    encoder.configure({'kind':'fixed_workspace','lower_m':[-.2,-.2,-1.1],'upper_m':[.2,.2,-.9]})
    camera={'name':'Front','world_from_camera':np.eye(4).tolist(),'intrinsics':[[100,0,50],[0,100,50],[0,0,1]],'world_units':'meters'}
    rgb=np.arange(100*100*3,dtype=np.float32).reshape(100,100,3)
    cropped=encoder.image_region(rgb,camera)
    assert 40<cropped.shape[0]<60 and 40<cropped.shape[1]<60
    assert np.array_equal(cropped,encoder.image_region(rgb,{**camera,'cube_pose':[999,999,999]}))
    with pytest.raises(ValueError):encoder.configure({'kind':'fixed_workspace','lower_m':[0,0,0],'upper_m':[0,1,1]})


def test_verified_hold_is_separate_from_a_failed_release():
    from fly_brain.visual_dopamine.trajectory_training import verified_native_hold
    e={'episode_id':'e','success':True,'held':True,'hold_seconds':5.,'clearance_mm':115.,'tilt_deg':.5}
    row={'observation':{'episode_id':'e'},'evaluation':e}
    assert verified_native_hold([row],{'hold_seconds':5,'lift_clearance_mm':100,'tilt_tolerance_deg':5})
    assert not verified_native_hold([{**row,'evaluation':{**e,'held':False,'clearance_mm':0}}],{})
    assert not verified_native_hold([{**row,'observation':{'episode_id':'different'}}],{})


def test_observable_goal_selection_does_not_depend_on_waypoint_index():
    from fly_brain.visual_dopamine.trajectory_training import observable_goal_target
    approach={'joints_deg':[1]*6,'gripper_mm':90};grasp={'joints_deg':[2]*6,'gripper_mm':90};lift={'joints_deg':[3]*6,'gripper_mm':0}
    obs={'finger_contacts':{'left':False,'right':False},'grasp_pose':{'position_mm':[350,0,60],'quaternion_xyzw':[0,2**-.5,0,2**-.5]}}
    row={'observation':obs,'evaluation':{'cube_pose':{'position_mm':[350,0,9],'quaternion_xyzw':[0,0,0,1]},'clearance_mm':0},'expert_target':approach,'stage':'approach'}
    first=observable_goal_target(row,{'cube_size_mm':20},approach,grasp,lift)
    second=observable_goal_target({**row,'stage':'descend'}, {'cube_size_mm':20},approach,grasp,lift)
    assert first==second==grasp


def test_teacher_accepts_a_dropped_cube_landing_on_another_face():
    from fly_brain.teacher import Teacher
    from fly_brain.adapters import ActionAdapter
    teacher=object.__new__(Teacher);teacher.task={'hold_seconds':5,'lift_clearance_mm':100};teacher.adapter=ActionAdapter(2)
    teacher.stage='release';teacher.stage_started=0;teacher.hold_succeeded=True;teacher.path={};teacher.hold_joints=[0]*6
    obs={'simulation_time':1,'joints_deg':[0]*6,'gripper_mm':90,'finger_contacts':{'left':False,'right':False}}
    evaluation={'cube_pose':{'position_mm':[350,0,9],'quaternion_xyzw':[1,0,0,0]},'clearance_mm':0,'tilt_deg':90,'held':False,'contacts':['floor'],'linear_speed_mm_s':0,'success':True}
    teacher.act(obs,evaluation)
    assert teacher.stage=='done' and teacher.hold_succeeded


@pytest.mark.parametrize('updates',[4,8,12])
def test_scheduled_sensory_matches_recurrent_signed_graph_with_clamped_inputs(updates):
    from fly_brain.visual_dopamine.feedback import FeedbackCore,ScheduledFeedbackEncoder
    c=circuit();c.visual_kc=c.kc[:5];c.ascending=c.kc[5:];c.body_indices=np.array([55,56]);c.sensory_groups=[]
    c.feedback_manifest={'recipe':{'version':2}}
    rng=np.random.default_rng(45)
    weights=rng.normal(0,.1,(58,58)).astype(np.float32);weights[rng.random(weights.shape)>.07]=0
    c.sensory=sparse.csr_matrix(weights)
    core=FeedbackCore(c,updates=updates)
    retinal=rng.random((7,2),dtype=np.float32);body=rng.random((7,2),dtype=np.float32)
    reference=np.array([core.response(a,b) for a,b in zip(retinal,body)])
    actual=ScheduledFeedbackEncoder(core).responses(retinal,body,batch_size=3)
    assert np.allclose(actual,reference,rtol=1e-5,atol=1e-7)


def test_sampling_retains_stage_transitions_and_the_final_hold_frame(tmp_path):
    import json
    from fly_brain.visual_dopamine.trajectory_training import load_demonstrations
    episode=tmp_path/'episode-a';episode.mkdir()
    meta={'complete':True,'success':True,'teacher_target_recorded':True,'task':{'cube_xy_mm':[350,0]},'hz':5,'result':{},'execution_mode':'absolute_targets'}
    (episode/'manifest.json').write_text(json.dumps(meta))
    stages=['approach']*2+['descend']*3+['hold']*4+['release']
    rows=[{'stage':stage,'observation':{'simulation_time':i*.2,'joints_deg':[0]*6,'gripper_mm':0,'images':[]},'expert_target':{'joints_deg':[0]*6,'gripper_mm':0},'evaluation':{}} for i,stage in enumerate(stages)]
    (episode/'steps.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    full,_=load_demonstrations(tmp_path,{'train':[[350,0]]})
    sampled,_=load_demonstrations(tmp_path,{'train':[[350,0]]},sample_hz=2)
    assert len(full)==9
    assert [r['step'] for r in sampled]==[0,2,5,8]



def test_gripper_teaching_distinguishes_lateral_miss_from_vertical_clearance():
    from fly_brain.visual_dopamine.trajectory_training import observable_goal_target
    from fly_brain.teacher import grasp_pose
    task={'cube_size_mm':20};pose=grasp_pose([350,0],20,0)
    observation={'grasp_pose':{**pose,'position_mm':[339,0,21]},'finger_contacts':{'left':False,'right':False},'cube_pose':{'position_mm':[350,0,9],'quaternion_xyzw':[0,0,0,1]}}
    row={'observation':observation,'stage':'descend','evaluation':{'cube_pose':{'position_mm':[355,0,9],'quaternion_xyzw':[0,0,0,1]},'clearance_mm':0}}
    goal={'joints_deg':[0]*6,'gripper_mm':90}
    assert observable_goal_target(row,task,goal,goal,goal,require_lateral_alignment=True)['gripper_mm']==90
    observation['grasp_pose']['position_mm']=[350,0,31]
    assert observable_goal_target(row,task,goal,goal,goal,require_lateral_alignment=True)['gripper_mm']==0
    observation['finger_contacts']={'left':True,'right':True}
    row['expert_target']={**goal,'gripper_mm':0};row['stage']='hold'
    assert observable_goal_target(row,task,goal,goal,goal,require_lateral_alignment=True)['gripper_mm']==0


def test_raised_grasp_uses_matching_height_for_closure_labels():
    from fly_brain.visual_dopamine.trajectory_training import observable_goal_target
    from fly_brain.teacher import grasp_pose
    pose=grasp_pose([350,0],20,0);pose['position_mm'][2]=36
    row={'stage':'descend','observation':{'grasp_pose':pose,'finger_contacts':{'left':False,'right':False}},
         'evaluation':{'cube_pose':{'position_mm':[350,0,9],'quaternion_xyzw':[0,0,0,1]},'clearance_mm':0}}
    goal={'joints_deg':[0]*6,'gripper_mm':90}
    assert observable_goal_target(row,{'cube_size_mm':20},goal,goal,goal,True)['gripper_mm']==90
    assert observable_goal_target(row,{'cube_size_mm':20},goal,goal,goal,True,27)['gripper_mm']==0
    pose['position_mm'][1]=8
    assert observable_goal_target(row,{'cube_size_mm':20},goal,goal,goal,True,27)['gripper_mm']==90


def test_run_rejects_unsafe_settle_overrides_before_starting_an_episode(tmp_path):
    from fly_brain.visual_dopamine.motor_run import run
    for delay in [0.,-.1,.01,.6,float('nan'),float('inf')]:
        with pytest.raises(ValueError,match='Settle override'):
            run(None,None,[{}],tmp_path/'unused',settle_seconds=delay)
    assert not (tmp_path/'unused').exists()
