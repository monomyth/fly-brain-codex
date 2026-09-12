from types import SimpleNamespace
import numpy as np
import pytest
from scipy import sparse
from fly_brain.visual_dopamine.temporal import TemporalFeedbackCore,TemporalFeedbackEncoder,sequence_responses


def fixture():
    matrix=np.zeros((6,6),dtype=np.float32)
    matrix[1,0]=.5;matrix[1,1]=.7;matrix[2,1]=.5;matrix[2,2]=.4
    matrix[4,3]=.6;matrix[4,4]=.6;matrix[5,4]=.5;matrix[5,5]=.4
    return SimpleNamespace(ids=np.arange(6),retina=np.array([0]),visual_kc=np.array([2]),ascending=np.array([5]),kc=np.array([2,5]),body_indices=np.array([3]),sensory_groups=[],sensory=sparse.csr_matrix(matrix),feedback_manifest={'recipe':{'version':2},'feedback_circuit_id':'a'*64})


def test_temporal_response_depends_on_history_and_reset_restores_initial_state():
    core=TemporalFeedbackCore(fixture(),updates=4)
    first=core.response([1.],[1.],simulation_time=0)
    remembered=core.response([0.],[0.],simulation_time=.5)
    fresh=TemporalFeedbackCore(fixture(),updates=4).response([0.],[0.],simulation_time=.5)
    assert np.linalg.norm(remembered-fresh)>.001
    core.reset()
    assert np.array_equal(first,core.response([1.],[1.],simulation_time=0))
    with pytest.raises(ValueError,match='increasing'):core.response([1.],[1.],simulation_time=0)
    with pytest.raises(ValueError,match='maximum gap'):core.response([1.],[1.],simulation_time=3)


def test_motor_display_state_cannot_contaminate_sensory_memory():
    a=TemporalFeedbackCore(fixture(),updates=4);b=TemporalFeedbackCore(fixture(),updates=4)
    a.response([1.],[.3],simulation_time=0);b.response([1.],[.3],simulation_time=0)
    a.state.fill(999)
    assert np.array_equal(a.response([.2],[.7],simulation_time=.3),b.response([.2],[.7],simulation_time=.3))


def test_batched_episode_encoding_matches_independent_live_updates():
    examples=[{'episode':episode,'observation':{'simulation_time':time}} for episode,times in [('a',[0,.2,.7]),('b',[0,.4])] for time in times]
    r=np.array([[1.],[.2],[.7],[.1],[.8]],dtype=np.float32);body=1-r
    core=TemporalFeedbackCore(fixture(),updates=4)
    expected=[];last=None
    for image,b,e in zip(r,body,examples):
        if e['episode']!=last:core.reset();last=e['episode']
        expected.append(core.response(image,b,simulation_time=e['observation']['simulation_time']))
    actual=TemporalFeedbackEncoder(core).responses(r,body,examples,batch_size=2)
    assert np.allclose(actual,expected,atol=1e-7)
    changed=r.copy();changed[0]=0
    second=TemporalFeedbackEncoder(core).responses(changed,body,examples,batch_size=2)
    assert np.array_equal(actual[3:],second[3:])
    assert not np.allclose(actual[2],second[2])


def test_sequence_cache_keys_include_history_and_episode_boundaries(tmp_path):
    core=TemporalFeedbackCore(fixture(),updates=4)
    examples=[{'episode':'a','observation':{'simulation_time':t}} for t in [0,.5]]
    r=np.array([[1.],[.2]],dtype=np.float32);body=np.array([[.1],[.4]],dtype=np.float32)
    first,report=sequence_responses(core,r,body,examples,tmp_path)
    second,cached=sequence_responses(core,r,body,examples,tmp_path)
    assert report['encoded_episodes']==1 and cached['reused_episodes']==1
    assert np.array_equal(first,second)
    changed=r.copy();changed[0]=0
    third,new=sequence_responses(core,changed,body,examples,tmp_path)
    assert new['encoded_episodes']==1 and not np.allclose(first[-1],third[-1])
    split=[{**e,'episode':str(i)} for i,e in enumerate(examples)]
    independent,_=sequence_responses(core,r,body,split,tmp_path)
    assert not np.allclose(first[-1],independent[-1])


def test_temporal_checkpoint_round_trip_and_policy_reset(tmp_path,monkeypatch):
    from test_motor_control import circuit
    from test_visual_dopamine import images
    from fly_brain.visual_dopamine.motor_core import MotorLearner
    from fly_brain.visual_dopamine.motor_policy import MotorPolicy
    from fly_brain.visual_dopamine.motor_training import save_checkpoint
    import fly_brain.visual_dopamine.feedback as feedback
    def make(*args):
        c=circuit();c.visual_kc=c.kc[:5];c.ascending=c.kc[5:];c.body_indices=np.array([55,56]);c.sensory_groups=[]
        c.manifest={'graph_id':'a'*64,'circuit_id':'b'*64};c.motor_manifest={'motor_circuit_id':'c'*64}
        c.feedback_manifest={'recipe':{'version':2},'feedback_circuit_id':'d'*64};return c
    c=make();core=TemporalFeedbackCore(c);learner=MotorLearner(c);learner.configure_centered_response()
    save_checkpoint(tmp_path/'model',core,learner,{'seed':0,'dopamine_enabled':True})
    monkeypatch.setattr(feedback,'FeedbackCircuit',make)
    policy=MotorPolicy(tmp_path/'model',tmp_path)
    assert policy.core.temporal_config==core.temporal_config
    observation={'episode_id':'test','frame_id':1,'simulation_time':0.,'images':images(),'joints_deg':[0]*6,'gripper_mm':40,'finger_contacts':{'left':False,'right':False}}
    policy.reset(observation);first=policy.act(observation)
    policy.act({**observation,'frame_id':2,'simulation_time':.5})
    policy.reset(observation);second=policy.act({**observation,'cube_pose':{'position_mm':[999]*3},'reward':1})
    assert first==second
    assert policy.core.last_time==0


def test_temporal_runtime_does_not_silently_use_stateless_remote_encoder(tmp_path,monkeypatch):
    from fly_brain.visual_dopamine.remote_features import configure_runtime
    p=SimpleNamespace(root=tmp_path,metadata={'temporal_sensory':{'kind':'persistent-sensory-v1'},'feedback_circuit_id':'a'*64})
    with pytest.raises(ValueError,match='stateless encoding'):configure_runtime(p,'unreachable.example')
    assert configure_runtime(p,'local') is p
