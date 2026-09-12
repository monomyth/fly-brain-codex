import base64
import io
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from scipy import sparse
from fly_brain.visual_dopamine.core import RetinalEncoder, VisualCore, DopamineLearner
from fly_brain.visual_dopamine.policy import VisualDopaminePolicy


def circuit():
    return SimpleNamespace(ids=np.arange(14),retina=np.array([0,1]),retina_uv=np.array([[.25,.5],[.75,.5]],dtype=np.float32),
        retina_side=np.array([0,1]),kc=np.array([4,5]),mbon=np.array([6,7]),dan=np.array([8,9]),dn=np.array([10,11]),motor=np.array([12,13]),
        sensory=sparse.csr_matrix(([-1.,-1.,1.,1.],([2,3,4,5],[0,1,2,3])),shape=(14,14),dtype=np.float32),
        plastic_base=np.array([[.5,.5],[0,1]],dtype=np.float32),dn_weights=np.eye(2,dtype=np.float32),
        motor_weights=np.eye(2,dtype=np.float32),dopamine_weights=np.eye(2,dtype=np.float32),
        neurons={'motor':[{'type':'Ti extensor MN'},{'type':'Ti flexor MN'}]})


def images():
    result=[]
    for name,half in [('Front',0),('Top',1)]:
        rgb=np.zeros((32,32,3),dtype=np.uint8);rgb[:,half*16:(half+1)*16]=255
        data=io.BytesIO();Image.fromarray(rgb).save(data,format='PNG')
        result.append({'name':name,'jpeg_base64':base64.b64encode(data.getvalue()).decode()})
    return result


def test_pixels_drive_retina_and_visual_path():
    c=circuit();core=VisualCore(c,updates=5)
    retinal=core.encoder.sample(images())
    assert (retinal>.9).all()
    response=core.visual_response(retinal)
    assert (core.state[c.retina]>.9).all() and (core.state[[2,3]]<0).all()
    assert (response<0).all()
    assert np.array_equal(core.encode(retinal,enabled=False),[0,0])
    with pytest.raises(ValueError,match='Front and Top'):core.encoder.sample(images()[:1])


def test_dopamine_is_required_to_change_anatomical_synapses():
    c=circuit();on=DopamineLearner(c,seed=3);off=DopamineLearner(c,seed=3)
    on.choose([1,.4],True);off.choose([1,.4],True)
    before=off.weights.copy()
    pulse=on.reinforce(1,delay_s=.2)
    blocked=off.reinforce(1,delay_s=.2,dopamine_enabled=False)
    assert pulse['dopamine_mean']>0 and pulse['weight_change_l1']>0
    assert blocked['dopamine_mean']==0 and np.array_equal(off.weights,before)
    assert on.weights[1,0]==0 and (on.weights>=0).all() and (on.weights<=4*c.plastic_base).all()


def test_reward_without_plasticity_does_not_train_and_old_eligibility_decays():
    c=circuit();frozen=DopamineLearner(c,seed=3);recent=DopamineLearner(c,seed=3);late=DopamineLearner(c,seed=3)
    for learner in [frozen,recent,late]:learner.choose([1,.4],True)
    before=frozen.weights.copy()
    assert frozen.reinforce(1,plasticity_enabled=False)['dopamine_mean']>0
    assert np.array_equal(frozen.weights,before)
    a=recent.reinforce(1,delay_s=0);b=late.reinforce(1,delay_s=200)
    assert a['weight_change_l1']>b['weight_change_l1']
    assert np.array_equal(late.weights,c.plastic_base)


def test_policy_ignores_cube_pose_and_reward_fields_until_feedback():
    c=circuit()
    policy=object.__new__(VisualDopaminePolicy)
    policy.circuit=c;policy.core=VisualCore(c,updates=5);policy.learner=DopamineLearner(c)
    policy.state=np.zeros((1,14,1),dtype=np.float32);policy.visual_enabled=True
    observation={'images':images(),'episode_id':'test','frame_id':1,'joints_deg':[0]*6,'gripper_mm':0}
    policy.reset(observation);first=policy.act(observation)
    fake={**observation,'cube_pose':{'position_mm':[999,-999,999]},'reward':1,'expert_action':[99]*7,'task_time':999}
    policy.reset(fake);second=policy.act(fake)
    assert first==second
    assert second['joints_deg'][1:]==[0]*5
    with pytest.raises(ValueError,match='frame'):policy.act(fake)


def test_only_innervated_plastic_targets_receive_dopamine_updates():
    c=circuit();c.dopamine_weights[1]=0
    learner=DopamineLearner(c,seed=9);before=learner.weights.copy()
    learner.choose([.6,1.],True);learner.reinforce(1)
    assert not np.array_equal(learner.weights[0],before[0])
    assert np.array_equal(learner.weights[1],before[1])


def saved_fixture(tmp_path):
    import json
    from fly_brain.assets import artifact_manifest
    c=circuit();identity='a'*64;graph='b'*64
    shared=tmp_path/'shared';directory=shared/'circuits'/identity;directory.mkdir(parents=True)
    np.savez(directory/'populations.npz',**{name:getattr(c,name) for name in ['ids','retina','retina_uv','retina_side','kc','mbon','dan','dn','motor']})
    np.savez(directory/'routes.npz',**{name:getattr(c,name) for name in ['plastic_base','dn_weights','motor_weights','dopamine_weights']})
    sparse.save_npz(directory/'sensory-signed.npz',c.sensory)
    (directory/'neurons.json').write_text(json.dumps(c.neurons))
    artifact_manifest(directory,{'graph_id':graph,'circuit_id':identity})
    checkpoint=tmp_path/'original';checkpoint.mkdir()
    core=VisualCore(c,updates=5);learner=DopamineLearner(c)
    np.savez(checkpoint/'model.npz',weights=learner.weights,retinal_mean=core.retinal_mean,
             retinal_std=core.retinal_std,kc_mean=core.kc_mean,kc_std=core.kc_std,
             motor_readout=learner.motor_readout,baseline=np.array(.5))
    artifact_manifest(checkpoint,{'schema':'visual-dopamine-checkpoint-v1','name':'test','graph_id':graph,
        'circuit_id':identity,'architecture':{'kind':'malecns_visual_dopamine','updates':5},'seed':0,
        'test_result':{'successes':1},'dopamine_enabled':True})
    return checkpoint,shared


def test_online_checkpoint_is_new_loadable_and_exportable(tmp_path,monkeypatch):
    from fly_brain.checkpoint import export_checkpoint
    from fly_brain.assets import sha256_file
    checkpoint,shared=saved_fixture(tmp_path)
    original_hash=sha256_file(checkpoint/'model.npz')
    policy=VisualDopaminePolicy(checkpoint,shared)
    observation={'images':images(),'episode_id':'test','frame_id':1,'joints_deg':[0]*6,'gripper_mm':0}
    policy.reset(observation);policy.act(observation,explore=True)
    feedback=policy.feedback(1,learn=True)
    assert feedback['weight_change_l1']>0
    learned=tmp_path/'learned';policy.save(learned,{'attempts':1,'successes':1})
    restored=VisualDopaminePolicy(learned,shared)
    assert np.array_equal(policy.learner.weights,restored.learner.weights)
    assert policy.learner.rng.bit_generator.state==restored.learner.rng.bit_generator.state
    policy.reset(observation)
    assert policy.activity_metadata()['dopamine']=={'signal':0.,'reward':0.,'cells':2,'weight_change_l1':0.}
    assert 'test_result' not in restored.metadata and restored.metadata['closed_loop_validated'] is False
    assert sha256_file(checkpoint/'model.npz')==original_hash
    with pytest.raises(FileExistsError):policy.save(learned,{})
    export=tmp_path/'consumer';export_checkpoint(learned,export,shared)
    monkeypatch.chdir(export)
    independent=VisualDopaminePolicy(export/'checkpoint',shared)
    independent.reset(observation);restored.reset(observation)
    assert independent.act(observation)==restored.act(observation)


def test_invalid_learner_parameters_and_missing_camera_fail():
    for values in [{'learning_rate':float('nan')},{'noise':float('inf')},{'eligibility_tau':0}]:
        with pytest.raises(ValueError):DopamineLearner(circuit(),**values)
    with pytest.raises(ValueError):VisualCore(circuit()).encoder.sample([])
