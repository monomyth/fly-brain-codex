import numpy as np
import pytest
from fly_brain.visual_dopamine.motor_policy import MotorPolicy
from fly_brain.visual_dopamine.motor_core import MotorLearner
from test_motor_control import circuit


def policy(tmp_path):
    c=circuit();c.motor_synapse_base=c.motor_from_dn_pm.copy()
    p=object.__new__(MotorPolicy);p.circuit=c;p.learner=MotorLearner(c);p.output_mode='joint_targets';p.checkpoint=tmp_path
    p.metadata={'motor_synapse_plasticity':True,'touch_gated_hold':{'version':1,'gate':'bilateral_finger_contact','file':'holding-model.npz'}}
    np.savez(tmp_path/'holding-model.npz',weights=2*p.learner.weights,excitability=np.full(7,.1,dtype=np.float32),motor_synapses=c.motor_from_dn_pm.data)
    p.load_touch_banks();return p


def test_touch_switches_learned_response_and_restores_exact_pickup(tmp_path):
    p=policy(tmp_path);features=np.linspace(-.2,.5,7,dtype=np.float32)
    original=p.learner.choose(features)[1].copy()
    p.select_motor_bank({'finger_contacts':{'left':True,'right':True}})
    held=p.learner.choose(features)[1].copy()
    assert p.active_motor_bank=='holding' and not np.array_equal(held,original)
    p.select_motor_bank({'finger_contacts':{'left':True,'right':False}})
    np.testing.assert_array_equal(p.learner.choose(features)[1],original)
    assert p.active_motor_bank=='pickup'


def test_selector_does_not_read_task_geometry_or_success(tmp_path):
    p=policy(tmp_path)
    p.select_motor_bank({'finger_contacts':{},'cube_pose':{'position_mm':[999,999,999]},'phase':'completed','success':True})
    assert p.active_motor_bank=='pickup'
    p.select_motor_bank({'finger_contacts':{'left':True,'right':True},'phase':'ready','success':False})
    assert p.active_motor_bank=='holding'
    with pytest.raises(ValueError,match='offline'):p.feedback(1,learn=True)


def test_holding_bank_cannot_reverse_measured_synapse_signs(tmp_path):
    p=policy(tmp_path)
    with np.load(tmp_path/'holding-model.npz') as z:values={key:z[key].copy() for key in z.files}
    values['motor_synapses'][0]*=-1
    np.savez(tmp_path/'holding-model.npz',**values)
    with pytest.raises(ValueError,match='motor bounds'):p.load_touch_banks()
