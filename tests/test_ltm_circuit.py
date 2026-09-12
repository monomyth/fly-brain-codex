from copy import deepcopy
import numpy as np
import pytest
from scipy import sparse
from test_motor_control import circuit
from fly_brain.visual_dopamine.ltm_circuit import validate_transfer,transferred_strengths
from fly_brain.visual_dopamine.motor_core import stimulation
from fly_brain.visual_dopamine.motor_circuit import MotorCircuit


def pair():
    a=circuit();a.ids=np.arange(80);a.sensory=sparse.block_diag((a.sensory,sparse.csr_matrix((22,22))),format='csr')
    a.visual_kc=a.kc[:5];a.ascending=a.kc[5:];a.body_indices=np.array([60,61]);a.sensory_groups=[];a.input_signs=np.ones(7)
    a.manifest={'graph_id':'a'*64};a.motor_manifest={'motor_circuit_id':'b'*64};a.motor_synapse_base=a.motor_from_dn_pm.copy()
    b=deepcopy(a);b.motor=np.r_[a.motor[:12],[70,71]];b.readout[6]=0;b.readout[6,-2:]=-.5
    value=b.motor_from_dn_pm.tolil();value[12:]=0;value[12,6]=.2;value[13,13]=.2;b.motor_from_dn_pm=value.tocsr();b.motor_synapse_base=b.motor_from_dn_pm.copy()
    a.motor_from_dn_pm.data*=.7
    return a,b


def test_ltm_transfer_preserves_joint_outputs_and_initializes_new_edges_from_measurements():
    a,b=pair();assert validate_transfer(a,b)['arm_pathways_identical']
    strength=transferred_strengths(a,b);base=b.motor_synapse_base.tocoo()
    assert np.allclose(strength[base.row>=12],.2)
    b.motor_from_dn_pm.data=(strength*np.sign(b.motor_from_dn_pm.data)).astype(np.float32)
    before,_=MotorCircuit.output(a,np.linspace(-.3,.5,7));after,_=MotorCircuit.output(b,np.linspace(-.3,.5,7))
    assert np.allclose(before[:6],after[:6],atol=1e-7)
    assert before[6]!=after[6]


def test_transfer_rejects_changes_to_sensory_or_measured_arm_paths():
    a,b=pair();b.sensory.data[0]*=.5
    with pytest.raises(ValueError,match='sensory'):validate_transfer(a,b)
    a,b=pair();b.motor_synapse_base.data[0]*=-1
    with pytest.raises(ValueError,match='arm motor edges'):validate_transfer(a,b)


def test_single_ltm_pool_uses_signed_rate_deviations_for_both_commands():
    _,c=pair()
    opening,rates=stimulation(c,6,1)
    closing,_=stimulation(c,6,-1)
    assert opening[6]>0 and closing[6]<0
    assert not np.any(opening[:6]) and not np.any(closing[:6])
    assert np.all(rates[-2:]<0)
