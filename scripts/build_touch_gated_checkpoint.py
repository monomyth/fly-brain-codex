"""Package constrained pickup/holding weights with an explicit tactile selector."""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
from fly_brain.assets import checked_files, sha256_file, artifact_manifest, write_json
from fly_brain.visual_dopamine.motor_policy import MotorPolicy
from fly_brain.visual_dopamine.motor_training import save_checkpoint

p=argparse.ArgumentParser(description=__doc__)
for name in ('pickup','holding','inputs','output'):p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();root=Path(__file__).resolve().parents[1]
if a.output.exists() or not a.output.resolve().is_relative_to(root/'data/checkpoints'):
    raise ValueError('Use a new project-local checkpoint')
pickup=MotorPolicy(a.pickup,root/'data');holding=MotorPolicy(a.holding,root/'data')
inputs=json.loads((a.inputs/'manifest.json').read_text());checked_files(a.inputs,inputs)
for meta in (holding.metadata,inputs):
    for key in ('graph_id','circuit_id','motor_circuit_id','feedback_circuit_id','architecture','observation_schema','temporal_sensory','learning_rule'):
        if meta.get(key)!=pickup.metadata.get(key):raise ValueError('Response banks have different contracts: '+key)
for path in (a.holding,a.inputs):
    with np.load(a.pickup/'model.npz') as before,np.load(path/'model.npz') as after:
        for key in ('retinal_mean','retinal_std','kc_mean','kc_std','readout','gain'):
            if not np.array_equal(before[key],after[key]):raise ValueError('Response banks changed fixed calibration or mapping: '+key)
if pickup.metadata.get('touch_gated_hold') or holding.metadata.get('touch_gated_hold'):
    raise ValueError('Nested touch-gated checkpoints are unsupported')
meta=dict(inputs)
for key in ('metrics','selection_statistics','stage_metrics','initial_pose_metrics','continuation','training_loss_recipe'):
    meta.pop(key,None)
meta.update(preparation_only=False,closed_loop_validated=False,
    task_scope='Two learned MaleCNS motor responses selected by bilateral finger contact; pickup/hold success requires native verification',
    training_method='Separately fitted constrained neural responses sharing fixed sensory calibration, connectome topology and the seven-channel muscle-pool readout',
    touch_gated_hold={'version':1,'gate':'bilateral_finger_contact','file':'holding-model.npz',
        'pickup_checkpoint':str(a.pickup.resolve()),'pickup_model_sha256':sha256_file(a.pickup/'model.npz'),
        'holding_checkpoint':str(a.holding.resolve()),'holding_model_sha256':sha256_file(a.holding/'model.npz'),
        'online_learning_supported':False,'note':'Touch selection is an engineered controller rule. Both responses generate all seven motor outputs through the measured neural route.'},
    response_training={'pickup':pickup.metadata.get('continuation'), 'holding':holding.metadata.get('continuation')})
save_checkpoint(a.output,pickup.core,pickup.learner,meta)
with np.load(a.holding/'model.npz') as saved:
    np.savez(a.output/'holding-model.npz',**{name:saved[name] for name in ('weights','excitability','motor_synapses')})
for name in ('training-features.npz','training-targets.npz','feature-rows.json','parameter-scaling.npy'):
    shutil.copy2(a.inputs/name,a.output/name)
m=json.loads((a.output/'manifest.json').read_text());m.pop('files',None);artifact_manifest(a.output,m)
if sha256_file(a.output/'model.npz')!=sha256_file(a.pickup/'model.npz'):
    raise ValueError('Packaging changed the frozen pickup weights')
verified=MotorPolicy(a.output,root/'data')
verified.select_motor_bank({'finger_contacts':{'left':True,'right':True}})
for key,expected in verified._motor_banks['holding'].items():
    actual=verified.circuit.motor_from_dn_pm.data if key=='motor_synapses' else getattr(verified.learner,key)
    np.testing.assert_array_equal(actual,expected)
verified.select_motor_bank({'finger_contacts':{'left':False,'right':False}})
write_json(a.output.parent/(a.output.name+'-packaging.json'),{'checkpoint':str(a.output.resolve()),'pickup_weights_exactly_preserved':True,'holding_bounds_checked':True,'gate':'bilateral_finger_contact','passed':True})
print(a.output.resolve())
