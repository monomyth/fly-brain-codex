"""Experimental measured visual-to-descending branch on the existing motor policy."""
from pathlib import Path
import json
import numpy as np
from scipy import sparse
from fly_brain.assets import sha256_file

class ReflexCore:
    def __init__(self,base,indices,mean,std):self.base=base;self.indices=indices;self.mean=mean;self.std=std
    def __getattr__(self,name):return getattr(self.base,name)
    def encode_observation(self,observation,base_directory=None,visual_enabled=True):
        original=self.base.encode_observation(observation,base_directory,visual_enabled)
        if hasattr(self.base,'gpu_state'):raw=np.array(self.base.gpu_state)[self.indices,0]
        else:raw=self.base._visual_state[self.indices,0]
        extra=np.tanh((raw-self.mean)/self.std).astype(np.float32) if visual_enabled else np.zeros_like(self.mean)
        return np.r_[original,extra]

class ReflexLearner:
    def __init__(self,base,topology,multipliers):
        self.base=base;self.indices=topology['indices'];self.original_count=len(base.circuit.kc)
        if (multipliers.shape!=topology['base'].shape or not np.isfinite(multipliers).all() or np.any((multipliers<0)|(multipliers>64))):raise ValueError('Invalid reflex strengths')
        values=topology['base']*multipliers*topology['signs'][topology['pre']]
        self.matrix=sparse.coo_matrix((values,(topology['post'],topology['pre'])),shape=tuple(topology['shape'])).tocsr()
        self.extra=np.zeros(len(self.indices),dtype=np.float32)
    def __getattr__(self,name):return getattr(self.base,name)
    def choose(self,features,explore=False,exploration_axis=None):
        if explore:raise ValueError('Visual-reflex validation uses frozen weights')
        features=np.asarray(features,dtype=np.float32)
        if features.shape!=(self.original_count+len(self.indices),) or not np.isfinite(features).all():raise ValueError('Invalid reflex neuron inputs')
        self.base.choose(features[:self.original_count],False)
        self.extra=features[self.original_count:];c=self.base.circuit;r=self.base.route
        r['dn']=np.tanh(c.dn_from_mbon_cb@np.r_[self.base.mbon,r['cb']]+self.matrix@(.1*self.extra))
        r['pm']=np.tanh(c.pm_from_dn@r['dn'])
        r['motor']=np.tanh(c.motor_from_dn_pm@np.r_[r['dn'],r['pm']])
        self.base.scores=c.gain*(c.readout@r['motor'])-self.base.reference_scores
        self.base.last_action=np.where(abs(self.base.scores)<=self.base.deadband,2,(self.base.scores>=0).astype(np.int8))
        return self.base.last_action.copy(),self.base.scores.copy()
    def state_into(self,state):
        self.base.state_into(state);state[self.indices]=.1*self.extra;return state
    def reinforce(self,reward,delay_s=.3,dopamine_enabled=True,plasticity_enabled=True):
        if plasticity_enabled:raise ValueError('Visual-reflex online learning is not implemented')
        return self.base.reinforce(reward,delay_s,dopamine_enabled,False)


def attach_reflex(policy,inputs,weights):
    inputs=Path(inputs);weights=Path(weights);origin=json.loads((inputs/'manifest.json').read_text())
    if not origin['passed'] or origin['source_model_sha256']!=sha256_file(policy.checkpoint/'model.npz') or origin['graph_id']!=policy.circuit.manifest['graph_id'] or origin['motor_circuit_id']!=policy.metadata['motor_circuit_id']:raise ValueError('Reflex inputs belong to a different base circuit')
    for name in ['topology.npz','features.npz']:
        if sha256_file(inputs/name)!=origin['files'][name]:raise ValueError('Reflex input artifact changed')
    with np.load(inputs/'topology.npz',allow_pickle=False) as z:topology={k:z[k].copy() for k in z.files}
    with np.load(inputs/'features.npz',allow_pickle=False) as z:mean=z['mean'].copy();std=z['std'].copy()
    with np.load(weights,allow_pickle=False) as z:multipliers=z['multipliers'].copy()
    if not np.array_equal(topology['body_ids'],policy.circuit.ids[topology['indices']]) or not np.array_equal(topology['dn_body_ids'],policy.circuit.ids[policy.circuit.dn]):raise ValueError('Reflex neuron identity mismatch')
    policy.core=ReflexCore(policy.core,topology['indices'],mean,std);policy.learner=ReflexLearner(policy.learner,topology,multipliers)
    policy.metadata={**policy.metadata,'name':'visual-reflex-20260911'}
    policy.reflex_runtime={'weights':str(weights.resolve()),'weights_sha256':sha256_file(weights),'inputs':str(inputs.resolve()),'manifest_sha256':sha256_file(inputs/'manifest.json'),'source_sha256':sha256_file(__file__),'input_neurons':len(topology['indices']),'measured_synapses':len(multipliers)}
    return policy
