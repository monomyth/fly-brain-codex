"""Camera-to-muscle-pool policy; task geometry is reserved for the evaluator."""
import json
from pathlib import Path
import numpy as np
from ..assets import home,checked_files,local_artifact
from .core import VisualCore
from .motor_circuit import MotorCircuit
from .motor_core import MotorLearner,motor_action
from .motor_training import save_checkpoint


class MotorPolicy:
    def __init__(self,checkpoint,root=None,visual_enabled=True,axis=None,selection='all',step_degrees=2.,step_mm=4.):
        self.root=home(root);self.checkpoint=local_artifact(checkpoint).resolve()
        self.metadata=json.loads((self.checkpoint/'manifest.json').read_text())
        if self.metadata.get('schema')!='seven-motor-checkpoint-v1':raise ValueError('Not a seven-motor checkpoint')
        checked_files(self.checkpoint,self.metadata)
        for name in ['graph_id','circuit_id','motor_circuit_id']+(['feedback_circuit_id'] if 'feedback_circuit_id' in self.metadata else []):
            identity=self.metadata.get(name,'')
            if len(identity)!=64 or any(c not in '0123456789abcdef' for c in identity):raise ValueError('Invalid circuit identity')
        if self.metadata.get('feedback_circuit_id'):
            from .feedback import FeedbackCircuit,FeedbackCore
            self.circuit=FeedbackCircuit(self.root/'feedback-circuits'/self.metadata['feedback_circuit_id'],self.root)
            core_type=FeedbackCore
            if self.metadata.get('temporal_sensory'):
                from .temporal import TemporalFeedbackCore
                core_type=lambda circuit,updates:TemporalFeedbackCore(circuit,updates,self.metadata['temporal_sensory'])
            if self.circuit.motor_manifest['motor_circuit_id']!=self.metadata['motor_circuit_id']:raise ValueError('Feedback motor circuit mismatch')
        else:
            self.circuit=MotorCircuit(self.root/'motor-circuits'/self.metadata['motor_circuit_id'],self.root)
            core_type=VisualCore
        if self.circuit.manifest['graph_id']!=self.metadata['graph_id'] or self.circuit.manifest['circuit_id']!=self.metadata['circuit_id']:raise ValueError('Checkpoint circuit mismatch')
        self.core=core_type(self.circuit,self.metadata['architecture']['updates'])
        schema=self.metadata.get('observation_schema',{})
        from ..cameras import LEGACY_CAMERAS, LEGACY_RIG
        self.core.encoder.configure_cameras(schema.get('camera_names',LEGACY_CAMERAS),schema.get('camera_rig_revision',LEGACY_RIG))
        self.core.encoder.configure(schema.get('retinal_profile'))
        rule=self.metadata['learning_rule']
        self.learner=MotorLearner(self.circuit,seed=self.metadata['seed'],learning_rate=rule['learning_rate'],noise=rule['exploration_sigma'],eligibility_tau=rule['eligibility_tau_seconds'])
        self.learner.weight_multiple=float(rule.get('weight_multiple',4.))
        if not np.isfinite(self.learner.weight_multiple) or not 1<=self.learner.weight_multiple<=256:raise ValueError('Invalid synaptic gain limit')
        if rule.get('response_mode')=='centered':self.learner.configure_centered_response()
        if rule.get('balanced_tonic',False):self.learner.configure_balanced_tonic()
        if rule.get('kc_tonic')!=self.learner.kc_tonic or rule.get('kc_gain')!=self.learner.kc_gain:raise ValueError('Unsupported motor operating point')
        self.learner.deadband=float(rule['deadband'])
        if not 0<=self.learner.deadband<=1:raise ValueError('Invalid no-command threshold')
        with np.load(self.checkpoint/'model.npz',allow_pickle=False) as data:
            for name in ['retinal_mean','retinal_std','kc_mean','kc_std']:
                value=data[name]
                if value.shape!=getattr(self.core,name).shape or not np.isfinite(value).all():raise ValueError('Invalid visual calibration')
                if name.endswith('_std') and (value<=0).any():raise ValueError('Invalid visual normalization')
                setattr(self.core,name,value.copy())
            weights=data['weights']
            if weights.shape!=self.learner.weights.shape or not np.isfinite(weights).all() or (weights<0).any() or (weights>self.learner.weight_multiple*self.learner.base+1e-6).any() or (weights[~self.learner.mask]!=0).any():raise ValueError('Invalid anatomical weights')
            for name in ['readout','gain']:
                if not np.array_equal(data[name],getattr(self.circuit,name)):raise ValueError('Checkpoint changed the fixed muscle-pool adapter')
            if 'excitability' in data:
                value=data['excitability']
                if value.shape!=self.learner.excitability.shape or not np.isfinite(value).all() or (np.abs(value)>2).any():raise ValueError('Invalid neuronal response offsets')
                self.learner.excitability[:]=value
            if self.metadata.get('motor_synapse_plasticity'):
                value=data['motor_synapses'];base=self.circuit.motor_from_dn_pm.data
                if value.shape!=base.shape or not np.isfinite(value).all() or (value*base<0).any() or (np.abs(value)>16*np.abs(base)+1e-7).any():raise ValueError('Invalid anatomical motor synapses')
                self.circuit.motor_from_dn_pm.data=value.copy()
            self.learner.weights[:]=weights;self.learner.baseline=float(data['baseline'])
            if not np.isfinite(self.learner.baseline) or not 0<=self.learner.baseline<=1:raise ValueError('Invalid reward expectation')
        if 'rng_state' in self.metadata:self.learner.rng.bit_generator.state=self.metadata['rng_state']
        self.axis=axis;self.selection=selection;self.step_degrees=step_degrees;self.step_mm=step_mm
        motor_action({'joints_deg':[0]*6,'gripper_mm':0},np.zeros(7),axis,step_degrees,step_mm,selection)
        self.output_mode=self.metadata['architecture'].get('output_mode','directions')
        if self.output_mode not in ['directions','joint_targets','joint_deltas']:raise ValueError('Unsupported motor output mode')
        self.joint_gain=float(self.metadata['architecture'].get('joint_gain',1.))
        self.active_joints=np.asarray(self.metadata['architecture'].get('active_joints',[True]*6),dtype=bool)
        self.command_deadband=np.asarray(self.metadata['architecture'].get('command_deadband_deg',[0.]*6),dtype=float)
        self.command_scale=float(self.metadata['architecture'].get('command_scale',1.))
        self.ramp_gripper=bool(self.metadata['architecture'].get('ramp_gripper',False))
        self.position_servo=bool(self.metadata['architecture'].get('position_servo',False))
        if self.active_joints.shape!=(6,) or self.command_deadband.shape!=(6,) or not np.isfinite(self.command_deadband).all() or (self.command_deadband<0).any() or not 0<self.command_scale<=1:raise ValueError('Invalid motor calibration')
        if not np.isfinite(self.joint_gain) or not 0<self.joint_gain<=5000:raise ValueError('Invalid joint readout gain')
        self.visual_enabled=visual_enabled;self.mode='vision';self.hz=float(self.metadata['architecture'].get('control_hz',1.));self.state=np.zeros((1,len(self.circuit.ids),1),dtype=np.float32)
        if not np.isfinite(self.hz) or not 1<=self.hz<=30:raise ValueError('Invalid motor control frequency')
        self.episode=None;self.last_frame=None;self.last_scores=np.zeros(7);self.last_choices=np.full(7,2)
        self.load_touch_banks()

    def load_touch_banks(self):
        """Load two constrained synaptic responses; the sensory circuit is shared."""
        self._motor_banks = None
        self.active_motor_bank = 'single'
        config = self.metadata.get('touch_gated_hold')
        if config is None:
            return
        if config.get('version') != 1 or config.get('gate') != 'bilateral_finger_contact' or config.get('file') != 'holding-model.npz':
            raise ValueError('Unsupported touch-gated controller')
        if self.output_mode != 'joint_targets' or not self.metadata.get('motor_synapse_plasticity'):
            raise ValueError('Touch gating requires absolute targets and mapped motor synapses')
        pickup = {'weights': self.learner.weights.copy(), 'excitability': self.learner.excitability.copy(),
                  'motor_synapses': self.circuit.motor_from_dn_pm.data.copy()}
        with np.load(self.checkpoint / config['file'], allow_pickle=False) as saved:
            holding = {key: saved[key].copy() for key in pickup}
        for key, value in holding.items():
            if value.shape != pickup[key].shape or not np.isfinite(value).all():
                raise ValueError('Invalid holding-bank parameter: ' + key)
        weights = holding['weights']; motor = holding['motor_synapses']
        if (weights < 0).any() or (weights > self.learner.weight_multiple * self.learner.base + 1e-6).any() or (weights[~self.learner.mask] != 0).any():
            raise ValueError('Holding bank changed anatomical input bounds')
        if (abs(holding['excitability']) > 2).any():
            raise ValueError('Invalid holding neuronal offsets')
        base = self.circuit.motor_synapse_base.data
        if (motor * base < 0).any() or (abs(motor) > 16 * abs(base) + 1e-7).any():
            raise ValueError('Holding bank changed anatomical motor bounds')
        self._motor_banks = {'pickup': pickup, 'holding': holding}
        self.active_motor_bank = 'pickup'

    def select_motor_bank(self, observation):
        if not getattr(self, '_motor_banks', None):
            return
        contacts = observation.get('finger_contacts', {})
        mode = 'holding' if len(contacts) == 2 and all(contacts.values()) else 'pickup'
        if mode == self.active_motor_bank:
            return
        bank = self._motor_banks[mode]
        self.learner.weights[:] = bank['weights']
        self.learner.excitability[:] = bank['excitability']
        self.circuit.motor_from_dn_pm.data[:] = bank['motor_synapses']
        self.active_motor_bank = mode

    def reset(self,observation):
        self.select_motor_bank(observation)
        self.episode=observation['episode_id'];self.last_frame=None
        if hasattr(self.core,'reset'):self.core.reset()
        else:self.core.state.fill(0)
        self.state.fill(0);self.learner.eligibility.fill(0);self.learner.excitability_eligibility.fill(0);self.learner.dopamine.fill(0)
        self.learner.last_reward=0.;self.learner.last_update_l1=0.

    def act(self,observation,explore=False):
        if self.episode!=observation['episode_id']:raise ValueError('Episode changed without reset')
        if self.last_frame is not None and observation['frame_id']<=self.last_frame:raise ValueError('Camera frame did not advance')
        self.select_motor_bank(observation)
        self.last_frame=observation['frame_id']
        if self.metadata.get('feedback_circuit_id'):
            features=self.core.encode_observation(observation,visual_enabled=self.visual_enabled)
        else:
            retinal=self.core.encoder.sample(observation['images']);features=self.core.encode(retinal,self.visual_enabled)
        self.last_exploration_axis=(self.axis if self.axis is not None else int(self.learner.rng.integers(7))) if explore else None
        self.last_choices,self.last_scores=self.learner.choose(features,explore,exploration_axis=self.last_exploration_axis)
        self.learner.state_into(self.core.state);self.state[0,:,0]=self.core.state
        if self.output_mode in ['joint_targets','joint_deltas']:
            from ..adapters import JOINT_LOWER,JOINT_UPPER
            values=self.last_scores[:6]*self.joint_gain
            if self.output_mode=='joint_deltas':
                values=np.where(self.active_joints & (np.abs(values)>=self.command_deadband),values,0.)*self.command_scale
                values=values*min(1.,(30./self.hz)/max(float(np.max(np.abs(values))),1e-9))
                values=np.asarray(observation['joints_deg'])+values
            if self.output_mode=='joint_targets' and self.position_servo:
                delta=np.clip(values,JOINT_LOWER,JOINT_UPPER)-np.asarray(observation['joints_deg'])
                delta*=min(1.,(30./self.hz)/max(float(np.max(np.abs(delta))),1e-9))*self.command_scale
                values=np.asarray(observation['joints_deg'])+delta
            joints=np.clip(values,JOINT_LOWER,JOINT_UPPER)
            grip=90. if self.last_scores[6]>=0 else 0.
            if self.ramp_gripper:
                grip=observation['gripper_mm']+float(np.clip(grip-observation['gripper_mm'],-25/self.hz*self.command_scale,25/self.hz*self.command_scale))
            if self.axis is not None:
                joints=np.array([joints[i] if i==self.axis else observation['joints_deg'][i] for i in range(6)])
                if self.axis!=6:grip=observation['gripper_mm']
            differences=np.r_[joints-np.asarray(observation['joints_deg']),grip-observation['gripper_mm']]
            self.last_choices=np.where(np.abs(differences)<.01,2,(differences>=0).astype(int))
            return {'joints_deg':joints.tolist(),'gripper_mm':float(grip)}
        drive=np.where(self.last_choices==2,0,self.last_scores)
        return motor_action(observation,drive,self.axis,self.step_degrees,self.step_mm,self.selection)

    def feedback(self,reward,delay_s=.3,learn=False):
        if learn and self.metadata.get('touch_gated_hold'):raise ValueError('Touch-gated banks are trained offline; disable online learning')
        result=self.learner.reinforce(reward,delay_s,self.metadata.get('dopamine_enabled',True),learn)
        self.learner.state_into(self.core.state);self.state[0,:,0]=self.core.state
        return result

    def activity_metadata(self):
        return {'dopamine':{'signal':float(self.learner.dopamine.mean()),'reward':self.learner.last_reward,'cells':len(self.circuit.dan),'weight_change_l1':self.learner.last_update_l1},
                'motor':{'response_bank':getattr(self,'active_motor_bank','single'),'scores':self.last_scores.tolist(),'choices':self.last_choices.tolist(),'motor_circuit_id':self.metadata['motor_circuit_id']}}

    def save(self,destination,training_report):
        if self.metadata.get('touch_gated_hold'):raise ValueError('Export both frozen banks using the touch-gated checkpoint builder')
        metadata={k:v for k,v in self.metadata.items() if k not in ['files','test_result','rng_state','package_source_hash','name']}
        metadata.update(parent_checkpoint=str(self.checkpoint),online_training_report=training_report,closed_loop_validated=False)
        save_checkpoint(destination,self.core,self.learner,metadata)
        return str(Path(destination).resolve())
