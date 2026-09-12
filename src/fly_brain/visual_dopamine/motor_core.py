"""Seven anatomical muscle-pool outputs with the same PAM-gated learning rule."""
import numpy as np
from ..adapters import JOINT_LOWER,JOINT_UPPER
from .core import DopamineLearner


class MotorLearner(DopamineLearner):
    def _initialize_output(self):
        self.input_signs=np.asarray(getattr(self.circuit,"input_signs",np.ones(len(self.circuit.kc))),dtype=np.float32)
        self.mbon=np.zeros(len(self.circuit.mbon),dtype=np.float32)
        self.route={name:np.zeros(len(getattr(self.circuit,name)),dtype=np.float32) for name in ['cb','dn','pm','motor']}
        self.scores=np.zeros(7,dtype=np.float32)
        self.last_action=np.zeros(7,dtype=np.int8)
        self.deadband=.08
        self.response_mode="legacy"
        self.excitability=np.zeros(len(self.circuit.mbon),dtype=np.float32)
        self.excitability_eligibility=np.zeros_like(self.excitability)
        self.kc_tonic=.2;self.kc_gain=.4;self.balanced_tonic=False
        self.rest_current=np.zeros(len(self.circuit.mbon),dtype=np.float32)
        self.reference_mbon=np.tanh(self.base@np.full(len(self.circuit.kc),self.kc_tonic,dtype=np.float32))
        self.reference_scores,_=self.circuit.output(self.reference_mbon)
        self.rates=np.full(len(self.circuit.kc),self.kc_tonic,dtype=np.float32)
        if hasattr(self.circuit,'transfer'):
            transfer=self.circuit.gain[:,None]*self.circuit.transfer
        else:
            transfer=np.column_stack([self.circuit.output(np.eye(len(self.circuit.mbon))[i]*1e-4)[0]/1e-4 for i in range(len(self.circuit.mbon))])
        directions=np.linalg.pinv(transfer).T
        self.exploration_directions=(directions/np.maximum(np.linalg.norm(directions,axis=1,keepdims=True),1e-8)).astype(np.float32)

    def configure_balanced_tonic(self):
        self.balanced_tonic=True;self.kc_tonic=1.;self.kc_gain=1.
        self.rest_current=self.base@np.ones(len(self.circuit.kc),dtype=np.float32)
        self.reference_mbon=np.zeros(len(self.circuit.mbon),dtype=np.float32)
        self.reference_scores,_=self.circuit.output(self.reference_mbon)
        self.rates=np.ones(len(self.circuit.kc),dtype=np.float32)

    def configure_centered_response(self):
        self.response_mode="centered";self.balanced_tonic=False
        self.kc_tonic=0.;self.kc_gain=.1
        self.rest_current.fill(0)
        self.reference_mbon.fill(0)
        self.reference_scores,_=self.circuit.output(self.reference_mbon)
        self.rates.fill(0)

    def _score_gradient(self,axis):
        """Chain-rule eligibility through recorded fixed routes; an engineering approximation."""
        c=self.circuit;r=self.route
        motor_gradient=c.gain[axis]*c.readout[axis]*(1-r['motor']**2)
        joined=c.motor_from_dn_pm.T@motor_gradient
        dn_gradient=joined[:len(c.dn)].copy()
        pm_gradient=joined[len(c.dn):]*(1-r['pm']**2)
        dn_gradient+=c.pm_from_dn.T@pm_gradient
        joined=c.dn_from_mbon_cb.T@(dn_gradient*(1-r['dn']**2))
        mbon_gradient=joined[:len(c.mbon)].copy()
        mbon_gradient+=c.cb_from_mbon.T@(joined[len(c.mbon):]*(1-r['cb']**2))
        return mbon_gradient*(1-self.mbon**2)

    def choose(self,features,explore=False,exploration_axis=None):
        features=np.asarray(features,dtype=np.float32)
        self.rates=self.kc_gain*features if self.response_mode=="centered" else np.clip(self.kc_tonic+self.kc_gain*features,0,2 if self.balanced_tonic else 1)
        self.mbon=np.tanh(self.weights@(self.rates*self.input_signs)-self.rest_current+self.excitability)
        self.scores,self.route=self.circuit.output(self.mbon)
        self.scores=self.scores-self.reference_scores if np.any(np.abs(self.route['motor'])>1e-8) else np.zeros(7,dtype=np.float32)
        self.eligibility.fill(0);self.excitability_eligibility.fill(0)
        if explore:
            axis=int(self.rng.integers(7)) if exploration_axis is None else exploration_axis
            if axis not in range(7):raise ValueError('Invalid exploratory motor channel')
            gradient=self._score_gradient(axis)
            perturbation=float(self.rng.normal(0,self.noise))
            # Motor babbling is actual activity in the selected antagonistic MN pools.
            # The differential readout therefore remains the source of the action.
            pool=self.circuit.readout[axis]
            activity_delta=perturbation*pool/(self.circuit.gain[axis]*float(pool@pool))
            self.route['motor']+=activity_delta
            self.scores=self.circuit.gain*(self.circuit.readout@self.route['motor'])-self.reference_scores
            self.eligibility[:]=(perturbation/self.noise**2)*gradient[:,None]*(self.rates*self.input_signs)[None,:]*self.mask
            self.excitability_eligibility[:]=(perturbation/self.noise**2)*gradient
        self.last_action=np.where(np.abs(self.scores)<=self.deadband,2,(self.scores>=0).astype(np.int8))
        return self.last_action.copy(),self.scores.copy()

    def reinforce(self,reward,delay_s=.3,dopamine_enabled=True,plasticity_enabled=True):
        result=super().reinforce(reward,delay_s,dopamine_enabled,plasticity_enabled)
        before=self.excitability.copy()
        if self.response_mode=="centered" and plasticity_enabled:
            modulation=self.circuit.dopamine_weights@self.dopamine
            self.excitability+=self.learning_rate*modulation*self.excitability_eligibility*np.exp(-delay_s/self.eligibility_tau)
            np.clip(self.excitability,-2,2,out=self.excitability)
        self.excitability_eligibility.fill(0)
        result['excitability_change_l1']=float(np.abs(self.excitability-before).sum())
        return result

    def state_into(self,state):
        state[self.circuit.kc]=self.rates
        state[self.circuit.mbon]=self.mbon
        state[self.circuit.dan]=self.dopamine
        for name,values in self.route.items():state[getattr(self.circuit,name)]=values
        return state


def motor_action(observation,scores,axis=None,step_degrees=2.,step_mm=4.,selection='strongest'):
    """Only body state and neuron output enter this adapter; no cube or reward state."""
    if axis is not None and (isinstance(axis,bool) or axis not in range(7)):raise ValueError('Axis must be 0–6')
    if selection not in ['strongest','all']:raise ValueError('Unknown motor selection')
    if not np.isfinite([step_degrees,step_mm]).all() or not 0<step_degrees<=5 or not 0<step_mm<=10:raise ValueError('Motor steps outside response limits')
    scores=np.asarray(scores,dtype=np.float32)
    if scores.shape!=(7,) or not np.isfinite(scores).all():raise ValueError('Seven finite motor outputs required')
    before=np.r_[observation['joints_deg'],observation['gripper_mm']].astype(float)
    lower=np.r_[JOINT_LOWER,0];upper=np.r_[JOINT_UPPER,90]
    target=np.clip(before+np.sign(scores)*np.r_[[step_degrees]*6,step_mm],lower,upper)
    possible=np.abs(target-before)>1e-6
    selected=np.zeros(7,dtype=bool)
    if axis is not None:selected[axis]=True
    elif selection=='all':selected[:]=True
    elif possible.any():selected[int(np.argmax(np.where(possible,np.abs(scores),-1)))]=True
    target=np.where(selected,target,before)
    return {'joints_deg':target[:6].tolist(),'gripper_mm':float(target[6])}


def stimulation(circuit,axis,direction,amplitude=.5):
    if axis not in range(7) or direction not in [-1,1] or not 0<amplitude<=1:raise ValueError('Invalid neuron stimulation')
    motor=np.zeros(len(circuit.motor),dtype=np.float32)
    selected=circuit.readout[axis]*direction>0
    if selected.any():motor[selected]=amplitude
    else:
        # A single pool opens via a below-reference rate deviation.
        selected=circuit.readout[axis]!=0
        if not selected.any():raise ValueError('Motor channel has no mapped neurons')
        motor[selected]=direction*np.sign(circuit.readout[axis,selected])*amplitude
    scores=circuit.gain*(circuit.readout@motor)
    return scores,motor
