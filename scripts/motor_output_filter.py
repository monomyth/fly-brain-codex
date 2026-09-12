"""Causal actuator filtering of neural commands; no task/cube geometry or IK."""
import math
import numpy as np

DEFAULTS={'joint_tau_seconds':.6,'close_commit_seconds':4.,'open_dwell_seconds':1.}

class MotorOutputFilter:
    def __init__(self,**parameters):
        self.parameters={**DEFAULTS,**parameters}
        if set(self.parameters)!=set(DEFAULTS) or not all(math.isfinite(v) and 0<v<=5 for v in self.parameters.values()):raise ValueError('Invalid motor-filter parameters')
        self.reset()
    def reset(self):
        self.joints=None;self.last_time=None;self.closed=False;self.close_until=None;self.open_since=None
    def apply(self,action,simulation_time):
        now=float(simulation_time)
        q=np.asarray(action['joints_deg'],dtype=float);grip=float(action['gripper_mm'])
        if q.shape!=(6,) or not np.isfinite(q).all() or not math.isfinite(grip) or not 0<=grip<=90 or not math.isfinite(now):raise ValueError('Invalid neural command')
        if self.last_time is not None and now<=self.last_time:raise ValueError('Motor filter requires advancing observation time')
        if self.joints is None:self.joints=q.copy()
        else:
            alpha=-math.expm1(-(now-self.last_time)/self.parameters['joint_tau_seconds'])
            self.joints+=alpha*(q-self.joints)
        self.last_time=now
        if not self.closed and grip<=9:
            self.closed=True;self.close_until=now+self.parameters['close_commit_seconds'];self.open_since=None
        elif self.closed and now>=self.close_until:
            if grip>=81:
                if self.open_since is None:self.open_since=now
                if now-self.open_since>=self.parameters['open_dwell_seconds']:
                    self.closed=False;self.open_since=None
            else:self.open_since=None
        return {'joints_deg':self.joints.tolist(),'gripper_mm':0. if self.closed else grip}


def attach_filter(policy,parameters=None):
    """Wrap only actuator output; retain the actual brain dynamics and activity."""
    stage=MotorOutputFilter(**(parameters or {}));original_act=policy.act;original_reset=policy.reset
    def reset(observation):original_reset(observation);stage.reset()
    def act(observation,explore=False):
        if explore:raise ValueError('The motor-filter experiment uses frozen weights')
        raw=original_act(observation,explore=False)
        policy.raw_neural_action=raw
        return stage.apply(raw,observation['simulation_time'])
    policy.reset=reset;policy.act=act;policy.motor_output_filter=stage
    return policy
