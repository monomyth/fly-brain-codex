"""Experimental continuous aperture from the existing trained neural grip logit."""
import numpy as np
from fly_brain.visual_dopamine.motor_policy import MotorPolicy

GRIP_LOGIT_SCALE = 2000.0  # Matches feedback_training's binary cross-entropy objective.


def neural_aperture(score):
    value=float(score)
    if not np.isfinite(value):raise ValueError('Nonfinite neural gripper output')
    return float(90.0/(1.0+np.exp(-np.clip(value*GRIP_LOGIT_SCALE,-60,60))))


class ProbabilityGripPolicy(MotorPolicy):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.output_mode!='joint_targets' or self.ramp_gripper or self.axis is not None:
            raise ValueError('This aperture experiment requires the full absolute-target policy')
        self.metadata['name']+='-continuous-grip'

    def act(self,observation,explore=False):
        action=super().act(observation,explore)
        aperture=neural_aperture(self.last_scores[6])
        difference=aperture-observation['gripper_mm']
        self.last_choices[6]=2 if abs(difference)<.01 else int(difference>=0)
        return {**action,'gripper_mm':aperture}
