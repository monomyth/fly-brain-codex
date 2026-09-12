"""Start folded, perturb the arm, then finish a verified conventional pickup.

Correct goal labels are retained during each recorded exploration prefix. This
breaks the accidental correlation between current arm pose and desired cube pose.
"""
import importlib
from pathlib import Path
from datetime import datetime
from collect_direct_pickups import DirectTeacher
from fly_brain.simulator import EpisodeSimulator
from fly_brain.visual_dopamine.motor_experiment import task
from fly_brain.assets import home

class GroundingTeacher(DirectTeacher):
    version='cube_feedback_v6_visual_grounding'
    perturbation=[6,-6,-8,2,3,-4]
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.prefix_start=None
    def act(self,observation,evaluation):
        result=super().act(observation,evaluation)
        if self.prefix_start is None:self.prefix_start=observation['simulation_time']
        if observation['simulation_time']-self.prefix_start<1.8:
            self.last_execution_target={'joints_deg':list(self.perturbation),'gripper_mm':0.}
            self.last_intervention={'reason':'conventional initial pose exploration; expert targets remain unchanged','exploration_pose':self.last_execution_target}
        return result

if __name__=='__main__':
    project=Path(__file__).resolve().parents[1];reports=project/'reports/pickup-fix'
    data=home()/'datasets/rebot-pick'/('visual-grounding-'+datetime.now().strftime('%Y%m%d-%H%M%S'));(reports/'grounding-dataset-path.txt').write_text(str(data)+'\n')
    collection=importlib.import_module('fly_brain.collect');collection.Teacher=GroundingTeacher
    count=0
    for xy in [[350,0],[340,-12],[340,12],[360,-12],[360,12]]:
        for direction in [-1,1]:
            GroundingTeacher.perturbation=[direction*6,-6,-8,2,direction*3,-direction*4]
            with EpisodeSimulator(log=reports/f'grounding-reference-{count}.log') as client:
                print(collection.collect(client,[task(*xy,seed=16200+count)],data,hz=5,images=True,seed=count,absolute_targets=True),flush=True)
            count+=1
