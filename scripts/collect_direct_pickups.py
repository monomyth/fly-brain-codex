"""Direct-goal conventional demonstrations using the same native drive as the actor."""
import argparse,importlib,json
from pathlib import Path
from datetime import datetime
from fly_brain.teacher import Teacher
from fly_brain.simulator import EpisodeSimulator
from fly_brain.visual_dopamine.motor_experiment import task
from fly_brain.assets import home,write_json

class DirectTeacher(Teacher):
    version='cube_feedback_v5_direct_goals'
    stages=('approach','descend','close','lift','hold','release','done')
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.curriculum_start=None;self.last_intervention=None
    def act(self,observation,evaluation):
        action,label=super().act(observation,evaluation);self.last_intervention=None
        if self.stage=='hold':
            if self.curriculum_start is None:self.curriculum_start=observation['simulation_time']
            index=int((observation['simulation_time']-self.curriculum_start)/1.2)
            heights=[85.,140.,90.,135.,95.]
            if index<len(heights):
                q=self._feedback_lift_target(observation,heights[index])
                self.last_execution_target={'joints_deg':q,'gripper_mm':0.}
                self.last_intervention={'reason':'conventional hold-height recovery curriculum','executed_clearance_mm':heights[index],'expert_clearance_mm':115.}
        return action,label

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int,default=8);args=parser.parse_args()
    project=Path(__file__).resolve().parents[1];reports=project/'reports/pickup-fix'
    data=home()/'datasets/rebot-pick'/('direct-goals-'+datetime.now().strftime('%Y%m%d-%H%M%S'));(reports/'direct-dataset-path.txt').write_text(str(data)+'\n')
    configurations={'train':[[350,0],[340,-12],[340,12],[360,-12],[360,12]],'validation':[[350,8]],'test':[[345,-4],[355,4]]}
    write_json(reports/'direct-configurations.json',configurations)
    collection=importlib.import_module('fly_brain.collect');collection.Teacher=DirectTeacher
    positions=[p for group in configurations.values() for p in group]
    for i,xy in enumerate(positions[:args.limit]):
        with EpisodeSimulator(log=reports/f'direct-reference-native-{i}.log') as client:
            result=collection.collect(client,[task(*xy,seed=15700+i)],data,hz=5,images=True,seed=i,absolute_targets=True)
            print({'xy_mm':xy,**result},flush=True)
            if result['successes']!=i+1:raise RuntimeError('Direct-goal reference failed; inspect before continuing')
