"""Conventional training-only height excursions teach recovery around a held cube.

Executed excursions are recorded separately from the correct expert targets.
They are never used by autonomous policy inference.
"""
import argparse
from pathlib import Path
from datetime import datetime
import importlib
from fly_brain.teacher import Teacher
from fly_brain.simulator import EpisodeSimulator
from fly_brain.visual_dopamine.motor_experiment import task
from fly_brain.assets import home

class HoldRecoveryTeacher(Teacher):
    version='cube_feedback_v3_hold_recovery_curriculum'
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.curriculum_start=None;self.last_intervention=None
    def act(self,observation,evaluation):
        correct,label=super().act(observation,evaluation);self.last_intervention=None
        if self.stage=='hold':
            if self.curriculum_start is None:self.curriculum_start=observation['simulation_time']
            index=int((observation['simulation_time']-self.curriculum_start)/1.2)
            heights=[85.,140.,90.,135.,95.]
            if index<len(heights):
                goal=self._feedback_lift_target(observation,heights[index])
                exploratory=self.adapter.label(observation,goal,0)
                self.last_intervention={'reason':'conventional held-height recovery curriculum','executed_clearance_mm':heights[index],'expert_clearance_mm':self.task['lift_clearance_mm']+15}
                return self.adapter.decode(observation,exploratory[:6],exploratory[6]),label
        return correct,label

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--episodes',type=int,default=4);p.add_argument('--output',type=Path);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];reports=root/'reports/pickup-fix';data=a.output or home()/'datasets/rebot-pick'/('hold-recovery-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
    (reports/'hold-recovery-dataset-path.txt').write_text(str(data)+'\n')
    collection=importlib.import_module('fly_brain.collect');collection.Teacher=HoldRecoveryTeacher
    for seed in range(a.episodes):
        with EpisodeSimulator(log=reports/f'hold-recovery-native-{seed}.log') as client:
            print(collection.collect(client,[task(350,0,15200+seed)],data,hz=5,images=True,seed=seed),flush=True)
