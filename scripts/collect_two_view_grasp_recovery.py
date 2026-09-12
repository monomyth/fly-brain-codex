"""Successful teacher pickups with measured near-grasp pose variation.

Only execution targets are perturbed; the recorded expert target remains the
correct pickup command. No perturbation or teacher is added to the live actor.
"""
import argparse
import importlib
import json
from datetime import datetime
from pathlib import Path
import numpy as np
from fly_brain.teacher import Teacher, grasp_pose
from fly_brain.simulator import EpisodeSimulator
from fly_brain.visual_dopamine.motor_experiment import task
from fly_brain.assets import home, write_json, sha256_file


class GraspRecoveryTeacher(Teacher):
    version='cube_feedback_v8_grasp_recovery'
    stages=('approach','descend','close','lift','hold','release','done')

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.recovery_start=None
        self.recovery_poses=[]
        self.last_intervention=None
        for dx,dy,dz in [(0,0,18),(2,0,10),(-2,0,10),(0,2,10),(0,-2,10),(0,0,6)]:
            xy=np.asarray(self.task['cube_xy_mm'])+[dx,dy]
            pose=grasp_pose(xy,self.task['cube_size_mm'],self.task['cube_yaw_deg'],dz)
            q=self.client.call('rebot_solve_pose',pose=pose,frame='grasp',gripper_mm=90)['joints_deg']
            self.recovery_poses.append(q)

    def act(self,observation,evaluation):
        result=super().act(observation,evaluation)
        self.last_intervention=None
        if self.stage=='descend' and not any(observation['finger_contacts'].values()):
            if self.recovery_start is None and observation['grasp_pose']['position_mm'][2]<70:
                self.recovery_start=observation['simulation_time']
            if self.recovery_start is not None:
                index=int((observation['simulation_time']-self.recovery_start)/.8)
                if index<len(self.recovery_poses):
                    self.last_execution_target={'joints_deg':self.recovery_poses[index],'gripper_mm':90.}
                    self.last_intervention={'reason':'conventional near-grasp exploration; expert labels unchanged',
                                            'recovery_pose_index':index,'executed_target':self.last_execution_target}
        return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',type=Path)
    parser.add_argument('--limit',type=int,default=7)
    args=parser.parse_args()
    project=Path(__file__).resolve().parents[1]
    run=project/'reports/two-view-20260910-233652'
    configs=json.loads((run/'training-configurations.json').read_text())
    positions=(configs['train']+configs['validation'])[:args.limit]
    dataset=args.dataset or home()/'datasets/rebot-pick'/('two-view-grasp-recovery-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
    write_json(run/'grasp-recovery-plan.json',{'dataset':str(dataset),'positions':positions,'native_demo_minimum_clearance_mm':110,
        'teacher_target_clearance_mm':125,'source_sha256':sha256_file(__file__),'purpose':'Near-grasp coverage, teacher-assisted execution, never autonomous results'})
    collection=importlib.import_module('fly_brain.collect')
    collection.Teacher=GraspRecoveryTeacher
    for i,xy in enumerate(positions):
        file=dataset/'collection-report.json'
        previous=json.loads(file.read_text())['episodes'] if file.exists() else []
        if any(r['success'] and r['requested_task']['cube_xy_mm']==xy for r in previous):continue
        with EpisodeSimulator(log=run/f'grasp-recovery-native-{i}-attempt-{len(previous)}.log') as client:
            result=collection.collect(client,[{**task(*xy,seed=36000+i),'lift_clearance_mm':110}],dataset,hz=5,images=True,seed=300+i,absolute_targets=True)
        print({'xy':xy,**result},flush=True)
        if not json.loads(file.read_text())['episodes'][-1]['success']:
            raise RuntimeError('Reference pickup failed; inspect retained evidence before continuing')


if __name__=='__main__':main()
