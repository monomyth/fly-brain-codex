"""Conventional validation of a common approach before cube-specific descent."""
import importlib,json
from pathlib import Path
from datetime import datetime
import fly_brain.teacher as teachers
from verify_raised_grasp import raised_grasp_pose
from verify_centered_hold import CenteredTeacher
from fly_brain.simulator import EpisodeSimulator
from fly_brain.visual_dopamine.motor_experiment import task
from fly_brain.assets import home,write_json,sha256_file


class CommonApproachTeacher(CenteredTeacher):
    version='cube_feedback_v12_common_approach'
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        pose=raised_grasp_pose([350,0],20,0,160)
        self.path['approach']=self.client.call('rebot_solve_pose',pose=pose,frame='grasp',gripper_mm=90)['joints_deg']


if __name__=='__main__':
    project=Path(__file__).resolve().parents[1];run=project/'reports/two-view-20260910-233652'
    dataset=home()/'datasets/rebot-pick'/('common-approach-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
    teachers.grasp_pose=raised_grasp_pose
    collection=importlib.import_module('fly_brain.collect');collection.Teacher=CommonApproachTeacher
    write_json(run/'common-approach-plan.json',{'dataset':str(dataset),'approach_center_mm':[350,0],'positions':[[340,-12],[360,12]],'source_sha256':sha256_file(__file__),'purpose':'Conventional verification and training examples; no change to deployed actor rules'})
    for i,xy in enumerate([[340,-12],[360,12]]):
        with EpisodeSimulator(log=run/f'common-approach-native-{i}.log') as client:
            result=collection.collect(client,[{**task(*xy,seed=40000+i),'lift_clearance_mm':110}],dataset,hz=5,images=True,absolute_targets=True)
        print(result,flush=True)
        if not json.loads((dataset/'collection-report.json').read_text())['episodes'][-1]['success']:
            raise RuntimeError('Common approach verification failed; inspect retained evidence')
