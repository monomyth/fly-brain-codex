"""Verify a higher conventional grasp before using it as neural supervision."""
import importlib,json
from pathlib import Path
from datetime import datetime
from fly_brain.simulator import EpisodeSimulator
from fly_brain.visual_dopamine.motor_experiment import task
from fly_brain.assets import home,write_json,sha256_file
import fly_brain.teacher as teachers

original_grasp_pose=teachers.grasp_pose

def raised_grasp_pose(xy,size,yaw,clearance=0):
    result=original_grasp_pose(xy,size,yaw,clearance)
    if clearance==0:result['position_mm'][2]+=6.
    return result


class RaisedTeacher(teachers.Teacher):
    version='cube_feedback_v9_raised_grasp_6mm'
    stages=('approach','descend','close','lift','hold','release','done')


if __name__=='__main__':
    project=Path(__file__).resolve().parents[1];run=project/'reports/two-view-20260910-233652'
    dataset=home()/'datasets/rebot-pick'/('raised-grasp-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
    teachers.grasp_pose=raised_grasp_pose
    collection=importlib.import_module('fly_brain.collect');collection.Teacher=RaisedTeacher
    write_json(run/'raised-grasp-plan.json',{'dataset':str(dataset),'grasp_offset_mm':6,'source_sha256':sha256_file(__file__),'purpose':'Physical conventional check before using a raised training target','positions':[[350,0],[340,-12],[360,12]]})
    for i,xy in enumerate([[350,0],[340,-12],[360,12]]):
        with EpisodeSimulator(log=run/f'raised-grasp-native-{i}.log') as client:
            result=collection.collect(client,[{**task(*xy,seed=37000+i),'lift_clearance_mm':110}],dataset,hz=5,images=True,absolute_targets=True)
        print(result,flush=True)
        if not json.loads((dataset/'collection-report.json').read_text())['episodes'][-1]['success']:
            raise RuntimeError('Raised grasp verification failed; keep the retained failure for inspection')
