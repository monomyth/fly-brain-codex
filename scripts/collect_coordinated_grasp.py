"""Verified demonstrations covering partial closure and horizontal hold recovery."""
import importlib,json
from pathlib import Path
from datetime import datetime
import numpy as np
import fly_brain.teacher as teachers
from verify_raised_grasp import raised_grasp_pose
from verify_centered_hold import CenteredTeacher
from fly_brain.simulator import EpisodeSimulator
from fly_brain.visual_dopamine.motor_experiment import task
from fly_brain.assets import home,write_json,sha256_file


class CoordinatedTeacher(CenteredTeacher):
    version='cube_feedback_v11_coordinated_grasp'
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.closure_start=None;self.hold_start=None;self.closure_poses=[];self.last_intervention=None
        for dx,dy,dz,aperture in [(0,0,8,80),(2,0,4,60),(-2,0,4,40),(0,2,4,70),(0,-2,4,40),(0,0,2,30)]:
            pose=raised_grasp_pose(np.asarray(self.task['cube_xy_mm'])+[dx,dy],20,0)
            pose['position_mm'][2]+=dz
            q=self.client.call('rebot_solve_pose',pose=pose,frame='grasp',gripper_mm=aperture)['joints_deg']
            self.closure_poses.append({'joints_deg':q,'gripper_mm':aperture})

    def act(self,observation,evaluation):
        result=super().act(observation,evaluation);self.last_intervention=None
        if self.stage=='descend' and not any(observation['finger_contacts'].values()):
            if self.closure_start is None and observation['grasp_pose']['position_mm'][2]<70:self.closure_start=observation['simulation_time']
            if self.closure_start is not None:
                i=int(observation['simulation_time']-self.closure_start)
                if i<len(self.closure_poses):
                    self.last_execution_target=self.closure_poses[i]
                    self.last_intervention={'reason':'conventional partial-aperture and pose exploration; expert target remains correct','index':i,'execution_target':self.last_execution_target}
        if self.stage=='hold':
            if self.hold_start is None:self.hold_start=observation['simulation_time']
            i=int(observation['simulation_time']-self.hold_start)
            offsets=[[-10,0],[10,0],[0,-10],[0,10]]
            if i<len(offsets):
                pose={**self.last_hold_pose,'position_mm':list(self.last_hold_pose['position_mm'])}
                pose['position_mm'][:2]=(np.asarray([350,0])+offsets[i]).tolist()
                q=self.client.call('rebot_solve_pose',pose=pose,frame='grasp',gripper_mm=observation['gripper_mm'])['joints_deg']
                self.last_execution_target={'joints_deg':q,'gripper_mm':0.}
                self.last_intervention={'reason':'conventional horizontal hold recovery; expert target remains centered','offset_mm':offsets[i]}
        return result


if __name__=='__main__':
    project=Path(__file__).resolve().parents[1];run=project/'reports/two-view-20260910-233652'
    dataset=home()/'datasets/rebot-pick'/('coordinated-grasp-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
    teachers.grasp_pose=raised_grasp_pose
    collection=importlib.import_module('fly_brain.collect');collection.Teacher=CoordinatedTeacher
    positions=[[350,0],[340,-12],[360,12],[345,8]]
    write_json(run/'coordinated-collection-plan.json',{'dataset':str(dataset),'positions':positions,'source_sha256':sha256_file(__file__),'purpose':'Conventional successful recovery demonstrations, not autonomous evaluation'})
    for i,xy in enumerate(positions):
        with EpisodeSimulator(log=run/f'coordinated-native-{i}.log') as client:
            result=collection.collect(client,[{**task(*xy,seed=39000+i),'lift_clearance_mm':110}],dataset,hz=5,images=True,absolute_targets=True)
        print(result,flush=True)
        if not json.loads((dataset/'collection-report.json').read_text())['episodes'][-1]['success']:
            raise RuntimeError('Coordinated reference failed; inspect retained evidence')
