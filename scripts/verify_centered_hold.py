"""Physically verify an anchored holding target using the conventional teacher."""
import importlib,json
from pathlib import Path
from datetime import datetime
import numpy as np
from scipy.spatial.transform import Rotation
import fly_brain.teacher as teachers
from verify_raised_grasp import RaisedTeacher,raised_grasp_pose
from fly_brain.simulator import EpisodeSimulator
from fly_brain.visual_dopamine.motor_experiment import task
from fly_brain.assets import home,write_json,sha256_file


class CenteredTeacher(RaisedTeacher):
    version='cube_feedback_v10_centered_hold'
    def _feedback_lift_target(self,observation,clearance,cube_pose=None):
        def matrix(pose):
            m=np.eye(4);m[:3,:3]=Rotation.from_quat(pose['quaternion_xyzw']).as_matrix();m[:3,3]=pose['position_mm'];return m
        cube=matrix(observation['cube_pose'] if cube_pose is None else cube_pose)
        grasp=matrix(observation['grasp_pose'])
        desired=np.eye(4);desired[:3,:3]=Rotation.from_euler('z',self.task['cube_yaw_deg'],degrees=True).as_matrix()
        desired[:3,3]=[cube[0,3],cube[1,3],-1+self.task['cube_size_mm']/2+clearance]
        goal=desired@np.linalg.inv(cube)@grasp
        goal[:2,3]=[350,0]
        pose={'position_mm':goal[:3,3].tolist(),'quaternion_xyzw':Rotation.from_matrix(goal[:3,:3]).as_quat().tolist()}
        self.last_hold_pose=pose
        return self.client.call('rebot_solve_pose',pose=pose,frame='grasp',gripper_mm=observation['gripper_mm'])['joints_deg']


if __name__=='__main__':
    project=Path(__file__).resolve().parents[1];run=project/'reports/two-view-20260910-233652'
    dataset=home()/'datasets/rebot-pick'/('centered-hold-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
    teachers.grasp_pose=raised_grasp_pose
    collection=importlib.import_module('fly_brain.collect');collection.Teacher=CenteredTeacher
    write_json(run/'centered-hold-proof-plan.json',{'dataset':str(dataset),'hold_center_mm':[350,0],'grasp_goal_height_mm':27,'source_sha256':sha256_file(__file__),'purpose':'Conventional physical check of training target; not an autonomous model result'})
    for i,xy in enumerate([[350,0],[340,-12]]):
        with EpisodeSimulator(log=run/f'centered-proof-native-{i}.log') as client:
            result=collection.collect(client,[{**task(*xy,seed=38000+i),'lift_clearance_mm':110}],dataset,hz=5,images=True,absolute_targets=True)
        print(result,flush=True)
        if not json.loads((dataset/'collection-report.json').read_text())['episodes'][-1]['success']:
            raise RuntimeError('Centered hold verification failed; inspect the retained evidence')
