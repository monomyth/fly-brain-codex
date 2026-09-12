import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from collect_policy_recovery import RecoveryTeacher
from fly_brain.teacher import grasp_pose


class Solver:
    def __init__(self):self.last_pose=None
    def call(self,name,**kwargs):
        self.last_pose=kwargs['pose']
        return {'joints_deg':[0]*6}


def test_angled_setup_keeps_canonical_grasp_supervision():
    client=Solver();teacher=RecoveryTeacher(client,{},setup={'roll_deg':6,'pitch_deg':0,'height_mm':31})
    teacher.stage='descend'
    observation={'cube_pose':{'position_mm':[350,0,9]}}
    action={'joints_deg':[1]*6,'gripper_mm':90}
    target,path=teacher.supervision(observation,action)
    assert client.last_pose['position_mm']==[350.,0.,27.]
    expected=Rotation.from_quat(grasp_pose([350,0],20,0)['quaternion_xyzw']).as_matrix()
    assert np.allclose(Rotation.from_quat(client.last_pose['quaternion_xyzw']).as_matrix(),expected)
    assert action['joints_deg']==[1]*6
    assert target['joints_deg']==[0]*6


def test_postgrasp_leveling_target_is_retained():
    teacher=RecoveryTeacher(Solver(),{},setup={'roll_deg':6})
    teacher.stage='hold';action={'joints_deg':[1,2,3,4,5,6],'gripper_mm':0}
    target,_=teacher.supervision({},action)
    assert target is action
    teacher.finish_setup()
    assert not teacher.setup_active
    assert teacher.approach==teacher.canonical_approach
