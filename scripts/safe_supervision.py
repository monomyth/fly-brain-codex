"""Training-only, observation-based grasp targets. Never imported by the actor."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from fly_brain.adapters import JOINT_LOWER,JOINT_UPPER

class BatchKinematics:
    def __init__(self):
        path=Path(__file__).resolve().parents[1]/'src/fly_brain/robot_kinematics.json'
        definition=json.loads(path.read_text());self.joints=definition['joints'];self.offset=np.array(definition['grasp_offset_m'])
        self.origins=[Rotation.from_euler('xyz',j['rpy']).as_matrix() for j in self.joints]
    def forward(self,q):
        q=np.asarray(q,dtype=float);n=len(q);r=np.tile(np.eye(3),(n,1,1));p=np.zeros((n,3));origins=[];axes=[];k=0
        for j,fixed in zip(self.joints,self.origins):
            p+=np.einsum('nij,j->ni',r,np.asarray(j['xyz']));r=r@fixed
            if j['type']=='revolute':
                axis=np.asarray(j['axis']);origins.append(p.copy());axes.append(np.einsum('nij,j->ni',r,axis))
                cross=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
                a=np.deg2rad(q[:,k])[:,None,None];r=r@(np.eye(3)+np.sin(a)*cross+(1-np.cos(a))*(cross@cross));k+=1
        p+=np.einsum('nij,j->ni',r,self.offset)
        axes=np.stack(axes,axis=2);origins=np.stack(origins,axis=2)
        linear=np.cross(axes.transpose(0,2,1),(p[:,:,None]-origins).transpose(0,2,1)).transpose(0,2,1)*1000*np.pi/180
        return p*1000,r,np.concatenate([linear,axes*np.pi/180],axis=1)
    def solve(self,position,rotation,seed,iterations=100):
        q=np.asarray(seed,dtype=float).copy();position=np.asarray(position);rotation=np.asarray(rotation);finished=np.zeros(len(q),dtype=bool)
        for _ in range(iterations):
            p,r,j=self.forward(q);angular=Rotation.from_matrix(rotation@r.transpose(0,2,1)).as_rotvec();delta=position-p
            finished=(np.linalg.norm(delta,axis=1)<.03)&(np.linalg.norm(angular,axis=1)<.0005)
            if finished.all():break
            error=np.c_[delta,angular*100];j[:,3:]*=100
            jt=j.transpose(0,2,1);h=jt@j+np.eye(6)*.04;rhs=np.einsum('nij,nj->ni',jt,error)
            change=np.linalg.solve(h,rhs[...,None])[...,0];change[finished]=0
            change*=np.minimum(1,8/np.maximum(abs(change).max(1),1e-9))[:,None]
            q=np.clip(q+change,JOINT_LOWER,JOINT_UPPER)
        p,r,_=self.forward(q);pos_error=np.linalg.norm(p-position,axis=1);angle_error=np.rad2deg(Rotation.from_matrix(rotation@r.transpose(0,2,1)).magnitude())
        return q,pos_error,angle_error


def safe_goal(observation,evaluation,approach_profile="discrete",hold_center=None):
    if approach_profile not in ("discrete","smooth","direct"):raise ValueError("Unknown approach profile")
    tool=observation['grasp_pose'];cube=observation.get('cube_pose',evaluation['cube_pose'])
    p=np.asarray(tool['position_mm']);cube_p=np.asarray(cube['position_mm']);r=Rotation.from_quat(tool['quaternion_xyzw']).as_matrix()
    desired=Rotation.from_euler('y',90,degrees=True).as_matrix()
    contacts=observation['finger_contacts'];angle=Rotation.from_matrix(desired@r.T).magnitude();xy=np.linalg.norm(p[:2]-cube_p[:2])
    if all(contacts.values()):
        if hold_center is None and evaluation.get('clearance_mm',0)>=120 and evaluation.get('tilt_deg',180)<=3:
            return p,r,0.,'hold'
        cube_r=Rotation.from_quat(cube['quaternion_xyzw']).as_matrix();yaw=Rotation.from_matrix(cube_r).as_euler('xyz')[2]
        level=Rotation.from_euler('z',yaw).as_matrix() if hold_center is None else np.eye(3);relative=cube_r.T@(p-cube_p)
        center=np.r_[cube_p[:2],max(134.,cube_p[2])] if hold_center is None else np.array([*hold_center,134.])
        target=center+level@relative
        return target,level@cube_r.T@r,0.,'lift'
    if any(contacts.values()) and evaluation.get('clearance_mm',0)>5:
        return p,r,0.,'secure'
    if angle>.15:
        return np.array([350.,0.,181.]),desired,90.,'unfold'
    if approach_profile == "direct":
        aperture=float(observation.get('gripper_mm',90.))
        if aperture < 17 or (xy > 30 and p[2] < 45):
            return np.r_[p[:2],max(65.,p[2])],desired,90.,'raise'
        # An open 90 mm gripper can descend while completing small lateral moves.
        # Closing persists inside the teacher's 5 mm lateral / 0.12 rad envelope.
        begin=xy<=3 and abs(p[2]-27)<=3 and angle<=.04
        continuing=aperture<70 and xy<=5 and abs(p[2]-27)<=9 and angle<=.12
        closing=begin or continuing
        return np.r_[cube_p[:2],27.],desired,0. if closing else 90.,'close' if closing else 'descend'
    if approach_profile == "smooth" and (xy > 3 or angle > .04):
        # Avoid a 38 mm vertical target jump at the alignment boundary.
        # Large lateral travel still starts above the cube and fingers.
        if p[2] < 45 and xy > 10:
            return np.r_[p[:2],65.],desired,90.,'raise'
        blend = np.clip(max((xy-3)/7, (angle-.04)/.11), 0., 1.)
        height = 27. + 38.*blend*blend*(3.-2.*blend)
        return np.r_[cube_p[:2],height],desired,90.,'align' if height>45 else 'descend'
    if xy>3 or angle>.04:
        if p[2]<45:
            return np.r_[p[:2],65.],desired,90.,'raise'
        return np.r_[cube_p[:2],65.],desired,90.,'align'
    return np.r_[cube_p[:2],27.],desired,0. if abs(p[2]-27)<=3 else 90.,'close' if abs(p[2]-27)<=3 else 'descend'
