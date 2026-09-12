"""Differentiable reference kinematics for supervision; never used by the actor."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.spatial.transform import Rotation

DEFINITION=Path(__file__).with_name('robot_kinematics.json')

class ToolKinematics(nn.Module):
    def __init__(self):
        super().__init__();definition=json.loads(DEFINITION.read_text());joints=definition['joints']
        rotations=np.array([Rotation.from_euler('xyz',j['rpy']).as_matrix() for j in joints])
        translations=np.array([j['xyz'] for j in joints]);cross=[];indices=[];index=0
        for joint in joints:
            if joint['type']=='revolute':
                x,y,z=joint['axis'];cross.append([[0,-z,y],[z,0,-x],[-y,x,0]]);indices.append(index);index+=1
            else:cross.append(np.zeros((3,3)));indices.append(-1)
        self.indices=indices
        for name,value in [('origins',rotations),('translations',translations),('cross',np.array(cross)),('offset',definition['grasp_offset_m'])]:
            self.register_buffer(name,torch.tensor(value,dtype=torch.float64))

    def forward(self,joints_deg):
        angles=joints_deg*(torch.pi/180)
        identity=torch.eye(3,dtype=angles.dtype,device=angles.device)
        rotation=identity.expand(len(angles),3,3);position=torch.zeros((len(angles),3),dtype=angles.dtype,device=angles.device)
        for i,index in enumerate(self.indices):
            position=position+(rotation@self.translations[i].reshape(3,1)).squeeze(-1)
            rotation=rotation@self.origins[i]
            if index>=0:
                angle=angles[:,index,None,None];axis=self.cross[i]
                motion=identity+torch.sin(angle)*axis+(1-torch.cos(angle))*(axis@axis)
                rotation=rotation@motion
        position=position+(rotation@self.offset.reshape(3,1)).squeeze(-1)
        return position*1000,rotation


def demonstration_target_poses(fk,joints,examples,desired_lift_clearance_mm=None,lift_goal_offset_mm=0.,grasp_goal_height_mm=None,hold_center_mm=None,approach_center_mm=None):
    """Declared training targets; role comes from teacher labels, never actor input."""
    position,rotation=fk(joints)
    rotation=rotation.clone()
    shift=torch.zeros_like(position)
    for i,example in enumerate(examples):
        # Verified recovery targets include deliberate reopening and repositioning.
        # Generic grasp/hold shifts would change the demonstrated correction.
        if example.get('preserve_expert_target'):continue
        if example['stage'] in ('lift','hold'):
            shift[i,2]=lift_goal_offset_mm if desired_lift_clearance_mm is None else desired_lift_clearance_mm-example['reference_lift_clearance_mm']
        if approach_center_mm is not None and example.get('goal_role')=='approach':
            shift[i,:2]=torch.as_tensor(approach_center_mm,dtype=position.dtype,device=position.device)-position[i,:2]
        if hold_center_mm is not None and example.get('goal_role')=='transport':
            shift[i,:2]=torch.as_tensor(hold_center_mm,dtype=position.dtype,device=position.device)-position[i,:2]
        if grasp_goal_height_mm is not None and example.get('goal_role')=='grasp':
            shift[i,2]=grasp_goal_height_mm-position[i,2]
        if example.get('grasp_pose_override') and example.get('goal_role')=='grasp':
            from scipy.spatial.transform import Rotation
            goal=example['grasp_pose_override']
            shift[i]=torch.as_tensor(goal['position_mm'],dtype=position.dtype,device=position.device)-position[i]
            rotation[i]=torch.as_tensor(Rotation.from_quat(goal['quaternion_xyzw']).as_matrix(),dtype=rotation.dtype,device=rotation.device)
    return position+shift,rotation
