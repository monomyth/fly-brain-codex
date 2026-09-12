"""Portable training of the same fixed-topology head from saved neural features."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from fly_brain.visual_dopamine.trajectory_training import MotorTargetNetwork
from fly_brain.kinematics import ToolKinematics


class RestoredHead(MotorTargetNetwork):
    def __init__(self,state,recipe):
        nn.Module.__init__(self)
        for name,value in state.items():
            if name in recipe['parameter_names']:self.register_parameter(name,nn.Parameter(value.clone()))
            else:self.register_buffer(name,value.clone())
        self.train_motor=True
        self.motor_shape=tuple(recipe['motor_shape'])
        self.joint_gain=recipe['joint_gain']


def load_bundle(directory,device):
    directory=Path(directory);recipe=json.loads((directory/'recipe.json').read_text())
    state=torch.load(directory/'head-state.pt',map_location='cpu',weights_only=True)
    model=RestoredHead(state,recipe).double().to(device)
    with np.load(directory/'training-data.npz',allow_pickle=False) as saved:
        arrays={k:saved[k].copy() for k in saved.files}
    data={k:torch.as_tensor(v,device=device) for k,v in arrays.items()}
    data['features']=data['features'].double()
    return model,ToolKinematics().double().to(device),data,recipe


def orientation_error(rotation,target_rotation,held_rows,relative_cube_up,mode='full_rotation'):
    full=(rotation-target_rotation).square().sum((1,2))/2*(180/torch.pi)**2
    if mode=='full_rotation':return full
    if mode!='cube_up':raise ValueError('Unknown held orientation objective')
    predicted_up=(rotation@relative_cube_up.unsqueeze(-1)).squeeze(-1)
    vertical=torch.tensor([0.,0.,1.],dtype=rotation.dtype,device=rotation.device)
    tilt=(predicted_up-vertical).square().sum(1)*(180/torch.pi)**2
    return torch.where(held_rows,tilt,full)


def cube_clearance(position,rotation,relative_position,relative_rotation,cube_size,floor_height=-1.):
    center=position+(rotation@relative_position.unsqueeze(-1)).squeeze(-1)
    cube_rotation=rotation@relative_rotation
    half_extent=cube_rotation[:,2,:].abs().sum(1)*cube_size/2
    return center[:,2]-half_extent-floor_height


def position_error(position,rotation,data,rows,held,recipe):
    distance=(position-data['goal_position_mm'][rows]).square().sum(1)
    if recipe.get('held_position_objective')=='clearance':
        clearance=cube_clearance(position,rotation,data['relative_cube_position_mm'][rows],data['relative_cube_rotation'][rows],data['cube_size_mm'][rows],recipe.get('floor_height_mm',-1.))
        shortfall=torch.relu(recipe['held_clearance_target_mm']-clearance).square()
        distance=torch.where(held,shortfall,distance)
    return distance


def posture_weights(held_rows,recipe,dtype=torch.float64):
    if recipe.get('held_orientation_objective','full_rotation')=='cube_up':
        return torch.where(held_rows,torch.full_like(held_rows,.01,dtype=dtype),torch.full_like(held_rows,.04,dtype=dtype))
    return torch.full_like(held_rows,.04,dtype=dtype)


def retention_anchor_rows(training_rows,held_rows,recovery_rows,group='all'):
    """Protect existing demonstrations without freezing the new correction targets."""
    if group not in ('all','legacy'):raise ValueError('Unknown retention training group')
    anchors=np.asarray(training_rows,dtype=bool) & ~np.asarray(held_rows,dtype=bool)
    if group=='legacy':anchors &= ~np.asarray(recovery_rows,dtype=bool)
    return anchors


def retention_loss(scores,reference_scores,anchor_rows,weights,joint_gain):
    joint=(scores[:,:6]-reference_scores[:,:6]).square().mean(1)*joint_gain**2
    logits=scores[:,6]*2000
    baseline_logits=reference_scores[:,6]*2000
    probability=torch.sigmoid(baseline_logits)
    cross=torch.nn.functional.binary_cross_entropy_with_logits(logits,probability,reduction='none')
    entropy=torch.nn.functional.binary_cross_entropy_with_logits(baseline_logits,probability,reduction='none')
    return ((joint+2*(cross-entropy))*weights*anchor_rows).mean()


def kinematic_joints(scores,model,data,rows,recipe):
    values=scores[:,:6]*model.joint_gain
    if recipe.get('output_mode')=='joint_deltas':
        from fly_brain.adapters import JOINT_LOWER,JOINT_UPPER
        maximum=values.abs().amax(1,keepdim=True).clamp_min(1e-9)
        values=values*torch.clamp((30./recipe.get('control_hz',2.))/maximum,max=1.)
        values=values+data['current_joints_deg'][rows]
        lower=torch.as_tensor(JOINT_LOWER,dtype=values.dtype,device=values.device)
        upper=torch.as_tensor(JOINT_UPPER,dtype=values.dtype,device=values.device)
        values=torch.maximum(lower,torch.minimum(upper,values))
    return values


def loss(model,fk,data,rows,recipe):
    scores=model(data['features'][rows]);target=data['motor_targets'][rows]
    weights=data['sample_weights'][rows]
    held=data.get('held_rows',torch.zeros(len(data['features']),device=scores.device,dtype=torch.bool))[rows]
    reference=target[:,:6]
    if recipe.get('held_orientation_objective')=='cube_up':
        reference=torch.where(held[:,None],data['current_joints_deg'][rows],reference)
    joints=((scores[:,:6]*model.joint_gain-reference).square().mean(1)*weights*posture_weights(held,recipe,dtype=scores.dtype)).mean()
    grip=(torch.nn.functional.binary_cross_entropy_with_logits(scores[:,6]*2000,target[:,6],reduction='none')*weights).mean()
    position,rotation=fk(kinematic_joints(scores,model,data,rows,recipe or {}))
    distance=position_error(position,rotation,data,rows,held,recipe)
    undershoot=torch.relu(recipe['grasp_goal_height_mm']-3-position[:,2])
    distance=distance+recipe['grasp_undershoot_penalty']*undershoot.square()*data['grasp_rows'][rows]
    angle=orientation_error(rotation,data['goal_rotation'][rows],held,data['relative_cube_up'][rows] if 'relative_cube_up' in data else None,recipe.get('held_orientation_objective','full_rotation'))
    regression=joints+((distance+recipe.get('orientation_loss_weight',1.)*angle)/25*weights).mean()
    objective=regression+recipe.get('gripper_loss_weight',2.)*grip
    if recipe.get('pregrasp_retention_strength',0):
        objective=objective+recipe['pregrasp_retention_strength']*retention_loss(scores,data['reference_scores'][rows],data['anchor_rows'][rows],weights,model.joint_gain)
    return objective*recipe['dopamine_multiplier']


def statistics(model,fk,data,rows,recipe=None):
    with torch.no_grad():
        scores=model(data['features'][rows]);target=data['motor_targets'][rows]
        position,rotation=fk(kinematic_joints(scores,model,data,rows,recipe or {}))
        error=scores[:,:6]*model.joint_gain-target[:,:6]
        recipe=recipe or {}
        held=data.get('held_rows',torch.zeros(len(data['features']),device=scores.device,dtype=torch.bool))[rows]
        angular=orientation_error(rotation,data['goal_rotation'][rows],held,data['relative_cube_up'][rows] if 'relative_cube_up' in data else None,recipe.get('held_orientation_objective','full_rotation'))
        result={'joint_rmse_deg':float(error.square().mean().sqrt()),
                'goal_position_rmse_mm':float(position_error(position,rotation,data,rows,held,recipe).mean().sqrt()),
                'goal_orientation_rmse_deg':float(angular.mean().sqrt()),
                'gripper_accuracy':float(((scores[:,6]>=0)==(target[:,6]>.5)).to(scores.dtype).mean())}
        if 'goal_role_code' in data:
            critical=(data['goal_role_code'][rows]==3)|(data['goal_role_code'][rows]==4)
            if bool(critical.any()):
                offset=position[critical]-data['goal_position_mm'][rows][critical]
                result['grasp_precision']={
                    'frames':int(critical.sum()),
                    'xy_rmse_mm':float(offset[:,:2].square().sum(1).mean().sqrt()),
                    'height_rmse_mm':float(offset[:,2].square().mean().sqrt()),
                    'orientation_rmse_deg':float(angular[critical].mean().sqrt()),
                    'gripper_accuracy':float(((scores[critical,6]>=0)==(target[critical,6]>.5)).to(scores.dtype).mean())}
        if 'reference_scores' in data and bool((~held).any()):
            reference=data['reference_scores'][rows][~held];selected=scores[~held]
            difference=(selected[:,:6]-reference[:,:6])*model.joint_gain
            result['nonheld_fidelity']={'joint_rmse_deg':float(difference.square().mean().sqrt()),
                'joint_max_abs_deg':float(difference.abs().max()),
                'gripper_aperture_mae_mm':float((90*(torch.sigmoid(selected[:,6]*2000)-torch.sigmoid(reference[:,6]*2000))).abs().mean())}
        return result
