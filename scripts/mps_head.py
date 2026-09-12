"""Dense Metal execution of the same measured small control circuit."""
import torch
from torch import nn

class DenseMotorHead(nn.Module):
    def __init__(self,source):
        super().__init__();self.joint_gain=source.joint_gain
        self.weights=nn.Parameter(source.weights.detach().float().clone())
        self.excitability=nn.Parameter(source.excitability.detach().float().clone())
        for name in ['mask','base','input_signs','cb_from_mbon','dn_from_mbon_cb','pm_from_dn','readout','gain','reference']:
            value=getattr(source,name).detach()
            if value.layout!=torch.strided:value=value.to_dense()
            self.register_buffer(name,value.clone() if not value.is_floating_point() else value.float().clone())
        shape=source.motor_shape;indices=source.motor_indices.cpu()
        base=torch.zeros(shape);signs=torch.zeros(shape);strength=torch.zeros(shape)
        base[indices[0],indices[1]]=source.motor_base.detach().cpu().float()
        signs[indices[0],indices[1]]=source.motor_signs.detach().cpu().float()
        strength[indices[0],indices[1]]=source.motor_strength.detach().cpu().float()
        self.register_buffer('motor_base',base);self.register_buffer('motor_signs',signs);self.register_buffer('motor_indices',indices)
        self.motor_strength=nn.Parameter(strength)
    def forward(self,features):
        mbon=torch.tanh((.1*features*self.input_signs)@(self.weights*self.mask).T+self.excitability)
        cb=torch.tanh(mbon@self.cb_from_mbon.T)
        dn=torch.tanh(torch.cat([mbon,cb],1)@self.dn_from_mbon_cb.T)
        pm=torch.tanh(dn@self.pm_from_dn.T)
        motor=torch.tanh(torch.cat([dn,pm],1)@(self.motor_strength*self.motor_signs).T)
        return (motor@self.readout.T)*self.gain-self.reference
    def project(self,weight_multiple):
        with torch.no_grad():
            self.weights.copy_(torch.minimum(self.weights.clamp_min(0),weight_multiple*self.base))
            self.weights.mul_(self.mask);self.excitability.clamp_(-2,2)
            self.motor_strength.copy_(torch.minimum(self.motor_strength.clamp_min(0),16*self.motor_base))


def mps_data(data):
    return {key:(value.float() if value.is_floating_point() else value).to('mps') for key,value in data.items()}
