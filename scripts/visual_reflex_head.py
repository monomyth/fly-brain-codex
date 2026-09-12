"""A learned measured visual-projection-to-descending synaptic branch."""
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from head_training_runtime import load_bundle

class VisualReflexHead(nn.Module):
    def __init__(self,brain,topology):
        super().__init__();self.joint_gain=brain.joint_gain;self.dn_count=topology['shape'][0]
        self.register_buffer('post',torch.tensor(topology['post'],dtype=torch.long))
        self.register_buffer('pre',torch.tensor(topology['pre'],dtype=torch.long))
        self.register_buffer('base',torch.tensor(topology['base'],dtype=torch.float64))
        self.register_buffer('sign',torch.tensor(topology['signs'][topology['pre']],dtype=torch.float64))
        self.shape=tuple(int(x) for x in topology['shape'])
        self.multipliers=nn.Parameter(torch.zeros(len(topology['base']),dtype=torch.float64))
        for name in ['pm_from_dn','readout','gain','reference']:self.register_buffer(name,getattr(brain,name).detach().clone())
        matrix=torch.sparse_coo_tensor(brain.motor_indices,brain.motor_strength.detach()*brain.motor_signs,brain.motor_shape).coalesce()
        self.register_buffer('motor_matrix',matrix)
    def forward(self,features):
        matrix=torch.sparse_coo_tensor(torch.stack([self.post,self.pre]),self.base*self.sign*self.multipliers,self.shape).coalesce()
        drive=torch.sparse.mm(matrix,features[:,self.dn_count:].T).T*.1
        dn=torch.tanh(features[:,:self.dn_count]+drive)
        pm=torch.tanh(torch.sparse.mm(self.pm_from_dn,dn.T).T)
        motor=torch.tanh(torch.sparse.mm(self.motor_matrix,torch.cat([dn,pm],1).T).T)
        return (motor@self.readout.T)*self.gain-self.reference


def load_reflex(bundle,reflex,device):
    brain,fk,data,recipe=load_bundle(bundle,device)
    with np.load(Path(reflex)/'topology.npz',allow_pickle=False) as z:topology={k:z[k].copy() for k in z.files}
    with np.load(Path(reflex)/'features.npz',allow_pickle=False) as z:visual=torch.tensor(z['features'],dtype=torch.float64,device=device)
    manifest=json.loads((Path(reflex)/'manifest.json').read_text())
    if not manifest['passed'] or manifest['source_model_sha256']!=recipe['model_sha256'] or manifest['source_features_sha256']!=recipe['features_sha256']:
        raise ValueError('Reflex data must match the source brain and observations')
    with torch.no_grad():
        x=data['features'];mbon=torch.tanh(.1*x*brain.input_signs@(brain.weights*brain.mask).T+brain.excitability)
        cb=torch.tanh(torch.sparse.mm(brain.cb_from_mbon,mbon.T).T)
        dn_pre=torch.sparse.mm(brain.dn_from_mbon_cb,torch.cat([mbon,cb],1).T).T
        expected=brain(x)
        if len(visual)!=len(x):raise ValueError('Reflex frame alignment mismatch')
        data['features']=torch.cat([dn_pre,visual],1).detach()
    model=VisualReflexHead(brain,topology).to(device)
    with torch.no_grad():
        error=(model(data['features'])-expected).abs().max().item()
        if error>1e-10:raise ValueError('Zero reflex weights changed the source brain')
    return model,fk,data,recipe,manifest
