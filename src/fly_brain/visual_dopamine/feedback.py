"""Recorded leg afferents and ascending inputs supply robot proprioception/touch.

Neuron identities and connections are measured. Robot-to-sensory tuning is an
explicit engineering adapter; it is not a claim about each fly neuron's tuning.
"""
import json,os,tempfile,shutil
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from scipy import sparse
from ..assets import home,canonical_hash,checked_files,artifact_manifest,write_json,locked
from ..adapters import JOINT_LOWER,JOINT_UPPER
from .motor_circuit import MotorCircuit,prepare as prepare_motor
from .circuit import normalized
from .core import VisualCore


def prepare(root=None):
    root=home(root);parent,pm=prepare_motor(root,comprehensive=True)
    recipe={'version':2,'modality_routing':'independent fixed sensory propagation: retina to visual KC; body afferents to ascending neurons','parent':pm['motor_circuit_id'],'afferents':'leg chordotonal organs, hair plates, leg bristles','tuning':'Gaussian joint-position population codes; bilateral contact activation','ascending_inputs':'recorded ascending/sensory-ascending neurons projecting directly to selected MBONs','normalization':'visual KC and ascending input blocks normalized separately','physiological_validation':False}
    identity=canonical_hash(recipe);dest=root/'feedback-circuits'/identity
    with locked(root/'.locks'/('feedback-'+identity)):
        if (dest/'manifest.json').exists():
            m=json.loads((dest/'manifest.json').read_text());checked_files(dest,m);return dest,m
        c=MotorCircuit(parent,root);g=root/'prepared'/c.manifest['graph_id'];t=pq.read_table(g/'neuron-features.parquet').to_pydict();counts=sparse.load_npz(g/'contact-counts.npz').tocsr()
        idx=lambda f:np.asarray([i for i in range(len(c.ids)) if f(i)],dtype=np.int64)
        groups=[]
        for side in ['L','R']:
            for nerves in [('ProCN','ProLN'),('MesoLN',),('MetaLN',)]:
                ids=idx(lambda i:t['superclass'][i]=='vnc_sensory' and t['subclass'][i]=='chordotonal organ' and t['rootSide'][i]==side and t['entryNerve'][i] in nerves)
                groups.append({'kind':'position','channel':len(groups),'indices':ids.tolist()})
        hp=idx(lambda i:t['superclass'][i]=='vnc_sensory' and t['subclass'][i]=='hair plate')
        groups.append({'kind':'position','channel':6,'indices':hp.tolist()})
        for side in ['L','R']:
            ids=idx(lambda i:t['superclass'][i]=='vnc_sensory' and t['subclass'][i]=='leg bristle' and t['rootSide'][i]==side)
            groups.append({'kind':'contact','channel':side,'indices':ids.tolist()})
        if any(not g['indices'] for g in groups):raise ValueError('An anatomical body-sensory group is empty')
        body_indices=np.unique(np.concatenate([g['indices'] for g in groups]))
        if len(body_indices)!=sum(len(g['indices']) for g in groups):raise ValueError('Body sensory groups must not overlap')
        all_an=idx(lambda i:t['superclass'][i] in ['ascending_neuron','sensory_ascending'] and c.manifest['recipe']['fast_signs'].get(t['consensus_nt'][i],1)!=0)
        an=all_an[np.asarray(counts[c.mbon][:,all_an].sum(axis=0)).ravel()>0]
        body_weights=normalized(counts[c.mbon][:,an]).toarray()
        signs=np.asarray([c.manifest['recipe']['fast_signs'].get(t['consensus_nt'][i],1) for i in an],dtype=np.float32)
        if np.linalg.matrix_rank(c.transfer[:,np.any(body_weights>0,axis=1)])<7:raise ValueError('Ascending input targets do not span all motor channels')
        dest.parent.mkdir(parents=True,exist_ok=True);tmp=Path(tempfile.mkdtemp(prefix='.feedback-',dir=dest.parent))
        try:
            np.savez(tmp/'inputs.npz',body_indices=body_indices,ascending=an,plastic_base=np.c_[c.plastic_base,body_weights],input_signs=np.r_[np.ones(len(c.kc),dtype=np.float32),signs])
            write_json(tmp/'sensory-groups.json',groups)
            details=[{'body_id':int(c.ids[i]),'type':t['type'][i],'class':t['subclass'][i],'root_side':t['rootSide'][i],'entry_nerve':t['entryNerve'][i]} for i in body_indices]
            write_json(tmp/'afferents.json',details)
            m=artifact_manifest(tmp,{'schema':'malecns-feedback-circuit-v1','feedback_circuit_id':identity,'motor_circuit_id':pm['motor_circuit_id'],'graph_id':c.manifest['graph_id'],'recipe':recipe,'afferents':len(body_indices),'ascending_inputs':len(an),'plastic_edges':int(np.count_nonzero(c.plastic_base)+np.count_nonzero(body_weights))})
            os.rename(tmp,dest)
        finally:
            if tmp.exists():shutil.rmtree(tmp)
    return dest,m


class FeedbackCircuit(MotorCircuit):
    def __init__(self,directory,root=None):
        directory=Path(directory);meta=json.loads((directory/'manifest.json').read_text());checked_files(directory,meta)
        super().__init__(home(root)/'motor-circuits'/meta['motor_circuit_id'],root)
        self.feedback_manifest=meta;self.visual_kc=self.kc.copy()
        with np.load(directory/'inputs.npz',allow_pickle=False) as d:
            self.body_indices=d['body_indices'].copy();self.ascending=d['ascending'].copy();self.plastic_base=d['plastic_base'].copy();self.input_signs=d['input_signs'].copy()
        self.kc=np.r_[self.visual_kc,self.ascending]  # Legacy name: all plastic-input neurons.
        self.sensory_groups=json.loads((directory/'sensory-groups.json').read_text())


class BodyEncoder:
    def __init__(self,circuit):
        self.circuit=circuit
        self.lower=np.r_[JOINT_LOWER,0.];self.upper=np.r_[JOINT_UPPER,90.]

    def sample(self,observation):
        body=np.r_[observation['joints_deg'],observation['gripper_mm']].astype(float)
        if body.shape!=(7,) or not np.isfinite(body).all():raise ValueError('Invalid robot proprioception')
        result=np.zeros(len(self.circuit.body_indices),dtype=np.float32)
        for group in self.circuit.sensory_groups:
            selected=np.searchsorted(self.circuit.body_indices,group['indices'])
            if group['kind']=='position':
                axis=group['channel'];value=np.clip((body[axis]-self.lower[axis])/(self.upper[axis]-self.lower[axis]),0,1)
                centers=np.linspace(0,1,len(selected));sigma=max(.06,1.5/max(1,len(selected)-1))
                values=np.exp(-.5*((centers-value)/sigma)**2)
                result[selected]=values
            else:result[selected]=float(observation['finger_contacts']['left' if group['channel']=='L' else 'right'])
        return result


class FeedbackCore(VisualCore):
    def __init__(self,circuit,updates=12):
        super().__init__(circuit,updates);self.body_encoder=BodyEncoder(circuit)

    def response(self,retinal,body,visual_enabled=True):
        if self.circuit.feedback_manifest['recipe']['version']==1:
            self.state.fill(0)
            drive=np.clip((retinal-self.retinal_mean)/self.retinal_std,-1,1) if visual_enabled else np.zeros_like(retinal)
            for _ in range(self.updates):
                self.state=.35*self.state+.65*np.tanh(.95*(self.circuit.sensory@self.state))
                self.state[self.circuit.retina]=drive
                self.state[self.circuit.body_indices]=body
            return self.state[self.circuit.kc].copy()
        # Separately propagate registered modalities so short body pathways cannot
        # swamp the much longer visual route. Both use recorded synaptic paths.
        self.visual_response(retinal,visual_enabled)
        visual_state=self.state.copy()
        self.state.fill(0)
        for _ in range(self.updates):
            self.state=.35*self.state+.65*np.tanh(.95*(self.circuit.sensory@self.state))
            self.state[self.circuit.body_indices]=body
        ascending=self.state[self.circuit.ascending].copy()
        self.state+=visual_state
        self.state[self.circuit.visual_kc]=visual_state[self.circuit.visual_kc]
        self.state[self.circuit.ascending]=ascending
        return self.state[self.circuit.kc].copy()

    def fit_samples(self,retinal,body):
        retinal=np.asarray(retinal);self.retinal_mean=retinal.mean(axis=0);self.retinal_std=np.maximum(retinal.std(axis=0),.01)
        raw=np.asarray([self.response(a,b) for a,b in zip(retinal,body)])
        self.kc_mean=raw.mean(axis=0);std=raw.std(axis=0)
        self.kc_std=np.maximum(std,max(float(std.max())*.0005,1e-12))
        return self.encode_responses(raw)

    def encode_observation(self,observation,base_directory=None,visual_enabled=True):
        retinal=self.encoder.sample(observation['images'],base_directory);body=self.body_encoder.sample(observation)
        encoded=self.encode_responses(self.response(retinal,body,visual_enabled));self.state[self.circuit.kc]=encoded
        return encoded


class TorchFeedbackEncoder:
    """Batched fixed sensory propagation for offline CPU/CUDA feature extraction."""
    def __init__(self,core,device='cuda'):
        import torch
        if core.circuit.feedback_manifest['recipe']['version']!=2:
            raise ValueError('Batched feedback requires modality-separated circuit version 2')
        self.core=core;self.device=torch.device(device)
        c=core.circuit;m=c.sensory
        self.matrix=torch.sparse_csr_tensor(torch.tensor(m.indptr,dtype=torch.int64,device=device),
            torch.tensor(m.indices,dtype=torch.int64,device=device),torch.tensor(m.data,dtype=torch.float32,device=device),size=m.shape)
        self.retina=torch.tensor(c.retina,device=device);self.body=torch.tensor(c.body_indices,device=device)
        self.visual_kc=torch.tensor(c.visual_kc,device=device);self.ascending=torch.tensor(c.ascending,device=device)

    def responses(self,retinal,body,batch_size=32,return_state=False):
        import torch
        outputs=[]
        if return_state and len(retinal)!=1:raise ValueError("Activity state requires one observation")
        with torch.no_grad():
            mean=torch.tensor(self.core.retinal_mean,device=self.device);std=torch.tensor(self.core.retinal_std,device=self.device)
            for start in range(0,len(retinal),batch_size):
                r=torch.tensor(np.asarray(retinal[start:start+batch_size]),dtype=torch.float32,device=self.device)
                b=torch.tensor(np.asarray(body[start:start+batch_size]),dtype=torch.float32,device=self.device)
                state=torch.zeros((self.matrix.shape[0],len(r)),device=self.device)
                drive=torch.clamp((r-mean)/std,-1,1).T
                for _ in range(self.core.updates):
                    state=.35*state+.65*torch.tanh(.95*torch.sparse.mm(self.matrix,state));state[self.retina]=drive
                visual=state[self.visual_kc].T.clone()
                visual_state=state.clone() if return_state else None
                state.zero_()
                for _ in range(self.core.updates):
                    state=.35*state+.65*torch.tanh(.95*torch.sparse.mm(self.matrix,state));state[self.body]=b.T
                outputs.append(torch.cat([visual,state[self.ascending].T],dim=1).cpu().numpy())
                if return_state:
                    combined=state+visual_state;combined[self.visual_kc]=visual.T;combined[self.ascending]=state[self.ascending]
                    self.last_state=combined[:,0].cpu().numpy()
        return np.concatenate(outputs)


class ScheduledFeedbackEncoder:
    """Exact offline CPU evaluation of the finite sensory integration horizon.

    At each step, retain only nodes that can already be influenced by a clamped
    input and can still reach an output before the final step. Original signed
    weights and leak/tanh updates are preserved; no graph weights are renormalized.
    This returns readout features only, not a full-brain activity visualization.
    """
    def __init__(self,core):
        if core.circuit.feedback_manifest['recipe']['version']!=2:
            raise ValueError('Scheduled feedback requires separated sensory modalities')
        self.core=core;c=core.circuit;graph=c.sensory.astype(bool);reverse=graph.T.tocsr()
        def distance(matrix,seeds):
            result=np.full(matrix.shape[0],core.updates+1,np.int16);result[seeds]=0;front=result==0
            for hop in range(1,core.updates):
                front=(matrix@front)&(result>hop)
                if not front.any():break
                result[front]=hop
            return result
        self.routes=[]
        for inputs,outputs in [(c.retina,c.visual_kc),(c.body_indices,c.ascending)]:
            forward=distance(graph,inputs);backward=distance(reverse,outputs)
            previous=np.array([],dtype=int);steps=[]
            for step in range(1,core.updates+1):
                nodes=np.flatnonzero((forward<=step-1)&(backward<=core.updates-step))
                matrix=c.sensory[nodes][:,previous]
                _,current_overlap,previous_overlap=np.intersect1d(nodes,previous,return_indices=True)
                _,clamped,input_values=np.intersect1d(nodes,inputs,return_indices=True)
                steps.append((matrix,current_overlap,previous_overlap,clamped,input_values));previous=nodes
            _,output_values,last_values=np.intersect1d(outputs,previous,return_indices=True)
            self.routes.append((steps,output_values,last_values,len(outputs)))

    def responses(self,retinal,body,batch_size=8,progress=None):
        if len(retinal)!=len(body) or not len(retinal) or batch_size<1:raise ValueError('Aligned nonempty sensory batches required')
        outputs=[]
        for start in range(0,len(retinal),batch_size):
            r=np.asarray(retinal[start:start+batch_size],dtype=np.float32)
            b=np.asarray(body[start:start+batch_size],dtype=np.float32)
            drives=[np.clip((r-self.core.retinal_mean)/self.core.retinal_std,-1,1).T,b.T]
            values=[]
            for (steps,output_values,last_values,count),drive in zip(self.routes,drives):
                state=np.zeros((0,len(r)),np.float32)
                for matrix,current_overlap,previous_overlap,clamped,input_values in steps:
                    value=.65*np.tanh(.95*(matrix@state))
                    value[current_overlap]+=.35*state[previous_overlap]
                    value[clamped]=drive[input_values];state=value
                result=np.zeros((len(r),count),np.float32);result[:,output_values]=state[last_values].T;values.append(result)
            outputs.append(np.concatenate(values,axis=1))
            if progress:progress(min(start+batch_size,len(retinal)),len(retinal))
        return np.concatenate(outputs)
