"""Identified antagonistic motor pools and a polysynaptic descending/VNC route."""
import json
import os
from pathlib import Path
import shutil
import tempfile
import numpy as np
import pyarrow.parquet as pq
from scipy import sparse
from ..assets import home, canonical_hash, locked, artifact_manifest, checked_files, write_json, ATTRIBUTION
from ..connectome import resolve_graph
from .circuit import prepare as prepare_visual, Circuit, normalized

# A declared body adapter, not anatomical homology between fly legs and a robot.
MAPPING = [
    {'axis':'J1','side':'L','positive':['Sternal anterior rotator MN'],'negative':['Sternal posterior rotator MN']},
    {'axis':'J2','side':'L','positive':['Tr extensor MN'],'negative':['Tr flexor MN','Acc. tr flexor MN']},
    {'axis':'J3','side':'L','positive':['Ti extensor MN'],'negative':['Ti flexor MN','Acc. ti flexor MN']},
    {'axis':'J4','side':'R','positive':['Sternal anterior rotator MN'],'negative':['Sternal posterior rotator MN']},
    {'axis':'J5','side':'R','positive':['Tr extensor MN'],'negative':['Tr flexor MN','Acc. tr flexor MN']},
    {'axis':'J6','side':'R','positive':['Ti extensor MN'],'negative':['Ti flexor MN','Acc. ti flexor MN']},
    {'axis':'gripper','side':'L','positive':['Ta levator MN'],'negative':['Ta depressor MN']},
]
RECIPE = {'schema':'malecns-seven-motor-circuit-v1','version':1,'leg':'fl','mapping':MAPPING,
    'routes':['MBON→CB relay→DN','MBON→DN','DN→VNC premotor→MN','DN→MN'],
    'relay_selection':'CB intrinsic other than MBON/KC receiving MBON; VNC intrinsic receiving selected DN and projecting to mapped MN',
    'normalization':'absolute incoming contact count over all included sources in each layer',
    'sensory_output_separation':'CB relays and VNC premotor rows blocked in the fixed sensory operator',
    'motor_decode':'mean positive pool minus mean negative pool; fixed channel gain from small-signal anatomical transfer',
    'sources':['https://www.nature.com/articles/s41586-024-07600-z','https://male-cns.janelia.org/download/']}


def prepare(root=None, graph_id=None, comprehensive=False):
    root=home(root);parent_path,parent=prepare_visual(root,graph_id,pam_min_synapses=1 if comprehensive else 10)
    identity=canonical_hash({'parent':parent['circuit_id'],'recipe':RECIPE});destination=root/'motor-circuits'/identity
    with locked(root/'.locks'/('motor-'+identity)):
        if (destination/'manifest.json').exists():
            meta=json.loads((destination/'manifest.json').read_text());checked_files(destination,meta);return destination,meta
        original=Circuit(parent_path);g,_=resolve_graph(root,parent['graph_id'],verify=False)
        counts=sparse.load_npz(g/'contact-counts.npz').tocsr();t=pq.read_table(g/'neuron-features.parquet').to_pydict()
        if t['bodyId']!=original.ids.tolist():raise ValueError('Motor annotations are not aligned')
        def indices(condition):return np.asarray([i for i in range(len(original.ids)) if condition(i)],dtype=np.int64)
        pools=[]
        for spec in MAPPING:
            pools.append({direction:indices(lambda i:t['superclass'][i]=='vnc_motor' and t['subclass'][i]=='fl' and t['somaSide'][i]==spec['side'] and t['type'][i] in spec[direction]) for direction in ['positive','negative']})
        if any(not len(pool[d]) for pool in pools for d in pool):raise ValueError('An annotated motor pool is empty')
        motor=np.unique(np.concatenate([p[d] for p in pools for d in p]));mbon=original.mbon
        cb_all=indices(lambda i:t['superclass'][i]=='cb_intrinsic' and t['class'][i] not in ['MBON','Kenyon_Cell'])
        cb=cb_all[np.asarray(counts[cb_all][:,mbon].sum(axis=1)).ravel()>0]
        dn_all=indices(lambda i:t['superclass'][i]=='descending_neuron')
        dn=dn_all[np.asarray(counts[dn_all][:,np.r_[mbon,cb]].sum(axis=1)).ravel()>0]
        pm_all=indices(lambda i:t['superclass'][i]=='vnc_intrinsic')
        pm=pm_all[(np.asarray(counts[pm_all][:,dn].sum(axis=1)).ravel()>0)&(np.asarray(counts[motor][:,pm_all].sum(axis=0)).ravel()>0)]
        signs=np.asarray([parent['recipe']['fast_signs'].get(nt,1) for nt in t['consensus_nt']],dtype=np.float32)
        def layer(target,source):return (normalized(counts[target][:,source])@sparse.diags(signs[source])).tocsr()
        layers={'cb_from_mbon':layer(cb,mbon),'dn_from_mbon_cb':layer(dn,np.r_[mbon,cb]),
                'pm_from_dn':layer(pm,dn),'motor_from_dn_pm':layer(motor,np.r_[dn,pm])}
        readout=np.zeros((7,len(motor)),dtype=np.float32);mapping=[]
        for axis,(spec,pool) in enumerate(zip(MAPPING,pools)):
            entry={**spec,'leg':'front','robot_units':'mm aperture' if axis==6 else 'degrees','positive_body_ids':original.ids[pool['positive']].tolist(),'negative_body_ids':original.ids[pool['negative']].tolist()}
            mapping.append(entry)
            for direction,sign in [('positive',1),('negative',-1)]:readout[axis,np.searchsorted(motor,pool[direction])]=sign/len(pool[direction])
        # Linearization measures controllability; inference uses each nonlinear neural layer.
        cb_linear=layers['cb_from_mbon'].toarray()
        dn_linear=layers['dn_from_mbon_cb']@np.vstack([np.eye(len(mbon),dtype=np.float32),cb_linear])
        pm_linear=layers['pm_from_dn']@dn_linear
        motor_linear=layers['motor_from_dn_pm']@np.vstack([dn_linear,pm_linear])
        transfer=readout@motor_linear
        singular=np.linalg.svd(transfer,compute_uv=False)
        if np.linalg.matrix_rank(transfer,tol=1e-5)!=7:raise ValueError('Anatomical output route cannot independently span all seven commands')
        gain=1/np.maximum(np.linalg.norm(transfer,axis=1),1e-6)
        allowed=np.ones(len(original.ids),dtype=np.float32);allowed[np.r_[cb,pm]]=0
        sensory=(sparse.diags(allowed)@original.sensory).tocsr();sensory.eliminate_zeros()
        adjacency=sensory.copy();adjacency.data[:]=1
        reached=np.zeros(len(original.ids),dtype=bool);reached[original.retina]=True;frontier=reached.astype(np.float32)
        for depth in range(1,17):
            nxt=(adjacency@frontier>0)&~reached;reached|=nxt;frontier=nxt.astype(np.float32)
            if reached[original.kc].all():break
        if not reached[original.kc].all():raise ValueError('Output isolation disconnected visual Kenyon cells')
        destination.parent.mkdir(parents=True,exist_ok=True);temp=Path(tempfile.mkdtemp(prefix='.motor-',dir=destination.parent))
        try:
            np.savez(temp/'populations.npz',cb=cb,dn=dn,pm=pm,motor=motor)
            np.savez(temp/'bridge.npz',readout=readout,gain=gain,transfer=transfer)
            for name,matrix in layers.items():sparse.save_npz(temp/(name+'.npz'),matrix)
            sparse.save_npz(temp/'sensory-signed.npz',sensory)
            write_json(temp/'mapping.json',mapping)
            details={name:[{'body_id':int(original.ids[i]),'type':t['type'][i],'side':t['somaSide'][i],'subclass':t['subclass'][i],'nt':t['consensus_nt'][i]} for i in values] for name,values in [('cb',cb),('dn',dn),('pm',pm),('motor',motor)]}
            write_json(temp/'neurons.json',details);(temp/'ATTRIBUTION.md').write_text(ATTRIBUTION)
            meta=artifact_manifest(temp,{'schema':RECIPE['schema'],'motor_circuit_id':identity,'parent_circuit_id':parent['circuit_id'],'graph_id':parent['graph_id'],'recipe':RECIPE,
                'populations':{'cb':len(cb),'dn':len(dn),'pm':len(pm),'motor':len(motor)},'transfer_rank':7,
                'transfer_singular_values':singular.tolist(),'visual_kc_reachable':len(original.kc),'retina_to_kc_max_hops':depth,'biologically_validated':False})
            os.rename(temp,destination)
        finally:
            if temp.exists():shutil.rmtree(temp)
    return destination,meta


class MotorCircuit(Circuit):
    def __init__(self,directory,root=None):
        self.motor_directory=Path(directory);meta=json.loads((self.motor_directory/'manifest.json').read_text());checked_files(self.motor_directory,meta)
        super().__init__(home(root)/'circuits'/meta['parent_circuit_id'])
        self.motor_manifest=meta
        with np.load(self.motor_directory/'populations.npz',allow_pickle=False) as data:
            for name in data.files:setattr(self,name,data[name].copy())
        with np.load(self.motor_directory/'bridge.npz',allow_pickle=False) as data:
            self.readout=data['readout'].copy();self.gain=data['gain'].copy();self.transfer=data['transfer'].copy()
        for name in ['cb_from_mbon','dn_from_mbon_cb','pm_from_dn','motor_from_dn_pm']:
            setattr(self,name,sparse.load_npz(self.motor_directory/(name+'.npz')).tocsr())
        self.motor_synapse_base=self.motor_from_dn_pm.copy()
        self.sensory=sparse.load_npz(self.motor_directory/'sensory-signed.npz').tocsr()
        self.neurons.update(json.loads((self.motor_directory/'neurons.json').read_text()))
        self.mapping=json.loads((self.motor_directory/'mapping.json').read_text())

    def output(self,mbon):
        cb=np.tanh(self.cb_from_mbon@mbon)
        dn=np.tanh(self.dn_from_mbon_cb@np.r_[mbon,cb])
        pm=np.tanh(self.pm_from_dn@dn)
        motor=np.tanh(self.motor_from_dn_pm@np.r_[dn,pm])
        scores=self.gain*(self.readout@motor)
        return scores,{'cb':cb,'dn':dn,'pm':pm,'motor':motor}
