"""Controlled LTM gripper endpoint comparison using existing MaleCNS connections.

The baseline sensory operator and CB/DN/premotor populations stay identical.
Opening is mapped to below-reference LTM activity in the rate-deviation model;
this is not an identified biological antagonist or negative firing rate.
"""
import json,os,shutil,tempfile
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from scipy import sparse
from ..assets import home,canonical_hash,checked_files,artifact_manifest,write_json,locked
from .motor_circuit import MotorCircuit,prepare as prepare_motor
from .circuit import normalized
from .feedback import FeedbackCircuit,prepare as prepare_feedback

LTM_TYPES=['ltm1-tibia MN','ltm2-femur MN','ltm MN']


def prepare(root=None):
    root=home(root);baseline_dir,baseline_meta=prepare_motor(root,comprehensive=True)
    baseline_feedback,feedback_meta=prepare_feedback(root)
    base=MotorCircuit(baseline_dir,root)
    recipe={'version':1,'baseline_motor_circuit_id':baseline_meta['motor_circuit_id'],
            'gripper_types':LTM_TYPES,'leg':'fl','side':'L','gripper_decode':'negative mean LTM rate deviation; above reference closes, below reference opens',
            'fixed_parts':'baseline retina/body inputs, sensory operator, CB/DN/premotor populations, six joint pools and their gains',
            'motor_edges':'only recorded DN/premotor inputs to the selected motor neurons','biologically_validated':False}
    identity=canonical_hash(recipe);destination=root/'motor-circuits'/identity
    with locked(root/'.locks'/('motor-'+identity)):
        if not (destination/'manifest.json').exists():
            graph=root/'prepared'/base.manifest['graph_id'];table=pq.read_table(graph/'neuron-features.parquet').to_pydict()
            counts=sparse.load_npz(graph/'contact-counts.npz').tocsr()
            ltm=np.asarray([i for i in range(len(base.ids)) if table['superclass'][i]=='vnc_motor' and table['subclass'][i]=='fl' and table['somaSide'][i]=='L' and table['type'][i] in LTM_TYPES],dtype=np.int64)
            if not len(ltm):raise ValueError('No identified left-front LTM motor neurons')
            arm=base.motor[np.any(base.readout[:6]!=0,axis=0)]
            if np.intersect1d(arm,ltm).size:raise ValueError('LTM pool overlaps an arm-joint pool')
            motor=np.union1d(arm,ltm);sources=np.r_[base.dn,base.pm]
            signs=np.asarray([base.manifest['recipe']['fast_signs'].get(nt,1) for nt in table['consensus_nt']],dtype=np.float32)
            output=(normalized(counts[motor][:,sources])@sparse.diags(signs[sources])).tocsr()
            readout=np.zeros((7,len(motor)),dtype=np.float32)
            for j,index in enumerate(base.motor):
                if index in arm:readout[:6,np.searchsorted(motor,index)]=base.readout[:6,j]
            readout[6,np.searchsorted(motor,ltm)]=-1/len(ltm)
            cb=base.cb_from_mbon.toarray()
            dn=base.dn_from_mbon_cb@np.vstack([np.eye(len(base.mbon),dtype=np.float32),cb])
            pm=base.pm_from_dn@dn
            transfer=readout@(output@np.vstack([dn,pm]))
            if np.linalg.matrix_rank(transfer,tol=1e-5)!=7:raise ValueError('The fixed pathways cannot independently span the LTM grip channel')
            gain=base.gain.copy();gain[6]=1/max(float(np.linalg.norm(transfer[6])),1e-6)
            destination.parent.mkdir(parents=True,exist_ok=True);tmp=Path(tempfile.mkdtemp(prefix='.ltm-',dir=destination.parent))
            try:
                for name in ['cb_from_mbon.npz','dn_from_mbon_cb.npz','pm_from_dn.npz','sensory-signed.npz','ATTRIBUTION.md']:shutil.copy2(baseline_dir/name,tmp/name)
                sparse.save_npz(tmp/'motor_from_dn_pm.npz',output)
                np.savez(tmp/'populations.npz',cb=base.cb,dn=base.dn,pm=base.pm,motor=motor)
                np.savez(tmp/'bridge.npz',readout=readout,gain=gain,transfer=transfer)
                mapping=base.mapping[:6]+[{'axis':'gripper','side':'L','leg':'front','robot_units':'mm aperture','positive':[],'negative':LTM_TYPES,'positive_body_ids':[],
                    'negative_body_ids':base.ids[ltm].tolist(),'decode':'single_pool_rate_deviation','opening':'below-reference activity, an engineered mapping'}]
                write_json(tmp/'mapping.json',mapping)
                details=json.loads((baseline_dir/'neurons.json').read_text())
                details['motor']=[{'body_id':int(base.ids[i]),'type':table['type'][i],'side':table['somaSide'][i],'subclass':table['subclass'][i],'nt':table['consensus_nt'][i]} for i in motor]
                write_json(tmp/'neurons.json',details)
                artifact_manifest(tmp,{'schema':'malecns-seven-motor-circuit-v1','motor_circuit_id':identity,'parent_circuit_id':baseline_meta['parent_circuit_id'],'graph_id':base.manifest['graph_id'],
                    'recipe':recipe,'populations':{'cb':len(base.cb),'dn':len(base.dn),'pm':len(base.pm),'motor':len(motor),'ltm':len(ltm)},'transfer_rank':7,'transfer_singular_values':np.linalg.svd(transfer,compute_uv=False).tolist(),'biologically_validated':False})
                os.rename(tmp,destination)
            finally:
                if tmp.exists():shutil.rmtree(tmp)
        motor_meta=json.loads((destination/'manifest.json').read_text());checked_files(destination,motor_meta)
    feedback_recipe={**feedback_meta['recipe'],'parent':identity,'sensory_inputs_source':feedback_meta['feedback_circuit_id']}
    feedback_id=canonical_hash(feedback_recipe);target=root/'feedback-circuits'/feedback_id
    with locked(root/'.locks'/('feedback-'+feedback_id)):
        if not (target/'manifest.json').exists():
            candidate=MotorCircuit(destination,root)
            with np.load(baseline_feedback/'inputs.npz',allow_pickle=False) as saved:
                body_weights=saved['plastic_base'][:,len(candidate.kc):]
            if np.linalg.matrix_rank(candidate.transfer[:,np.any(body_weights>0,axis=1)])<7:raise ValueError('Body inputs do not span the LTM variant')
            tmp=Path(tempfile.mkdtemp(prefix='.ltm-feedback-',dir=target.parent))
            try:
                for name in ['inputs.npz','sensory-groups.json','afferents.json']:shutil.copy2(baseline_feedback/name,tmp/name)
                artifact_manifest(tmp,{'schema':'malecns-feedback-circuit-v1','feedback_circuit_id':feedback_id,'motor_circuit_id':identity,'graph_id':base.manifest['graph_id'],'recipe':feedback_recipe,
                    'afferents':feedback_meta['afferents'],'ascending_inputs':feedback_meta['ascending_inputs'],'plastic_edges':feedback_meta['plastic_edges']})
                os.rename(tmp,target)
            finally:
                if tmp.exists():shutil.rmtree(tmp)
        meta=json.loads((target/'manifest.json').read_text());checked_files(target,meta)
    validate_transfer(FeedbackCircuit(baseline_feedback,root),FeedbackCircuit(target,root))
    return target,meta


def validate_transfer(old,new):
    if old.manifest['graph_id']!=new.manifest['graph_id']:raise ValueError('Readout transfer changed the graph')
    for name in ['ids','retina','retina_uv','retina_side','visual_kc','ascending','body_indices','kc','mbon','dan','cb','dn','pm','plastic_base','input_signs']:
        if not np.array_equal(getattr(old,name),getattr(new,name)):raise ValueError('Readout transfer changed '+name)
    if old.sensory_groups!=new.sensory_groups:raise ValueError('Readout transfer changed body encoding')
    for name in ['sensory','cb_from_mbon','dn_from_mbon_cb','pm_from_dn']:
        a=getattr(old,name).tocsr();b=getattr(new,name).tocsr()
        if a.shape!=b.shape or any(not np.array_equal(getattr(a,key),getattr(b,key)) for key in ['indptr','indices','data']):raise ValueError('Readout transfer changed '+name)
    for axis in range(6):
        a={int(old.motor[i]):float(w) for i,w in enumerate(old.readout[axis]) if w}
        b={int(new.motor[i]):float(w) for i,w in enumerate(new.readout[axis]) if w}
        if a!=b or old.gain[axis]!=new.gain[axis]:raise ValueError('Readout transfer changed an arm-joint mapping')
    old_base=getattr(old,'motor_synapse_base',old.motor_from_dn_pm).tocsr()
    new_base=getattr(new,'motor_synapse_base',new.motor_from_dn_pm).tocsr()
    for neuron in old.motor[np.any(old.readout[:6]!=0,axis=0)]:
        a=old_base.getrow(int(np.flatnonzero(old.motor==neuron)[0]))
        b=new_base.getrow(int(np.flatnonzero(new.motor==neuron)[0]))
        if not np.array_equal(a.indices,b.indices) or not np.array_equal(a.data,b.data):raise ValueError('Readout transfer changed measured arm motor edges')
    return {'sensory_operator_identical':True,'arm_pathways_identical':True,'source_motor_circuit_id':old.motor_manifest['motor_circuit_id']}


def transferred_strengths(old,new):
    validate_transfer(old,new)
    sources=np.r_[old.dn,old.pm];before=old.motor_from_dn_pm.tocoo();after=getattr(new,'motor_synapse_base',new.motor_from_dn_pm).tocoo()
    lookup={(int(old.motor[r]),int(sources[c])):abs(float(w)) for r,c,w in zip(before.row,before.col,before.data)}
    values=np.asarray([lookup.get((int(new.motor[r]),int(sources[c])),abs(float(w))) for r,c,w in zip(after.row,after.col,after.data)])
    if (values>16*np.abs(after.data)+1e-7).any():raise ValueError('Transferred weights exceed the measured-edge bounds')
    return values
