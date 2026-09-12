"""Measure recorded visual-projection inputs to the mapped descending neurons."""
import argparse,json,shutil,time
from pathlib import Path
from collections import Counter
import numpy as np
import pyarrow.parquet as pq
from scipy import sparse
from fly_brain.assets import write_json,sha256_file
from fly_brain.visual_dopamine.motor_policy import MotorPolicy
from mlx_sensory_backend import MLXFeedbackCore

p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[1];cp=a.checkpoint.resolve();out=a.output.resolve()
if out.exists() or not out.is_relative_to(root/'data'):raise ValueError('Use a new project-local output')
out.mkdir(parents=True);policy=MotorPolicy(cp,root/'data');c=policy.circuit;core=policy.core;gpu=MLXFeedbackCore(core)
graph=root/'data/prepared'/c.manifest['graph_id'];counts=sparse.load_npz(graph/'contact-counts.npz').tocsr();table=pq.read_table(graph/'neuron-features.parquet').to_pydict()
influence=abs(c.motor_from_dn_pm[:,:len(c.dn)])+abs(c.motor_from_dn_pm[:,len(c.dn):])@abs(c.pm_from_dn)
active_dn=np.asarray(influence.sum(0)).ravel()>0
sub=counts[c.dn].copy();sub=sparse.diags(active_dn.astype(float))@sub
incoming=np.asarray(sub.sum(0)).ravel()
indices=np.array([i for i in range(len(c.ids)) if incoming[i]>=10 and table['superclass'][i]=='visual_projection' and c.sensory.indptr[i+1]>c.sensory.indptr[i]],dtype=np.int64)
signs=np.array([c.manifest['recipe']['fast_signs'].get(table['consensus_nt'][i],1) for i in indices],dtype=np.float32)
indices=indices[signs!=0];signs=signs[signs!=0]
matrix=sub[:,indices].tocsr().astype(np.float32);totals=np.asarray(matrix.sum(1)).ravel();matrix=(sparse.diags(np.divide(1,totals,out=np.zeros_like(totals),where=totals>0))@matrix).tocoo()
np.savez(out/'topology.npz',indices=indices,body_ids=c.ids[indices],signs=signs,post=matrix.row,pre=matrix.col,base=matrix.data,shape=np.array(matrix.shape),dn_body_ids=c.ids[c.dn])
write_json(out/'neurons.json',[{'body_id':int(c.ids[i]),'type':table['type'][i],'nt':table['consensus_nt'][i]} for i in indices])
rows=json.loads((cp/'feature-rows.json').read_text());folders={f.name:f for d in policy.metadata['datasets'] for f in Path(d).glob('episode-*')}
with np.load(cp/'training-features.npz',allow_pickle=False) as saved:reference=saved['features'].copy()
raw=np.empty((len(rows),len(indices)),dtype=np.float32);cache={};last=None;within=0;episodes=[];cpu_rows=[];cpu_values=[];old_error=0.;start=time.perf_counter()
wide={e['episode'] for e in policy.metadata['episodes'] if abs(e['cube_xy_mm'][1])>20};full_cpu=next(r['episode'] for r in rows if r['episode'] in wide)
# Use an independent CPU core; the MLX object's base public state is display-only.
cpu_policy=MotorPolicy(cp,root/'data');cpu=cpu_policy.core
for i,row in enumerate(rows):
 ep=row['episode']
 if ep!=last:
  gpu.reset();cpu.reset();last=ep;within=0;episodes.append(ep)
 if ep not in cache:cache[ep]=[json.loads(line) for line in (folders[ep]/'steps.jsonl').read_text().splitlines()]
 observation=cache[ep][row['step']]['observation']
 features=gpu.encode_observation(observation,folders[ep]);old_error=max(old_error,float(abs(features-reference[i]).max()))
 raw[i]=np.array(gpu.gpu_state)[indices,0]
 if ep==full_cpu or (len(episodes)<=2 and within<6):
  cpu.encode_observation(observation,folders[ep]);cpu_rows.append(i);cpu_values.append(cpu._visual_state[indices,0].copy())
 within+=1
 if (i+1)%200==0:print(json.dumps({'frames':i+1,'inputs':len(indices),'edges':matrix.nnz,'old_feature_error':old_error,'seconds':time.perf_counter()-start}),flush=True)
train=np.array([r['split']=='train' for r in rows]);mean=raw[train].mean(0);std=raw[train].std(0);std=np.maximum(std,max(float(std.max())*.005,1e-8));features=np.tanh((raw-mean)/std)
cpu_values=np.asarray(cpu_values);cpu_rows=np.asarray(cpu_rows,dtype=np.int64);cpu_features=np.tanh((cpu_values-mean)/std)
error=float(abs(cpu_features-features[cpu_rows]).max())
np.savez(out/'features.npz',features=features,mean=mean,std=std,cpu_rows=cpu_rows,cpu_features=cpu_features)
shutil.copy2(cp/'feature-rows.json',out/'feature-rows.json')
report={'source_checkpoint':str(cp),'source_model_sha256':sha256_file(cp/'model.npz'),'source_features_sha256':sha256_file(cp/'training-features.npz'),'source_rows_sha256':sha256_file(cp/'feature-rows.json'),'graph_id':c.manifest['graph_id'],'motor_circuit_id':policy.metadata['motor_circuit_id'],'input_count':len(indices),'measured_synapses':matrix.nnz,'active_descending':int(active_dn.sum()),'frames':len(rows),'calibration':'Train rows only; standardized rate deviations with a shared minimum variance','cpu_checked_frames':len(cpu_rows),'cpu_feature_max_error':error,'old_feature_max_error':old_error,'backend_source_sha256':sha256_file(root/'scripts/mlx_sensory_backend.py'),'extractor_source_sha256':sha256_file(__file__),'files':{f.name:sha256_file(f) for f in out.iterdir() if f.is_file()},'passed':error<1e-4 and old_error<1e-4}
write_json(out/'manifest.json',report);print(json.dumps(report,indent=2),flush=True)
if not report['passed']:raise SystemExit('Reflex input numerical check failed')
