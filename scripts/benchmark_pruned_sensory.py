import json,time
from pathlib import Path
import numpy as np
from fly_brain.visual_dopamine.motor_policy import MotorPolicy
from fly_brain.cameras import CURRENT_CAMERAS,CURRENT_RIG
from scipy import sparse
project=Path(__file__).resolve().parents[1];settings=json.loads((project/'reports/two-view-current.json').read_text());run=Path(settings['run'])
policy=MotorPolicy(project/'data/checkpoints/rebot/retinal-grounding-20260910-220536',project/'data');core=policy.core;c=core.circuit
core.encoder.configure_cameras(CURRENT_CAMERAS,CURRENT_RIG);core.encoder.configure(None)
episode=next(Path(settings['dataset']).glob('episode-*/steps.jsonl')).parent
rows=[json.loads(x) for x in (episode/'steps.jsonl').read_text().splitlines()];rows=[rows[i] for i in np.linspace(0,len(rows)-1,8,dtype=int)]
r=np.asarray([core.encoder.sample(x['observation']['images'],episode) for x in rows]);b=np.asarray([core.body_encoder.sample(x['observation']) for x in rows])
G=c.sensory.astype(bool);GT=G.T.tocsr()
def distances(graph,seed):
 d=np.full(graph.shape[0],core.updates+1,np.int16);d[seed]=0;front=d==0
 for hop in range(1,core.updates):
  front=(graph@front)&(d>hop)
  if not front.any():break
  d[front]=hop
 return d
routes=[];t=time.perf_counter()
for inputs,outputs in [(c.retina,c.visual_kc),(c.body_indices,c.ascending)]:
 nodes=np.unique(np.r_[np.flatnonzero(distances(G,inputs)+distances(GT,outputs)<=core.updates-1),inputs,outputs])
 routes.append((c.sensory[nodes][:,nodes],np.searchsorted(nodes,inputs),np.searchsorted(nodes,outputs)))
setup=time.perf_counter()-t
reference=np.asarray([core.response(x,y) for x,y in zip(r,b)])
results=[];t=time.perf_counter()
for (m,inputs,outputs),drive in zip(routes,[np.clip((r-core.retinal_mean)/core.retinal_std,-1,1).T,b.T]):
 state=np.zeros((m.shape[0],len(r)),np.float32)
 for _ in range(core.updates):
  state=.35*state+.65*np.tanh(.95*(m@state));state[inputs]=drive
 results.append(state[outputs].T)
value=np.concatenate(results,axis=1);elapsed=time.perf_counter()-t
report={'frames':len(r),'setup_seconds':setup,'propagation_seconds':elapsed,'max_abs_difference':float(np.abs(value-reference).max()),'nodes':[x[0].shape[0] for x in routes],'edges':[x[0].nnz for x in routes],'original_nodes':len(c.ids),'original_edges':c.sensory.nnz}
assert np.allclose(value,reference,rtol=1e-4,atol=2e-6)
schedules=[];t=time.perf_counter()
for inputs,outputs in [(c.retina,c.visual_kc),(c.body_indices,c.ascending)]:
    forward=distances(G,inputs);back=distances(GT,outputs);previous=np.array([],dtype=int);steps=[]
    for step in range(1,core.updates+1):
        nodes=np.flatnonzero((forward<=step-1)&(back<=core.updates-step))
        matrix=c.sensory[nodes][:,previous]
        _,current_overlap,previous_overlap=np.intersect1d(nodes,previous,return_indices=True)
        _,clamped,input_values=np.intersect1d(nodes,inputs,return_indices=True)
        steps.append((nodes,matrix,current_overlap,previous_overlap,clamped,input_values));previous=nodes
    _,output_values,last_values=np.intersect1d(outputs,previous,return_indices=True)
    schedules.append((steps,output_values,last_values,len(outputs)))
schedule_setup=time.perf_counter()-t;result=[];t=time.perf_counter()
for (steps,output_values,last_values,count),drive in zip(schedules,[np.clip((r-core.retinal_mean)/core.retinal_std,-1,1).T,b.T]):
    state=np.zeros((0,len(r)),np.float32)
    for nodes,m,current_overlap,previous_overlap,clamped,input_values in steps:
        value=.65*np.tanh(.95*(m@state))
        value[current_overlap]+=.35*state[previous_overlap]
        value[clamped]=drive[input_values];state=value
    values=np.zeros((len(r),count),np.float32);values[:,output_values]=state[last_values].T;result.append(values)
time_steps=time.perf_counter()-t;value=np.concatenate(result,axis=1)
report.update(scheduled_setup_seconds=schedule_setup,scheduled_propagation_seconds=time_steps,scheduled_max_abs_difference=float(np.abs(value-reference).max()),scheduled_nodes=[[len(x[0]) for x in a[0]] for a in schedules],scheduled_edges=[[x[1].nnz for x in a[0]] for a in schedules])
assert np.allclose(value,reference,rtol=1e-4,atol=2e-6)
(run/'pruned-cpu-benchmark.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
