"""Compile corrected, stateless training goals while retaining all recorded inputs."""
import argparse,json,shutil
from pathlib import Path
from collections import Counter
import numpy as np
from fly_brain.assets import artifact_manifest,write_json,sha256_file
from safe_supervision import BatchKinematics,safe_goal
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--report',type=Path,required=True);p.add_argument('--approach-profile',choices=['discrete','smooth','direct'],default='discrete');p.add_argument('--precision-weight',type=float,default=1.);p.add_argument('--hold-center',type=float,nargs=2);a=p.parse_args()
if a.hold_center is not None and not np.isfinite(a.hold_center).all():raise ValueError('Finite hold center required')
if not np.isfinite(a.precision_weight) or not 1<=a.precision_weight<=16:raise ValueError('Precision weight must be between 1 and 16')
if a.output.exists():raise ValueError('Use a new checkpoint')
m=json.loads((a.source/'manifest.json').read_text());rows=json.loads((a.source/'feature-rows.json').read_text());folders={f.name:f for d in m['datasets'] for f in Path(d).glob('episode-*')};cache={};seeds=[];positions=[];rotations=[];grips=[];roles=[];current=[]
kin=BatchKinematics();canonical=None
for row in rows:
 ep=row['episode']
 if ep not in cache:cache[ep]=[json.loads(line) for line in (folders[ep]/'steps.jsonl').read_text().splitlines()]
 record=cache[ep][row['step']];obs=record['observation'];goal,r,grip,role=safe_goal(obs,record['evaluation'],a.approach_profile,a.hold_center)
 if canonical is None and record['stage']=='approach':canonical=np.asarray(record['expert_target']['joints_deg'])
 seed=obs['joints_deg'] if role in ['raise','hold','secure'] else record['expert_target']['joints_deg']
 if role=='unfold' and canonical is not None:seed=canonical
 seeds.append(seed);positions.append(goal);rotations.append(r);grips.append(float(grip>45));roles.append(role);current.append(obs['joints_deg'])
q,error,angle=kin.solve(positions,rotations,seeds)
failed=(error>.15)|(angle>.15)
write_json(a.report/'ik-audit.json',{'rows':len(q),'failed':int(failed.sum()),'max_position_error_mm':float(error.max()),'max_orientation_error_deg':float(angle.max()),'failures':[{'row':int(i),'role':roles[i],'position':np.asarray(positions[i]).tolist(),'position_error':float(error[i]),'orientation_error':float(angle[i])} for i in np.flatnonzero(failed)]})
if failed.any():raise ValueError(f'{failed.sum()} safe targets need a valid IK solution; no training artifact created')
shutil.copytree(a.source,a.output)
with np.load(a.source/'training-targets.npz',allow_pickle=False) as z:targets={k:z[k].copy() for k in z.files}
targets['preserve_expert_target']=np.zeros(len(rows),dtype=bool)
targets['motor_targets']=np.c_[q,grips];targets['goal_position_mm']=np.asarray(positions);targets['goal_rotation']=np.asarray(rotations);targets['current_joints_deg']=np.asarray(current)
role_names=['unfold','raise','align','descend','close','lift','hold','secure']
targets.pop('safe_goal_roles',None)
targets['goal_role_code']=np.array([role_names.index(role) for role in roles],dtype=np.int8)
# Equalize the representation of the corrected control states in the training objective.
train=[i for i,r in enumerate(rows) if r['split']=='train'];counts=Counter(roles[i] for i in train)
weights=np.array([(a.precision_weight if role in ('descend','close') else 1.)/np.sqrt(counts.get(role,1)) for role in roles]);weights/=weights[train].mean();targets['sample_weights']=weights
np.savez(a.output/'training-targets.npz',**targets)
m.update(safe_hold_center_mm=a.hold_center,precision_weight=a.precision_weight,goal_role_names=role_names,name=a.output.name,preparation_only=True,supervision_contract='safe-observable-grasp-v3',approach_profile=a.approach_profile,supervision_note='Raise before large lateral travel; lower the approach target with alignment using the declared approach profile; begin closing within 3 mm and 0.04 rad; the direct profile sustains closure inside the recorded teacher envelope; use the declared local or fixed world holding target. Geometry and IK are training-only.',source_target_sha256=sha256_file(a.source/'training-targets.npz'))
m.pop('files',None);artifact_manifest(a.output,m)
write_json(a.report/'targets.json',{'checkpoint':str(a.output.resolve()),'roles':dict(Counter(roles)),'frames':len(rows),'source_features_unchanged':sha256_file(a.output/'training-features.npz')==sha256_file(a.source/'training-features.npz')});print((a.report/'targets.json').read_text())
