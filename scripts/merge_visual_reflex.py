"""Join previously verified and live-captured reflex features without re-encoding."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from fly_brain.assets import sha256_file,write_json
p=argparse.ArgumentParser()
for name in ['base','prior','dataset','output']:p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args()
if a.output.exists():raise ValueError('Use a new reflex corpus')
origin=json.loads((a.prior/'manifest.json').read_text());origin_hash=sha256_file(a.prior/'manifest.json')
if not origin['passed'] or sha256_file(a.base/'model.npz')!=origin['source_model_sha256']:raise ValueError('Reflex corpus must use the same frozen brain')
oldrows=json.loads((a.prior/'feature-rows.json').read_text());rows=json.loads((a.base/'feature-rows.json').read_text())
with np.load(a.prior/'features.npz',allow_pickle=False) as z:
 old=z['features'].copy();mean=z['mean'].copy();std=z['std'].copy();cpurows=z['cpu_rows'];cpufeatures=z['cpu_features'].copy()
lookup={(r['episode'],r['step']):x for r,x in zip(oldrows,old)};cpu={(oldrows[i]['episode'],oldrows[i]['step']):x for i,x in zip(cpurows,cpufeatures)};sources=[]
for folder in a.dataset.glob('episode-*'):
 meta=json.loads((folder/'manifest.json').read_text());path=folder/'reflex-features.npz'
 if not meta.get('complete') or not path.exists():continue
 if meta.get('reflex_input_manifest_sha256')!=origin_hash or meta.get('reflex_features_sha256')!=sha256_file(path):raise ValueError('Unmatched live reflex capture')
 observations=[json.loads(line) for line in (folder/'steps.jsonl').read_text().splitlines()]
 with np.load(path,allow_pickle=False) as z:
  values=z['features'].copy();frames=z['frame_ids'].copy()
 if values.shape!=(len(observations),len(mean)) or not np.isfinite(values).all() or frames.tolist()!=[r['observation']['frame_id'] for r in observations]:raise ValueError('Live reflex alignment mismatch')
 for i,x in enumerate(values):
  key=(folder.name,i)
  if key in lookup:raise ValueError('Duplicate reflex observation')
  lookup[key]=x
 sources.append({'episode':folder.name,'sha256':sha256_file(path),'frames':len(values)})
features=np.array([lookup[(r['episode'],r['step'])] for r in rows],dtype=np.float32)
cpu_indices=np.array([i for i,r in enumerate(rows) if (r['episode'],r['step']) in cpu],dtype=np.int64)
cpu_values=np.array([cpu[(rows[i]['episode'],rows[i]['step'])] for i in cpu_indices])
a.output.mkdir(parents=True)
np.savez(a.output/'features.npz',features=features,mean=mean,std=std,cpu_rows=cpu_indices,cpu_features=cpu_values)
for name in ['topology.npz','neurons.json']:shutil.copy2(a.prior/name,a.output/name)
shutil.copy2(a.base/'feature-rows.json',a.output/'feature-rows.json')
manifest={**origin,'source_checkpoint':str(a.base.resolve()),'source_features_sha256':sha256_file(a.base/'training-features.npz'),'source_rows_sha256':sha256_file(a.base/'feature-rows.json'),'frames':len(rows),'calibration':'Inherited train-only calibration; no held-out refit','prior_manifest':str((a.prior/'manifest.json').resolve()),'prior_manifest_sha256':origin_hash,'live_sources':sources,'verification_scope':'Prior CPU/MLX checks plus unchanged-kernel live capture provenance and frame alignment; not a new full CPU replay','files':{p.name:sha256_file(p) for p in a.output.iterdir() if p.is_file()},'passed':True}
write_json(a.output/'manifest.json',manifest);print(json.dumps({'frames':len(rows),'live_sequences':len(sources),'features':features.shape[1]},indent=2))
