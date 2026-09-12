"""Check CUDA loss/gradient agreement and speed before continued training."""
import os
import sys
from pathlib import Path
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'src'))
os.environ['CUDA_CACHE_PATH']=str(root/'.cache/cuda')
os.environ['TORCH_HOME']=str(root/'.cache/torch')
import json,time
import numpy as np
import torch
from fly_brain.assets import checked_files,write_json
from head_training_runtime import load_bundle,loss

checked_files(root,json.loads((root/'manifest.json').read_text()))
torch.set_num_threads(2)
results={};references={}
for device in ['cpu','cuda']:
    model,fk,data,recipe=load_bundle(root,device)
    rows=torch.nonzero(data['split']==0).flatten()
    for _ in range(2):model.zero_grad();value=loss(model,fk,data,rows,recipe);value.backward()
    if device=='cuda':torch.cuda.synchronize()
    timings=[]
    for _ in range(5):
        start=time.perf_counter();model.zero_grad();value=loss(model,fk,data,rows,recipe);value.backward()
        if device=='cuda':torch.cuda.synchronize()
        timings.append(time.perf_counter()-start)
    references[device]={'loss':float(value.detach().cpu()),'gradient':np.concatenate([p.grad.detach().cpu().numpy().ravel() for p in model.parameters()])}
    results[device]={'median_step_seconds':float(np.median(timings)),'training_rows':len(rows),'loss':references[device]['loss']}
    del model,fk,data
error=np.max(np.abs(references['cpu']['gradient']-references['cuda']['gradient']))
relative=error/max(float(np.max(np.abs(references['cpu']['gradient']))),1e-12)
report={'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'results':results,'gradient_max_abs_error':float(error),
        'gradient_relative_to_max_error':float(relative),'loss_abs_error':abs(references['cpu']['loss']-references['cuda']['loss']),
        'passed':np.allclose(references['cpu']['gradient'],references['cuda']['gradient'],atol=1e-6,rtol=1e-7) and abs(references['cpu']['loss']-references['cuda']['loss'])<1e-7}
write_json(root/'benchmark.json',report);print(json.dumps(report,indent=2),flush=True)
if not report['passed']:raise SystemExit('CUDA math validation failed')
