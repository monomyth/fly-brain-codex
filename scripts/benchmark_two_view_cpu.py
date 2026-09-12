import json,time
from pathlib import Path
import numpy as np,torch
from fly_brain.visual_dopamine.motor_policy import MotorPolicy
from fly_brain.visual_dopamine.feedback import TorchFeedbackEncoder
from fly_brain.cameras import CURRENT_CAMERAS,CURRENT_RIG
project=Path(__file__).resolve().parents[1];settings=json.loads((project/'reports/two-view-current.json').read_text());run=Path(settings['run'])
checkpoint=project/'data/checkpoints/rebot/retinal-grounding-20260910-220536'
policy=MotorPolicy(checkpoint,project/'data');core=policy.core;core.encoder.configure_cameras(CURRENT_CAMERAS,CURRENT_RIG);core.encoder.configure(None)
episode=next(Path(settings['dataset']).glob('episode-*/steps.jsonl')).parent
rows=[json.loads(x) for x in (episode/'steps.jsonl').read_text().splitlines()];rows=[rows[i] for i in np.linspace(0,len(rows)-1,8,dtype=int)]
r=np.asarray([core.encoder.sample(x['observation']['images'],episode) for x in rows]);b=np.asarray([core.body_encoder.sample(x['observation']) for x in rows]);torch.set_num_threads(2)
t=time.perf_counter();reference=np.asarray([core.response(x,y) for x,y in zip(r,b)]);serial=time.perf_counter()-t
encoder=TorchFeedbackEncoder(core,'cpu');t=time.perf_counter();batch=encoder.responses(r,b,batch_size=8);batched=time.perf_counter()-t
report={'frames':len(r),'scipy_seconds':serial,'torch_batch_seconds':batched,'raw_max_abs_difference':float(np.abs(reference-batch).max()),'threads':torch.get_num_threads()}
assert np.allclose(reference,batch,rtol=1e-4,atol=2e-6)
def scipy_batch(retinal,body):
    c=core.circuit;state=np.zeros((len(c.ids),len(retinal)),np.float32)
    drive=np.clip((retinal-core.retinal_mean)/core.retinal_std,-1,1).T
    for _ in range(core.updates):
        state=.35*state+.65*np.tanh(.95*(c.sensory@state));state[c.retina]=drive
    visual=state[c.visual_kc].T.copy();state.fill(0)
    for _ in range(core.updates):
        state=.35*state+.65*np.tanh(.95*(c.sensory@state));state[c.body_indices]=body.T
    return np.concatenate([visual,state[c.ascending].T],axis=1)
t=time.perf_counter();scipy_result=scipy_batch(r,b);scipy_time=time.perf_counter()-t
assert np.allclose(reference,scipy_result,rtol=1e-4,atol=2e-6)
report.update(scipy_batch_seconds=scipy_time,scipy_batch_max_abs_difference=float(np.abs(reference-scipy_result).max()))
(run/'cpu-benchmark.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
