"""Train one member of the declared static / memory / LTM comparison."""
import argparse,json,os,subprocess
from pathlib import Path
from fly_brain.assets import write_json,canonical_hash,sha256_file

p=argparse.ArgumentParser();p.add_argument('variant',choices=['static','temporal','ltm']);a=p.parse_args()
root=Path(__file__).resolve().parents[1];run=root/'reports/temporal-control-20260911'
plan=json.loads((run/'comparison-plan.json').read_text());item=plan['variants'][a.variant]
source=root/'src/fly_brain';digest=canonical_hash({str(f.relative_to(source)):sha256_file(f) for f in source.rglob('*.py')})
if digest!=plan['source_hash']:raise RuntimeError('Source changed after the comparison was declared; record a new plan before training')
cmd=json.loads((root/'reports/two-view-20260910-233652/common-training-command.json').read_text())
cmd[cmd.index('--output')+1]=item['checkpoint'];cmd[cmd.index('--warm-start')+1]=plan['warm_start'];cmd[cmd.index('--iterations')+1]=str(plan['iterations'])
cmd+=['--refit-feature-adaptation']
if a.variant=='temporal':cmd+=['--temporal-sensory']
if a.variant=='ltm':cmd+=['--feedback-circuit',plan['ltm_feedback_circuit'],'--transfer-readout']
write_json(run/f'train-{a.variant}-command.json',cmd)
with (run/f'train-{a.variant}.log').open('w') as log:
    result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'2','VECLIB_MAXIMUM_THREADS':'2'})
write_json(run/f'train-{a.variant}-exit.json',{'returncode':result.returncode,'checkpoint':item['checkpoint']})
raise SystemExit(result.returncode)
