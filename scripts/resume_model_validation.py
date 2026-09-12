"""Validate an existing checkpoint after a zero-action environment-blocked startup."""
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from fly_brain.assets import write_json,sha256_file
from train_policy_recovery import execute

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
run=a.run.resolve();root=Path(__file__).resolve().parents[1];plan=json.loads((run/'plan.json').read_text())
checkpoint=root/'data/checkpoints/rebot'/run.name
original=run/'evaluation/evaluation.json'
old=json.loads(original.read_text())
if any(row['steps'] for row in old['episodes']):raise ValueError('This resume helper requires zero executed original commands')
write_json(run/'startup-diagnostic-summary.json',{'original_report':str(original),'original_report_sha256':sha256_file(original),
    'startup_attempts':len(old['episodes']),'completed_autonomous_trials':0,'reason':'Screen locked before native simulation could start','model_failure_count':0})
output=run/'evaluation-unlocked'
if output.exists():raise ValueError('The resumed output already exists; retain it and inspect before another resume')
command=[str(root/'.venv-mlx/bin/python'),str(root/'scripts/run_mlx_checkpoint.py'),'--checkpoint',str(checkpoint),
         '--tasks',str(run/'evaluation-tasks.json'),'--parity',str(run/'mlx-parity.json'),'--output',str(output),
         '--gripper-response',plan['evaluation_gripper_response'],'--stop-after-failures','3']
write_json(run/'evaluate-unlocked-command.json',command)
source=run/'unlocked-runtime-source';source.mkdir()
for file in ['scripts/run_mlx_checkpoint.py','scripts/probability_gripper.py','scripts/mlx_sensory_backend.py','src/fly_brain/visual_dopamine/motor_run.py']:
    shutil.copy2(root/file,source/Path(file).name)
guard=subprocess.Popen(['/usr/bin/caffeinate','-di','-w',str(os.getpid())],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
write_json(run/'validation-process.json',{'pid':os.getpid(),'power_guard_pid':guard.pid})
try:
    write_json(run/'STATUS.json',{'status':'evaluating','checkpoint':str(checkpoint),'original_startup_attempts_excluded':len(old['episodes']),'qualified_for_promotion':None,'ui_checkpoint_changed':False})
    execute(command,run/'evaluation-unlocked.log')
    report=json.loads((output/'evaluation.json').read_text())
    blocked=report.get('execution_blocked')
    write_json(run/'STATUS.json',{'status':'blocked_environment' if blocked else 'qualification_rejected_early' if report.get('stopped_early_after_failures') else 'completed',
        'checkpoint':str(checkpoint),'attempts':report['attempts'],'successes':report['successes'],'startup_failures':report.get('startup_failures',0),
        'qualified_for_promotion':None if blocked else report['attempts']==20 and report['successes']>=18,
        'ui_checkpoint_changed':False,'evaluation':str(output/'evaluation.json'),'execution_blocked':blocked})
finally:
    if guard.poll() is None:guard.terminate();guard.wait(timeout=5)
