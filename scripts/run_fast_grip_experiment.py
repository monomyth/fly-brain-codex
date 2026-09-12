"""Declared faster-feedback experiment with the frozen continuous-grip model."""
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from fly_brain.assets import write_json
from train_policy_recovery import execute

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
run=a.run.resolve();root=Path(__file__).resolve().parents[1]
previous=json.loads((run/'continuous-grip-status.json').read_text())
if previous.get('qualified_for_promotion') is not False:raise ValueError('Slower profile must finish first')
plan=json.loads((run/'continuous-grip-plan.json').read_text());checkpoint=plan['checkpoint']
source=run/'fast-grip-source';source.mkdir()
for name in ['run_fast_grip_experiment.py','check_mlx_cadence.py','run_mlx_checkpoint.py','probability_gripper.py','mlx_sensory_backend.py']:
    shutil.copy2(root/'scripts'/name,source/name)
write_json(run/'fast-grip-plan.json',{'checkpoint':checkpoint,'control_hz':4,'settle_seconds':.05,'max_steps':320,'approximate_decision_time_budget_seconds':80,
                                   'native_speed_limits_changed':False,'native_success_criteria_changed':False,
                                   'trials':20,'minimum_successes':18,'early_rejection_after_failures':3,'teacher_inference':False})
tasks=json.loads((run/'continuous-grip-tasks.json').read_text())
for i,t in enumerate(tasks):t['seed']=64000+i
write_json(run/'fast-grip-tasks.json',tasks)
guard=subprocess.Popen(['/usr/bin/caffeinate','-di','-w',str(os.getpid())],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
write_json(run/'fast-grip-process.json',{'pid':os.getpid(),'power_guard_pid':guard.pid})
try:
    proof=run/'fast-grip-cadence-parity.json'
    write_json(run/'fast-grip-status.json',{'status':'validating_cadence'})
    execute([str(root/'.venv-mlx/bin/python'),str(root/'scripts/check_mlx_cadence.py'),'--checkpoint',checkpoint,'--interval','.25','--output',str(proof)],run/'fast-grip-cadence-parity.log')
    output=run/'fast-grip-evaluation'
    command=[str(root/'.venv-mlx/bin/python'),str(root/'scripts/run_mlx_checkpoint.py'),'--checkpoint',checkpoint,
             '--tasks',str(run/'fast-grip-tasks.json'),'--parity',str(run/'continuous-grip-parity.json'),
             '--output',str(output),'--gripper-response','probability','--control-hz','4','--cadence-parity',str(proof),
             '--max-steps','320','--settle-seconds','.05','--stop-after-failures','3']
    write_json(run/'fast-grip-command.json',command);write_json(run/'fast-grip-status.json',{'status':'evaluating'})
    execute(command,run/'fast-grip-evaluation.log')
    r=json.loads((output/'evaluation.json').read_text())
    write_json(run/'fast-grip-status.json',{'status':'qualification_rejected_early' if r.get('stopped_early_after_failures') else 'completed',
                                        'attempts':r['attempts'],'successes':r['successes'],
                                        'qualified_for_promotion':r['attempts']==20 and r['successes']>=18,'ui_checkpoint_changed':False,
                                        'evaluation':str(output/'evaluation.json')})
except BaseException as error:
    write_json(run/'fast-grip-status.json',{'status':'interrupted' if isinstance(error,KeyboardInterrupt) else 'error','error':str(error)})
    raise
finally:
    if guard.poll() is None:guard.terminate();guard.wait(timeout=5)
