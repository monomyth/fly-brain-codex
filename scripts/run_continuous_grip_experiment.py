"""Run the declared continuous-aperture variant after a binary candidate rejects."""
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from fly_brain.assets import write_json
from train_policy_recovery import execute

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);a=p.parse_args()
plan=json.loads(a.plan.read_text());run=a.plan.resolve().parent;root=Path(__file__).resolve().parents[1]
base=json.loads((run/'STATUS.json').read_text())
if base.get('qualified_for_promotion') is not False:raise ValueError('The binary candidate must finish and reject first')
source=run/'continuous-grip-source';source.mkdir()
for name in ['probability_gripper.py','run_mlx_checkpoint.py','benchmark_mlx_sensory.py','mlx_sensory_backend.py','run_continuous_grip_experiment.py']:
    shutil.copy2(root/'scripts'/name,source/name)
guard=subprocess.Popen(['/usr/bin/caffeinate','-di','-w',str(os.getpid())],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
write_json(run/'continuous-grip-process.json',{'pid':os.getpid(),'power_guard_pid':guard.pid})
try:
    parity=run/'continuous-grip-parity.json'
    write_json(run/'continuous-grip-status.json',{'status':'validating_numerics'})
    execute([str(root/'.venv-mlx/bin/python'),str(root/'scripts/benchmark_mlx_sensory.py'),'--checkpoint',plan['checkpoint'],'--output',str(parity)],run/'continuous-grip-parity.log')
    output=run/'continuous-grip-evaluation'
    command=[str(root/'.venv-mlx/bin/python'),str(root/'scripts/run_mlx_checkpoint.py'),'--checkpoint',plan['checkpoint'],
             '--tasks',plan['tasks'],'--parity',str(parity),'--output',str(output),'--gripper-response','probability',
             '--stop-after-failures',str(plan['qualification']['stop_after_failures'])]
    write_json(run/'continuous-grip-command.json',command)
    write_json(run/'continuous-grip-status.json',{'status':'evaluating'})
    execute(command,run/'continuous-grip-evaluation.log')
    result=json.loads((output/'evaluation.json').read_text())
    write_json(run/'continuous-grip-status.json',{'status':'qualification_rejected_early' if result.get('stopped_early_after_failures') else 'completed',
                                               'attempts':result['attempts'],'successes':result['successes'],
                                               'qualified_for_promotion':result['attempts']==20 and result['successes']>=18,
                                               'ui_checkpoint_changed':False,'evaluation':str(output/'evaluation.json')})
except BaseException as error:
    write_json(run/'continuous-grip-status.json',{'status':'interrupted' if isinstance(error,KeyboardInterrupt) else 'error','error':str(error)})
    raise
finally:
    if guard.poll() is None:guard.terminate();guard.wait(timeout=5)
