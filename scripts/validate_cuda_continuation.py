"""Validate the transferred candidate in the native simulator without changing the UI model."""
import json
import os
import shutil
import signal
import subprocess
from pathlib import Path
from fly_brain.assets import write_json, sha256_file
from fly_brain.simulator import screen_locked

root=Path(__file__).resolve().parents[1]
report=root/'reports/cuda-continuation-20260911'
checkpoint=root/'data/checkpoints/rebot/cuda-continuation-20260911'
for name in ['import-parity.json','mlx-parity.json']:
    proof=json.loads((report/name).read_text())
    if not proof['passed'] or proof['model_sha256']!=sha256_file(checkpoint/'model.npz'):
        raise ValueError('Current checkpoint needs both numerical checks before evaluation')
command=[str(root/'.venv-mlx/bin/python'),str(root/'scripts/run_mlx_checkpoint.py'),'--checkpoint',str(checkpoint),
         '--tasks',str(report/'evaluation-tasks.json'),'--parity',str(report/'mlx-parity.json'),
         '--output',str(report/'evaluation'),'--gripper-response','probability','--stop-after-failures','3']
write_json(report/'evaluate-command.json',command)
guard=subprocess.Popen(['/usr/bin/caffeinate','-di','-w',str(os.getpid())])
write_json(report/'validation-process.json',{'pid':os.getpid(),'power_guard_pid':guard.pid})
process=None
try:
    if screen_locked():
        write_json(report/'STATUS.json',{'status':'blocked_environment','reason':'screen_locked','ui_checkpoint_changed':False})
        raise RuntimeError('Screen is locked; no trial was started')
    write_json(report/'STATUS.json',{'status':'evaluating','checkpoint':str(checkpoint),'ui_checkpoint_changed':False})
    with (report/'evaluation.log').open('w') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                                 env={**os.environ,'OMP_NUM_THREADS':'2','VECLIB_MAXIMUM_THREADS':'2'})
        code=process.wait()
    if code:raise RuntimeError(f'Evaluation exited {code}; see evaluation.log')
    results=json.loads((report/'evaluation/evaluation.json').read_text())
    write_json(report/'STATUS.json',{'status':'blocked_environment' if results.get('execution_blocked') else 'qualified' if results['attempts']==20 and results['successes']>=18 else 'qualification_rejected',
               'checkpoint':str(checkpoint),'attempts':results['attempts'],'successes':results['successes'],
               'startup_failures':results['startup_failures'],'execution_blocked':results.get('execution_blocked'),
               'qualified_for_promotion':results['attempts']==20 and results['successes']>=18,'ui_checkpoint_changed':False})
    print((report/'STATUS.json').read_text(),flush=True)
except BaseException as error:
    if process is not None and process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:process.wait(timeout=15)
        except subprocess.TimeoutExpired:process.terminate();process.wait(timeout=5)
    raise
finally:
    if guard.poll() is None:guard.terminate();guard.wait(timeout=5)
