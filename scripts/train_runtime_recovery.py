"""Prepare captured recovery data, fit on Blacktower, and qualify on the Mac."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
from fly_brain.assets import write_json,sha256_file


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text());root=Path(plan['project']);report=args.plan.resolve().parent
    inputs=root/'data/checkpoints/rebot'/(report.name+'-inputs')
    checkpoint=root/'data/checkpoints/rebot'/report.name
    bundle=root/'data/exports'/('blacktower-'+report.name)
    remote_root='/home/monomyth/code/codex/fly-brain';remote=remote_root+'/experiments/'+report.name
    iterations=int(plan.get('training_iterations',6000))
    if not 1<=iterations<=20000:raise ValueError('Invalid training iteration count')
    remote_fit=remote+'/fit-'+str(iterations)
    guard=subprocess.Popen(['/usr/bin/caffeinate','-di','-w',str(os.getpid())])
    write_json(report/'training-process.json',{'pid':os.getpid(),'power_guard_pid':guard.pid})
    def status(value,**extra):write_json(report/'STATUS.json',{'status':value,'ui_checkpoint_changed':False,**extra})
    def execute(command,label):
        write_json(report/(label+'-command.json'),command)
        with (report/(label+'.log')).open('w') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=root,
                                     env={**os.environ,'OMP_NUM_THREADS':'2','VECLIB_MAXIMUM_THREADS':'2'})
            try:code=process.wait()
            except BaseException:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                    try:process.wait(timeout=15)
                    except subprocess.TimeoutExpired:process.terminate();process.wait(timeout=5)
                raise
        if code:raise RuntimeError(f'{label} failed with exit {code}; see {report/(label+".log")}')
    try:
        status('preparing_inputs')
        execute([str(root/'.venv/bin/python'),str(root/'scripts/prepare_runtime_recovery.py'),'--plan',str(args.plan.resolve()),'--output',str(inputs)],'prepare')
        status('exporting_training_bundle')
        execute([str(root/'.venv/bin/python'),str(root/'scripts/export_head_training_bundle.py'),'--checkpoint',str(inputs),
                 '--configurations',str(report/'training-configurations.json'),'--output',str(bundle),'--held-orientation-objective',plan.get('held_orientation_objective','full_rotation'),'--pregrasp-retention-strength',str(plan.get('pregrasp_retention_strength',0)),'--retention-selection-group',plan.get('retention_selection_group','all'),'--retention-training-group',plan.get('retention_training_group','all'),'--retention-joint-rmse-limit',str(plan.get('retention_joint_rmse_limit',.1)),'--retention-joint-max-limit',str(plan.get('retention_joint_max_limit',.5)),'--retention-aperture-limit',str(plan.get('retention_aperture_limit',1.)),'--held-position-objective',plan.get('held_position_objective','tool_position')],'export')
        shutil.copy2(root/'scripts/continue_head_training.py',bundle/'continue_head_training.py')
        execute(['ssh','-o','ConnectTimeout=8','blacktower.local','mkdir -p '+shlex.quote(remote)],'remote-directory')
        execute(['rsync','-a',str(bundle)+'/', 'blacktower.local:'+remote+'/'],'upload')
        python=remote_root+'/.venv-cuda/bin/python'
        status('checking_cuda_math')
        execute(['ssh','-o','ConnectTimeout=8','blacktower.local',shlex.join([python,remote+'/benchmark_training_bundle.py'])],'cuda-benchmark')
        status('training_on_blacktower',iterations=iterations)
        execute(['ssh','-o','ConnectTimeout=8','blacktower.local',shlex.join([python,remote+'/continue_head_training.py','--iterations',str(iterations),'--output',remote_fit])],'cuda-training')
        execute(['rsync','-a','blacktower.local:'+remote_fit+'/',str(report/'remote-result')+'/'],'download')
        execute(['rsync','-a','blacktower.local:'+remote+'/benchmark.json',str(report/'benchmark.json')],'download-benchmark')
        status('importing_checkpoint')
        execute([str(root/'.venv/bin/python'),str(root/'scripts/import_continued_head.py'),'--source',str(inputs),'--result',str(report/'remote-result'),'--output',str(checkpoint)],'import')
        import numpy as np
        with np.load(checkpoint/'model.npz',allow_pickle=False) as fitted,np.load(Path(plan['warm_start'])/'model.npz',allow_pickle=False) as original:
            unchanged=set(fitted.files)==set(original.files) and all(np.array_equal(fitted[key],original[key]) for key in original.files)
        if unchanged:
            status('unchanged_model',checkpoint=str(checkpoint),qualified_for_promotion=False,reason='The fit selected the original parameters; no new controller to qualify')
            return
        execute([str(root/'.venv/bin/python'),str(root/'scripts/check_imported_head.py'),'--bundle',str(bundle),'--result',str(report/'remote-result'),
                 '--checkpoint',str(checkpoint),'--output',str(report/'import-parity.json')],'import-parity')
        status('checking_sensory_replay')
        if plan.get('sensory_parity_source'):
            execute([str(root/'.venv-mlx/bin/python'),str(root/'scripts/check_cached_mlx_head.py'),'--checkpoint',str(checkpoint),'--source-proof',plan['sensory_parity_source'],'--output',str(report/'mlx-parity.json')],'mlx-parity')
        else:
            execute([str(root/'.venv-mlx/bin/python'),str(root/'scripts/benchmark_mlx_sensory.py'),'--checkpoint',str(checkpoint),'--output',str(report/'mlx-parity.json')],'mlx-parity')
        status('evaluating',checkpoint=str(checkpoint))
        execute([str(root/'.venv-mlx/bin/python'),str(root/'scripts/run_mlx_checkpoint.py'),'--checkpoint',str(checkpoint),'--tasks',str(report/'evaluation-tasks.json'),
                 '--parity',str(report/'mlx-parity.json'),'--output',str(report/'evaluation'),'--gripper-response','probability','--stop-after-failures','3'],'evaluation')
        result=json.loads((report/'evaluation/evaluation.json').read_text())
        qualified=result['attempts']==20 and result['successes']>=18
        status('blocked_environment' if result.get('execution_blocked') else 'qualified' if qualified else 'qualification_rejected',
               checkpoint=str(checkpoint),attempts=result['attempts'],successes=result['successes'],startup_failures=result['startup_failures'],
               execution_blocked=result.get('execution_blocked'),qualified_for_promotion=qualified)
        print((report/'STATUS.json').read_text(),flush=True)
    except BaseException as error:
        status('interrupted' if isinstance(error,KeyboardInterrupt) else 'error',error=str(error))
        raise
    finally:
        if guard.poll() is None:guard.terminate();guard.wait(timeout=5)


if __name__=='__main__':main()
