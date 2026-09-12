"""Fit the declared recovery dataset and test the frozen candidate in fresh trials."""
import argparse
import json
import os
import shutil
import signal
import subprocess
from pathlib import Path

from fly_brain.assets import canonical_hash, sha256_file, write_json
from fly_brain.visual_dopamine.trajectory_training import load_demonstrations


def execute(command, log_path):
    with log_path.open('w') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                                 env={**os.environ,'OMP_NUM_THREADS':'2','VECLIB_MAXIMUM_THREADS':'2'})
        try:
            code=process.wait()
        except KeyboardInterrupt:
            process.send_signal(signal.SIGINT)
            try: process.wait(timeout=15)
            except subprocess.TimeoutExpired: process.terminate();process.wait(timeout=5)
            raise
    if code:
        raise RuntimeError(f'Command failed with exit {code}: {log_path}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True,type=Path)
    args=parser.parse_args();run=args.plan.resolve().parent;plan=json.loads(args.plan.read_text());root=Path(plan['project'])
    guard=subprocess.Popen(['/usr/bin/caffeinate','-di','-w',str(os.getpid())],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    write_json(run/'process.json',{'pid':os.getpid(),'power_guard_pid':guard.pid})
    try:
        collected=json.loads((Path(plan['dataset'])/'collection-report.json').read_text())['episodes']
        if {r['case_id'] for r in collected}!={c['id'] for c in plan['collection_cases']}:
            raise ValueError('All declared collection cases must finish before training')
        configs=json.loads((root/'reports/two-view-20260910-233652/training-configurations.json').read_text())
        splits={c['id']:c['split'] for c in plan['collection_cases']}
        configs['episode_splits']={Path(r['episode_directory']).name:splits[r['case_id']] for r in collected if r.get('episode_directory')}
        write_json(run/'training-configurations.json',configs)
        examples,episodes=load_demonstrations(plan['dataset'],configs,goal_supervision='aligned',sample_hz=2,grasp_goal_height_mm=27,recovery_target_policy=plan.get('recovery_target_policy','recorded'))
        counts={name:sum(e['split']==name for e in episodes) for name in ['train','validation','test']}
        if counts['train']<4 or counts['validation']<1 or counts['test']<1:
            raise RuntimeError(f'Insufficient verified recovery coverage: {counts}')
        write_json(run/'admitted-recovery.json',{'episodes':episodes,'counts':counts,'frames':len(examples),'all_targets_preserved':all(e['preserve_expert_target'] for e in examples)})
        source=root/'src/fly_brain';digest=canonical_hash({str(p.relative_to(source)):sha256_file(p) for p in source.rglob('*.py')})
        snapshot=root/'data/source-snapshots'/digest
        if not snapshot.exists():shutil.copytree(source,snapshot,ignore=shutil.ignore_patterns('__pycache__'))
        command=json.loads((root/'reports/temporal-control-20260911/train-temporal-command.json').read_text())
        command.insert(command.index('--configurations'),plan['dataset'])
        output=root/'data/checkpoints/rebot'/run.name
        for flag,value in [('--configurations',str(run/'training-configurations.json')),('--output',str(output)),('--warm-start',plan['warm_start']),('--iterations','1200')]:
            command[command.index(flag)+1]=value
        command+=['--recovery-target-policy',plan.get('recovery_target_policy','recorded'),'--sample-weight-policy',plan.get('sample_weight_policy','recorded_stage'),'--policy-visited-weight',str(plan.get('policy_visited_weight',1.)),'--validation-aggregation',plan.get('validation_aggregation','frames')]
        if not plan.get('refit_feature_adaptation',True):command.remove('--refit-feature-adaptation')
        write_json(run/'train-command.json',command)
        training={'source_hash':digest,'source_snapshot':str(snapshot),'checkpoint':str(output),'recovery_counts':counts,'iterations':1200,
                  'new_data_contract':('verified recovery sequences; observable targets with retained local escape/lateral corrections' if plan.get('recovery_target_policy')=='reactive' else 'recorded-recovery-v1; no generic target substitution; native hold required'),
                  'correction_weight':8,'retinal_calibration':'retained','feature_normalization':'train rows only' if plan.get('refit_feature_adaptation',True) else 'retained from warm-start checkpoint','recovery_target_policy':plan.get('recovery_target_policy','recorded')}
        write_json(run/'training-plan.json',training)
        write_json(run/'STATUS.json',{'status':'training',**training})
        execute(command,run/'training.log')
        manifest=json.loads((output/'manifest.json').read_text())
        if manifest['package_source_hash']!=digest:
            raise RuntimeError('Training source changed after snapshot')
        tasks=json.loads((run/'evaluation-tasks.json').read_text())
        if len(tasks)!=20:raise ValueError('Expected 20 fresh tasks')
        evaluation=run/'evaluation'
        command=[str(root/'.venv/bin/fly-brain'),'--home',str(root/'data'),'motor-run','--checkpoint',str(output),
                 '--tasks',str(run/'evaluation-tasks.json'),'--output',str(evaluation),'--sensory-host','local',
                 '--max-steps','160','--settle-seconds','0.1','--record-images']
        if plan.get('evaluation_backend')=='mlx':
            write_json(run/'STATUS.json',{'status':'validating_mlx','checkpoint':str(output),'training_metrics':manifest['metrics']})
            parity=run/'mlx-parity.json'
            execute([str(root/'.venv-mlx/bin/python'),str(root/'scripts/benchmark_mlx_sensory.py'),'--checkpoint',str(output),'--output',str(parity)],run/'mlx-parity.log')
            command=[str(root/'.venv-mlx/bin/python'),str(root/'scripts/run_mlx_checkpoint.py'),'--checkpoint',str(output),
                     '--tasks',str(run/'evaluation-tasks.json'),'--parity',str(parity),'--output',str(evaluation)]
        if plan.get('evaluation_gripper_response'):
            if plan.get('evaluation_backend')!='mlx':raise ValueError('This decoder experiment requires the MLX runtime wrapper')
            command+=['--gripper-response',plan['evaluation_gripper_response']]
        if plan.get('early_rejection'):command+=['--stop-after-failures',str(len(tasks)-plan['qualification']['minimum_successes']+1)]
        write_json(run/'evaluate-command.json',command)
        write_json(run/'STATUS.json',{'status':'evaluating','checkpoint':str(output),'training_metrics':manifest['metrics']})
        execute(command,run/'evaluation.log')
        report=json.loads((evaluation/'evaluation.json').read_text())
        if report.get('execution_blocked'):
            write_json(run/'STATUS.json',{'status':'blocked_environment','reason':report['execution_blocked'],'checkpoint':str(output),'completed_attempts':report['attempts'],'startup_failures':report['startup_failures'],'qualified_for_promotion':None,'ui_checkpoint_changed':False})
            return
        write_json(run/'STATUS.json',{'status':'qualification_rejected_early' if report.get('stopped_early_after_failures') else 'completed','checkpoint':str(output),'attempts':report['attempts'],'successes':report['successes'],
                                     'qualified_for_promotion':report['attempts']==20 and report['successes']>=18,'ui_checkpoint_changed':False,
                                     'training_metrics':manifest['metrics'],'evaluation':str(evaluation/'evaluation.json')})
    except BaseException as error:
        write_json(run/'STATUS.json',{'status':'interrupted' if isinstance(error,KeyboardInterrupt) else 'error','error':str(error)})
        raise
    finally:
        if guard.poll() is None:
            guard.terminate();guard.wait(timeout=5)


if __name__=='__main__':
    main()
