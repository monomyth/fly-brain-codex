"""Compare all declared stage variants, then run a predeclared 20-task battery."""
import json,os,signal,subprocess
from pathlib import Path
from fly_brain.assets import write_json,canonical_hash,sha256_file

root=Path(__file__).resolve().parents[1];run=root/'reports/temporal-control-20260911'
plan=json.loads((run/'comparison-plan.json').read_text())
child=None
guard=subprocess.Popen(['/usr/bin/caffeinate','-di','-w',str(os.getpid())],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
write_json(run/'evaluation-process.json',{'pid':os.getpid(),'power_guard_pid':guard.pid})

def execute(command,log_path):
    global child
    with log_path.open('w') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        try:code=child.wait()
        except KeyboardInterrupt:
            child.send_signal(signal.SIGINT)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.terminate();child.wait(timeout=5)
            raise
        finally:child=None
    if code:raise RuntimeError(f'Command failed with exit {code}; inspect {log_path.name}')

try:
    evaluation_source=canonical_hash({str(p.relative_to(root/'src/fly_brain')):sha256_file(p) for p in (root/'src/fly_brain').rglob('*.py')})
    write_json(run/'evaluation-plan.json',{'training_source_hash':plan['source_hash'],'evaluation_source_hash':evaluation_source,
        'variants':['static','temporal','ltm'],'stage_max_steps':50,'selection_rule':'Highest weaker-stage success rate, then summed stage success rates, then lowest validation tool-position RMSE',
        'full_task_file':str(run/'full-tasks-20.json'),'full_max_steps':160,'settle_seconds':.1,'promotion_minimum_successes':18,
        'setup':'All fixtures start folded; conventional setup earns no hold credit; temporal warmup sampled at nominal cadence while physics is paused'})
    outcomes=[]
    for name in ['static','temporal','ltm']:
        checkpoint=Path(plan['variants'][name]['checkpoint']);meta=json.loads((checkpoint/'manifest.json').read_text())
        if meta['package_source_hash']!=plan['source_hash']:raise RuntimeError('Training source snapshot mismatch: '+name)
        output=run/f'stages-{name}-matched'
        execute([str(root/'.venv/bin/python'),str(root/'scripts/run_stage_diagnostics.py'),'--checkpoint',str(checkpoint),'--output',str(output),'--max-steps','50'],run/f'stages-{name}-matched.log')
        report=json.loads((output/'stages.json').read_text())
        if len(report['results'])!=6:raise RuntimeError('Stage battery did not complete: '+name)
        if any('Unlock the Mac' in str(r.get('error','')) for r in report['results']):raise RuntimeError('Unlock the Mac before continuing physical validation')
        stages={stage:{'attempts':sum(r['stage']==stage for r in report['results']),'successes':sum(r['stage']==stage and r['stage_success'] for r in report['results'])} for stage in ['grasp','lift_hold']}
        rates=[v['successes']/v['attempts'] for v in stages.values()]
        outcome={'variant':name,'checkpoint':str(checkpoint),'stages':stages,'rank':[min(rates),sum(rates),-meta['metrics']['validation']['goal_position_rmse_mm']]}
        outcomes.append(outcome);write_json(run/'stage-comparison.json',{'results':outcomes});print(outcome,flush=True)
        execute([str(root/'.venv/bin/python'),str(root/'scripts/build_stage_gallery.py'),str(output)],run/f'gallery-{name}.log')
    selected=max(outcomes,key=lambda r:r['rank']);write_json(run/'selected-candidate.json',selected)
    output=run/f'full-{selected["variant"]}-20'
    command=[str(root/'.venv/bin/fly-brain'),'--home',str(root/'data'),'motor-run','--checkpoint',selected['checkpoint'],'--tasks',str(run/'full-tasks-20.json'),'--output',str(output),'--sensory-host','local','--max-steps','160','--settle-seconds','0.1','--record-images']
    write_json(run/'full-evaluation-command.json',command)
    execute(command,run/'full-evaluation.log')
    report=json.loads((output/'evaluation.json').read_text())
    finished={'status':'completed','selected_variant':selected['variant'],'evaluation':str(output/'evaluation.json'),'attempts':report['attempts'],'successes':report['successes'],'promotion_target_met':report['attempts']==20 and report['successes']>=18}
    write_json(run/'comparison-result.json',finished);print(finished,flush=True)
except BaseException as error:
    write_json(run/'comparison-result.json',{'status':'interrupted' if isinstance(error,KeyboardInterrupt) else 'error','error':str(error)})
    raise
finally:
    if guard.poll() is None:guard.terminate();guard.wait(timeout=5)
