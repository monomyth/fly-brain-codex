"""Attach the numerically verified MLX controller to a user-owned simulator window."""
import argparse
import json
import os
from pathlib import Path
from fly_brain.assets import checked_files,sha256_file,write_json

ROOT=Path(__file__).resolve().parents[1]


def validate_deployment(settings,checkpoint=None):
    checkpoint=Path(checkpoint or settings['checkpoint']).resolve()
    proof_path=Path(settings['parity']);qualification_path=Path(settings['qualification'])
    proof=json.loads(proof_path.read_text());qualification=json.loads(qualification_path.read_text())
    manifest=json.loads((checkpoint/'manifest.json').read_text());checked_files(checkpoint,manifest)
    rows=json.loads((checkpoint/'feature-rows.json').read_text())
    hashes={'model_sha256':sha256_file(checkpoint/'model.npz'),'manifest_sha256':sha256_file(checkpoint/'manifest.json'),
            'backend_source_sha256':sha256_file(ROOT/'scripts/mlx_sensory_backend.py'),
            'gripper_response_source_sha256':sha256_file(ROOT/'scripts/probability_gripper.py')}
    if not proof['passed'] or proof.get('checkpoint')!=str(checkpoint) or proof['frames']!=len(rows) or proof['episodes']!=len({r['episode'] for r in rows}) or any(proof.get(k)!=v for k,v in hashes.items()):
        raise ValueError('The selected checkpoint needs matching complete MLX parity results')
    from touch_gate_validation import validate_touch_gate
    validate_touch_gate(checkpoint,proof,manifest)
    runtime=json.loads((qualification_path.parent/'runtime.json').read_text())
    if runtime.get('motor_filter') is not None:raise ValueError('Experimental motor filtering is not qualified for this UI adapter')
    if any(runtime.get(k)!=v for k,v in hashes.items()) or runtime.get('gripper_response')!='probability' or runtime.get('learn') is not False:
        raise ValueError('Qualification used a different controller runtime')
    if runtime.get('max_steps')!=160 or runtime.get('settle_seconds')!=.1 or runtime.get('control_hz_override') not in (None,2.) or manifest.get('architecture',{}).get('control_hz')!=2.:
        raise ValueError('Qualification used a different timing profile')
    if qualification.get('checkpoint')!=str(checkpoint) or qualification.get('learning_enabled') is not False or qualification.get('attempts')!=20 or len(qualification.get('episodes',[]))!=20 or qualification.get('startup_failures',0) or qualification.get('interrupted_attempts',0):
        raise ValueError('A completed 20-trial qualification is required before UI deployment')
    valid=0
    for episode in qualification['episodes']:
        task=episode.get('task_for_evaluator_only',{});result=episode.get('final_evaluation',{})
        if (episode.get('success') and episode.get('trial_started') and not episode.get('error') and result.get('success') and result.get('held')
                and task.get('cube_size_mm')==20 and task.get('initial_joints_deg')==[0]*6 and task.get('initial_gripper_mm')==0
                and result.get('hold_seconds',0)>=5 and result.get('clearance_mm',0)>=100 and result.get('tilt_deg',180)<=5):
            valid+=1
    if valid<18:raise ValueError(f'UI deployment requires 18/20 verified holds; found {valid}/20')
    return {'checkpoint':str(checkpoint),'qualification_successes':valid,'qualification_attempts':20,
            'parity':str(proof_path.resolve()),'qualification':str(qualification_path.resolve()),'max_steps':160,'settle_seconds':.1,'control_hz':2.,**hashes}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',nargs='?',choices=['motor-run'])
    p.add_argument('--home',type=Path,default=ROOT/'data')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--deployment',type=Path,default=Path(os.environ.get('FLY_BRAIN_DEPLOYMENT',ROOT/'configs/trained-runtime.json')))
    p.add_argument('--check-deployment',action='store_true')
    p.add_argument('--current-episode',action='store_true')
    p.add_argument('--episode-id')
    p.add_argument('--control-directory',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--record-images',action='store_true')
    p.add_argument('--floor-projection',action='store_true')
    p.add_argument('--max-steps',type=int)
    p.add_argument('--learn',action='store_true')
    a=p.parse_args();settings=json.loads(a.deployment.read_text());verified=validate_deployment(settings,a.checkpoint)
    if a.check_deployment:
        print(json.dumps(verified,indent=2));return
    if a.command!='motor-run' or not a.current_episode or not a.episode_id or a.control_directory is None or a.output is None:
        p.error('This UI adapter requires motor-run, a current episode ID, control directory, and output directory')
    if not a.output.resolve().is_relative_to(ROOT):p.error('UI records must stay inside this project')
    if a.max_steps is not None and a.max_steps!=verified['max_steps']:p.error('Use the qualified 160-decision runtime')
    from probability_gripper import ProbabilityGripPolicy
    from mlx_sensory_backend import MLXFeedbackCore
    from fly_brain.simulator import Simulator
    from fly_brain.visual_dopamine.motor_run import run
    policy=ProbabilityGripPolicy(verified['checkpoint'],a.home);policy.core=MLXFeedbackCore(policy.core)
    a.output.mkdir(parents=True,exist_ok=True)
    write_json(a.output/'runtime.json',{'backend':'MLX Metal','gripper_response':'probability','settle_seconds':.1,
               'control_hz':policy.hz,'learn':a.learn,'entrypoint':'existing native UI','runner_source_sha256':sha256_file(__file__),**verified})
    with Simulator(a.control_directory) as client:
        result=run(client,policy,[{}],a.output,current_episode=True,expected_episode_id=a.episode_id,max_steps=verified['max_steps'],learn=a.learn,settle_seconds=verified['settle_seconds'])
    if sha256_file(Path(verified['checkpoint'])/'model.npz')!=verified['model_sha256']:
        raise RuntimeError('The original checkpoint changed during the UI run')
    print(json.dumps({'attempts':result['attempts'],'successes':result['successes'],'startup_failures':result['startup_failures']}),flush=True)
    if result.get('interrupted_attempts'):raise SystemExit(130)


if __name__=='__main__':main()
