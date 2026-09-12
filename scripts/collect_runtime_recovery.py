"""Collect longer, explicitly teacher-assisted corrections with the deployed runtime."""
import argparse
import json
import os
import signal
import subprocess
from pathlib import Path
from fly_brain.assets import write_json, sha256_file
from fly_brain.simulator import EpisodeSimulator, screen_locked
from collect_policy_recovery import collect_one
from mlx_sensory_backend import MLXFeedbackCore
from probability_gripper import ProbabilityGripPolicy
from sensory_feature_tap import SensoryFeatureTap


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--limit',type=int)
    args=parser.parse_args();plan=json.loads(args.plan.read_text());report=args.plan.resolve().parent
    root=Path(plan['project']);checkpoint=Path(plan.get('collection_checkpoint',plan['warm_start']));dataset=Path(plan['dataset'])
    if not dataset.resolve().is_relative_to(root/'data/datasets'):
        raise ValueError('Collection data must stay inside this project')
    proof=json.loads(Path(plan.get('collection_parity',plan['parity'])).read_text())
    hashes={'model_sha256':sha256_file(checkpoint/'model.npz'),
            'manifest_sha256':sha256_file(checkpoint/'manifest.json'),
            'backend_source_sha256':sha256_file(Path(__file__).with_name('mlx_sensory_backend.py')),
            'gripper_response_source_sha256':sha256_file(Path(__file__).with_name('probability_gripper.py'))}
    feature_rows=json.loads((checkpoint/'feature-rows.json').read_text())
    if not proof['passed'] or any(proof.get(k)!=v for k,v in hashes.items()) or proof['frames']!=len(feature_rows):
        raise ValueError('Current checkpoint and runtime require complete numerical parity')
    import fcntl
    (root/'.runtime').mkdir(exist_ok=True)
    native_lock=(root/'.runtime/native-qualification.lock').open('a')
    fcntl.flock(native_lock.fileno(),fcntl.LOCK_EX)
    if screen_locked():raise RuntimeError('Unlock the Mac before recovery collection')
    dataset.mkdir(parents=True,exist_ok=True)
    collection=dataset/'collection-report.json'
    results=json.loads(collection.read_text())['episodes'] if collection.exists() else []
    policy=ProbabilityGripPolicy(checkpoint,root/'data')
    policy.core=SensoryFeatureTap(MLXFeedbackCore(policy.core))
    if plan.get('extra_reflex_inputs'):
        import numpy as np
        extra=Path(plan['extra_reflex_inputs']);meta=json.loads((extra/'manifest.json').read_text())
        if not meta['passed'] or meta['source_model_sha256']!=hashes['model_sha256'] or meta['backend_source_sha256']!=hashes['backend_source_sha256']:
            raise ValueError('Supplemental reflex capture must match the collector sensory runtime')
        for name in ['topology.npz','features.npz']:
            if sha256_file(extra/name)!=meta['files'][name]:raise ValueError('Reflex calibration changed')
        with np.load(extra/'topology.npz',allow_pickle=False) as z:indices=z['indices'].copy()
        with np.load(extra/'features.npz',allow_pickle=False) as z:mean=z['mean'].copy();std=z['std'].copy()
        policy.reflex_capture={'indices':indices,'mean':mean,'std':std,'manifest_sha256':sha256_file(extra/'manifest.json')}
    guard=subprocess.Popen(['/usr/bin/caffeinate','-di','-w',str(os.getpid())])
    write_json(report/'collection-process.json',{'pid':os.getpid(),'power_guard_pid':guard.pid})
    try:
        completed=0
        for case in plan['collection_cases']:
            if any(row['case_id']==case['id'] for row in results):continue
            if args.limit is not None and completed>=args.limit:break
            write_json(report/'STATUS.json',{'status':'collecting','case':case['id'],'completed':len(results),
                                           'verified_successes':sum(r['success'] for r in results),'ui_checkpoint_changed':False})
            with EpisodeSimulator(log=report/f"native-{case['id']}.log") as client:
                result=collect_one(client,policy,case,dataset,sha256_file(Path(__file__).with_name('collect_policy_recovery.py')),
                                   max_steps=210,runtime_matched=True)
            results.append(result);completed+=1
            write_json(collection,{'episodes':results,'successes':sum(r['success'] for r in results),
                                   'runtime_profile':'mlx-continuous-grip-2hz','checkpoint_hashes':hashes})
            print(json.dumps(result),flush=True)
            if result.get('interrupted') or 'Unlock the Mac' in result.get('error',''):break
        if sha256_file(checkpoint/'model.npz')!=hashes['model_sha256']:
            raise RuntimeError('The collection actor checkpoint changed')
        write_json(report/'STATUS.json',{'status':'collection_complete' if len(results)==len(plan['collection_cases']) else 'collection_partial',
                                        'completed':len(results),'verified_successes':sum(r['success'] for r in results),
                                        'ui_checkpoint_changed':False})
    finally:
        if guard.poll() is None:guard.terminate();guard.wait(timeout=5)


if __name__=='__main__':main()
