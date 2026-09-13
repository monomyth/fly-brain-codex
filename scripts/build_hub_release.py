"""Package immutable arrays and portable provenance for the public model release."""
import argparse, hashlib, json, re, shutil, tarfile
from pathlib import Path
from fly_brain.assets import checked_files, sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]

from fly_brain.publication import portable as portable_metadata, anonymous_tar_member
PRIVATE_CONFIG = ROOT/'data/publication-private.json'


def portable(value):
    settings=json.loads(PRIVATE_CONFIG.read_text()) if PRIVATE_CONFIG.exists() else {}
    return portable_metadata(value, ROOT, settings.get('redact_strings', []))


def archive(source, target):
    with tarfile.open(target, 'w:gz', compresslevel=6) as tar:
        for f in sorted(source.rglob('*')):
            if f.is_file() and '__pycache__' not in f.parts and '.cache' not in f.parts and f.suffix not in ('.pyc','.pyo') and not f.name.startswith('.env'):
                tar.add(f, arcname=f.relative_to(source), recursive=False, filter=anonymous_tar_member)
    return {'file': target.name, 'bytes': target.stat().st_size, 'sha256': sha256_file(target)}

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True); p.add_argument('--training-bundle',type=Path,required=True); a=p.parse_args()
    out=a.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    stage=out/'staging'; stage.mkdir(exist_ok=True)
    stable=ROOT/'data/checkpoints/rebot/retain-grasp-20260911'
    meta=json.loads((stable/'manifest.json').read_text())
    assets=stage/'assets'; assets.mkdir(exist_ok=True)
    identities={'prepared':'graph_id','circuits':'circuit_id','motor-circuits':'motor_circuit_id','feedback-circuits':'feedback_circuit_id','visualizations':'graph_id'}
    for directory,key in identities.items():
        src=ROOT/'data'/directory/meta[key]
        if (src/'manifest.json').exists(): checked_files(src,json.loads((src/'manifest.json').read_text()))
        shutil.copytree(src, assets/directory/meta[key],dirs_exist_ok=True)
    release={'schema':'fly-brain-hub-release-v1','version':'2026-09-12','repo':'monomyth/fly-brain-codex',
             'license':'cc-by-4.0','source':'https://github.com/monomyth/fly-brain-codex',
             'assets':archive(assets,out/'runtime-assets.tar.gz'),'profiles':{}}
    for profile,name in [('stable','retain-grasp-20260911'),('experimental','touch-direct-20260912')]:
        src=ROOT/'data/checkpoints/rebot'/name; original=json.loads((src/'manifest.json').read_text()); checked_files(src,original)
        dest=stage/profile/'checkpoint'; dest.mkdir(parents=True,exist_ok=True)
        changes={}
        for filename in original['files']:
            f=src/filename; target=dest/filename
            if f.suffix=='.json': write_json(target,portable(json.loads(f.read_text())))
            else: shutil.copy2(f,target)
            changes[filename]={'original_sha256':sha256_file(f),'release_sha256':sha256_file(target)}
            if f.suffix in ('.npz','.npy'): assert changes[filename]['original_sha256']==changes[filename]['release_sha256']
        manifest=portable(original); manifest['files']={f:sha256_file(dest/f) for f in original['files']}
        write_json(dest/'manifest.json',manifest); checked_files(dest,manifest)
        write_json(stage/profile/'relocation.json',{'description':'Only JSON provenance paths were normalized. All numerical arrays are byte-identical. Historical evaluation is preserved; relocation is not a new trial.',
            'original_manifest_sha256':sha256_file(src/'manifest.json'),'release_manifest_sha256':sha256_file(dest/'manifest.json'),'files':changes})
        evidence=stage/profile/'evidence'; evidence.mkdir(exist_ok=True)
        base=ROOT/'reports'/('retain-grasp-20260911' if profile=='stable' else 'direct-pickup-20260912')
        for source,filename in [(base/'evaluation/evaluation.json','evaluation.json'),(base/'evaluation/runtime.json','runtime.json')]+([(base/'mlx-parity.json','parity.json')] if profile=='stable' else []):
            write_json(evidence/filename,portable(json.loads(source.read_text())))
        if profile=='stable':
            source=ROOT/'reports/normal-ui-startup-20260912/normal-launch-test.json'
            if source.exists(): write_json(evidence/'normal-ui-launch.json',portable(json.loads(source.read_text())))
        release['profiles'][profile]={**archive(stage/profile,out/(profile+'.tar.gz')),'checkpoint_name':name,
            'model_sha256':sha256_file(dest/'model.npz'),'manifest_sha256':sha256_file(dest/'manifest.json'),
            'status':'19/20 pickup and hold in a narrow region; separate UI drop check passed' if profile=='stable' else '12/15 complete tasks; 5 planned trials unrun; not UI qualified'}
    # Keep the newest GPU fit inspectable, without presenting it as a validated controller.
    continuation=stage/'cuda-continuation'; continuation.mkdir(exist_ok=True)
    for f in (ROOT/'reports/direct-pickup-20260912/cuda-result').iterdir():
        if f.suffix=='.npz': shutil.copy2(f,continuation/f.name)
        elif f.suffix=='.json' and f.name!='manifest.json': write_json(continuation/f.name,portable(json.loads(f.read_text())))
    write_json(continuation/'manifest.json',{'status':'Unqualified raw 12000-iteration CUDA continuation; not deployed','files':{f.name:sha256_file(f) for f in continuation.iterdir() if f.name!='manifest.json'}})
    release['unqualified_cuda_continuation']=archive(continuation,out/'cuda-continuation.tar.gz')
    training = stage/'training-bundle'
    shutil.copytree(a.training_bundle, training, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__','*.pyc','.cache'))
    shutil.copy2(ROOT/'scripts/continue_head_training.py',training/'continue_head_training.py')
    shutil.copy2(ROOT/'src/fly_brain/visual_dopamine/remote_features.py',training/'src/fly_brain/visual_dopamine/remote_features.py')
    for f in training.rglob('*.json'):
        if f.name != 'manifest.json': write_json(f,portable(json.loads(f.read_text())))
    write_json(training/'manifest.json',{'schema':'frozen-sensory-head-training-v1',
        'description':'Portable frozen-feature continuation bundle; benchmark again on the target GPU before fitting.',
        'files':{str(f.relative_to(training)):sha256_file(f) for f in training.rglob('*') if f.is_file() and f.name not in ('manifest.json','benchmark.json')}})
    release['training_bundle']=archive(training,out/'training-bundle.tar.gz')
    write_json(out/'release.json',release)
    print(json.dumps(release,indent=2))

if __name__=='__main__': main()
