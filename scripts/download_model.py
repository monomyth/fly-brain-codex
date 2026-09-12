#!/usr/bin/env python3
"""Download a pinned public release into this project's data directory."""
import argparse, hashlib, json, os, shutil, tarfile, tempfile, urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,indent=2)+'\n')

def fetch(base,entry,cache):
    filename=entry['file']
    if Path(filename).name!=filename: raise ValueError('Invalid release filename')
    target=cache/filename
    if not target.exists() or digest(target)!=entry['sha256']:
        partial=target.with_suffix('.partial')
        try:
            with urllib.request.urlopen(base+'/'+filename,timeout=60) as src,partial.open('wb') as dst: shutil.copyfileobj(src,dst)
            if partial.stat().st_size!=entry['bytes'] or digest(partial)!=entry['sha256']: raise ValueError('Download checksum mismatch: '+filename)
            partial.replace(target)
        finally: partial.unlink(missing_ok=True)
    return target

def extract(archive,destination):
    destination.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar.getmembers():
            path=Path(member.name)
            if path.is_absolute() or '..' in path.parts or not member.isfile(): raise ValueError('Unsafe archive entry')
            target=destination/path
            if not target.resolve().is_relative_to(destination.resolve()): raise ValueError('Archive destination escapes root')
            target.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(member) as src,target.open('wb') as dst: shutil.copyfileobj(src,dst)

def install(release,base,project,profile,activate):
    project=project.resolve(); data=project/'data'; cache=data/'downloads/fly-brain-codex'; cache.mkdir(parents=True,exist_ok=True)
    assets=fetch(base,release['assets'],cache); model=fetch(base,release['profiles'][profile],cache)
    checkpoint=data/'checkpoints/rebot'/('hub-'+profile+'-'+release['version'])
    if checkpoint.exists(): raise FileExistsError(f'Preserving existing checkpoint: {checkpoint}')
    with tempfile.TemporaryDirectory(prefix='release-',dir=cache) as temp:
        stage=Path(temp); extract(assets,stage/'assets'); extract(model,stage/'model')
        info=release['profiles'][profile]; cp=stage/'model/checkpoint'
        if digest(cp/'model.npz')!=info['model_sha256'] or digest(cp/'manifest.json')!=info['manifest_sha256']: raise ValueError('Release model mismatch')
        manifest=json.loads((cp/'manifest.json').read_text())
        for name,sha in manifest['files'].items():
            if Path(name).is_absolute() or '..' in Path(name).parts or digest(cp/name)!=sha: raise ValueError('Invalid checkpoint file')
        # Existing derived assets are reused only when identical; never silently overwritten.
        for source in (stage/'assets').rglob('*'):
            if source.is_file():
                target=data/source.relative_to(stage/'assets')
                if target.exists() and digest(target)!=digest(source): raise ValueError(f'Existing asset differs: {target}')
        for source in (stage/'assets').rglob('*'):
            if source.is_file():
                target=data/source.relative_to(stage/'assets'); target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists(): shutil.copy2(source,target)
        checkpoint.parent.mkdir(parents=True,exist_ok=True); shutil.copytree(cp,checkpoint)
        evidence=data/'releases'/release['version']/profile
        shutil.copytree(stage/'model/evidence',evidence/'historical',dirs_exist_ok=True)
        shutil.copy2(stage/'model/relocation.json',evidence/'relocation.json')
    if activate:
        if profile!='stable': raise ValueError('Experimental model has not passed the UI qualification gate')
        # Rebase only the address and metadata hash; retain original measurements and source-code hashes.
        for filename in ('parity.json','evaluation.json','runtime.json'):
            value=json.loads((evidence/'historical'/filename).read_text())
            if 'checkpoint' in value: value['checkpoint']=str(checkpoint)
            if 'manifest_sha256' in value: value['manifest_sha256']=digest(checkpoint/'manifest.json')
            if filename=='parity.json': value['feature_rows_sha256']=digest(checkpoint/'feature-rows.json')
            value['relocation_note']='Historical measurements, identical numerical arrays; local paths and metadata hashes rebased by release installer. Not a new hardware qualification.'
            write(evidence/filename,value)
        settings={'checkpoint':str(checkpoint),'parity':str(evidence/'parity.json'),'qualification':str(evidence/'evaluation.json')}
        cfg=project/'configs/trained-runtime.json'
        if cfg.exists(): shutil.copy2(cfg,cfg.with_suffix('.json.before-hub'))
        write(cfg,settings)
    print(json.dumps({'checkpoint':str(checkpoint),'activated':activate,'profile':profile,'status':release['profiles'][profile]['status']},indent=2))
    return checkpoint

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',choices=['stable','experimental'],default='stable')
    p.add_argument('--project',type=Path,default=ROOT)
    p.add_argument('--no-activate',action='store_true')
    p.add_argument('--local-release',type=Path,help='Verify a prepared release without network access')
    a=p.parse_args()
    pin=json.loads((ROOT/'configs/model-release.json').read_text())
    base=a.local_release.resolve().as_uri() if a.local_release else 'https://huggingface.co/monomyth/fly-brain-codex/resolve/'+pin['revision']
    with urllib.request.urlopen(base+'/release.json',timeout=30) as f: raw=f.read()
    if hashlib.sha256(raw).hexdigest()!=pin['release_sha256']: raise ValueError('Release manifest differs from source pin')
    install(json.loads(raw),base,a.project,a.profile,not a.no_activate and a.profile=='stable')

if __name__=='__main__':main()
