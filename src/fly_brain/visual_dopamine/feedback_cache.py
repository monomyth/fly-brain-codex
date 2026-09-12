"""Reuse exact sensory responses across datasets with the same calibration."""
import hashlib
from pathlib import Path
import tempfile
import numpy as np
from ..assets import canonical_hash,locked,sha256_file


def calibration_recipe(core,device):
    return {'circuit':core.circuit.feedback_manifest['feedback_circuit_id'],'updates':core.updates,'device':device,
            'source':{name:sha256_file(Path(__file__).with_name(name)) for name in ['feedback.py','core.py']},
            'normalization':{name:canonical_hash(getattr(core,name).tolist()) for name in ['retinal_mean','retinal_std']}}


def input_keys(retinal,body):
    retinal=np.ascontiguousarray(retinal,dtype=np.float32);body=np.ascontiguousarray(body,dtype=np.float32)
    if retinal.ndim!=2 or body.ndim!=2 or len(retinal)!=len(body) or not len(retinal):raise ValueError('Aligned sensory matrices required')
    if not np.isfinite(retinal).all() or not np.isfinite(body).all():raise ValueError('Finite sensory matrices required')
    shape=np.asarray([retinal.shape[1],body.shape[1]],dtype=np.int64).tobytes()
    return np.asarray([hashlib.sha256(shape+r.tobytes()+b.tobytes()).hexdigest() for r,b in zip(retinal,body)],dtype='S64')


def cached_responses(path,retinal,body,output_width,encode,precomputed=None):
    """The caller keys path by circuit, fixed operator, calibration and device."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    keys=input_keys(retinal,body)
    def valid(value,count):
        value=np.asarray(value,dtype=np.float32)
        if value.shape!=(count,output_width) or not np.isfinite(value).all():raise ValueError('Invalid sensory response cache')
        return value
    with locked(path.with_suffix('.lock')):
        known={}
        if path.exists():
            with np.load(path,allow_pickle=False) as saved:
                old_keys=saved['input_keys'];raw=valid(saved['raw_responses'],len(old_keys))
            if old_keys.ndim!=1 or old_keys.dtype.kind!='S' or len(set(old_keys))!=len(old_keys):raise ValueError('Invalid sensory cache keys')
            known=dict(zip(old_keys,raw))
        reused=sum(key in known for key in keys)
        if precomputed is not None:
            raw=valid(precomputed,len(keys))
            for key,value in zip(keys,raw):
                if key in known and not np.array_equal(known[key],value):raise ValueError('Conflicting responses for identical sensory input')
                known[key]=value
        missing={}
        for i,key in enumerate(keys):
            if key not in known:missing.setdefault(key,i)
        if missing:
            indices=list(missing.values())
            raw=valid(encode(np.asarray(retinal)[indices],np.asarray(body)[indices]),len(indices))
            known.update(zip(missing,raw))
        if missing or precomputed is not None:
            with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.responses-',suffix='.npz',delete=False) as temp:
                temporary=Path(temp.name)
            try:
                np.savez_compressed(temporary,input_keys=np.asarray(list(known),dtype='S64'),raw_responses=np.asarray(list(known.values())))
                temporary.replace(path)
            finally:temporary.unlink(missing_ok=True)
        return np.asarray([known[key] for key in keys]),{'reused_rows':reused,'encoded_unique_rows':len(missing),'rows':len(keys)}
