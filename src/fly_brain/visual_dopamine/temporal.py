"""Persistent, timed sensory activity through the same measured neural operator.

This is an engineered rate model. Visual and body states remain separate, and
learned motor/display state never feeds back into the fixed sensory computation.
"""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import tempfile
import numpy as np
from ..assets import canonical_hash,sha256_file
from .feedback import FeedbackCore
from .feedback_cache import calibration_recipe,input_keys

DEFAULT_CONFIG={'kind':'persistent-sensory-v1','nominal_interval_seconds':.5,'maximum_gap_seconds':2.,'base_retention':.35}


def configuration(value=None):
    result={**DEFAULT_CONFIG,**(value or {})}
    if set(result)!=set(DEFAULT_CONFIG) or result['kind']!=DEFAULT_CONFIG['kind']:raise ValueError('Unsupported temporal sensory configuration')
    values=np.asarray([result[k] for k in ('nominal_interval_seconds','maximum_gap_seconds','base_retention')],dtype=float)
    if not np.isfinite(values).all() or not .1<=values[0]<=2 or not values[0]<=values[1]<=5 or not .05<=values[2]<=.95:
        raise ValueError('Invalid temporal sensory time constants')
    return result


def integration_alpha(intervals,config):
    intervals=np.asarray(intervals,dtype=np.float64)
    if not np.isfinite(intervals).all() or (intervals<=0).any() or (intervals>config['maximum_gap_seconds']).any():
        raise ValueError('Temporal observations need increasing times within the maximum gap')
    return (1-np.power(config['base_retention'],intervals/config['nominal_interval_seconds'])).astype(np.float32)


def advance(matrix,state,drive,indices,alpha,updates):
    alpha=np.asarray(alpha,dtype=np.float32).reshape(1,-1)
    for _ in range(updates):
        state=(1-alpha)*state+alpha*np.tanh(.95*(matrix@state))
        state[indices]=drive
    return state


class TemporalFeedbackCore(FeedbackCore):
    def __init__(self,circuit,updates=12,config=None):
        if circuit.feedback_manifest['recipe']['version']!=2:raise ValueError('Temporal sensing requires separated sensory modalities')
        super().__init__(circuit,updates)
        self.temporal_config=configuration(config)
        self._visual_state=np.zeros((len(circuit.ids),1),dtype=np.float32)
        self._body_state=np.zeros_like(self._visual_state)
        self.reset()

    def reset(self):
        self._visual_state.fill(0);self._body_state.fill(0);self.state.fill(0);self.last_time=None

    def response(self,retinal,body,visual_enabled=True,simulation_time=None):
        if simulation_time is None or not np.isfinite(simulation_time):raise ValueError('Temporal sensory input requires simulation time')
        dt=self.temporal_config['nominal_interval_seconds'] if self.last_time is None else simulation_time-self.last_time
        alpha=integration_alpha([dt],self.temporal_config)
        drive=np.clip((np.asarray(retinal,dtype=np.float32)-self.retinal_mean)/self.retinal_std,-1,1)
        body=np.asarray(body,dtype=np.float32)
        if drive.shape!=(len(self.circuit.retina),) or body.shape!=(len(self.circuit.body_indices),) or not np.isfinite(drive).all() or not np.isfinite(body).all():raise ValueError('Invalid temporal sensory input')
        if visual_enabled:
            self._visual_state=advance(self.circuit.sensory,self._visual_state,drive[:,None],self.circuit.retina,alpha,self.updates)
        else:self._visual_state.fill(0)
        self._body_state=advance(self.circuit.sensory,self._body_state,body[:,None],self.circuit.body_indices,alpha,self.updates)
        self.last_time=float(simulation_time)
        self.state=(self._visual_state[:,0]+self._body_state[:,0]).copy()
        self.state[self.circuit.visual_kc]=self._visual_state[self.circuit.visual_kc,0]
        self.state[self.circuit.ascending]=self._body_state[self.circuit.ascending,0]
        return self.state[self.circuit.kc].copy()

    def encode_observation(self,observation,base_directory=None,visual_enabled=True):
        retinal=self.encoder.sample(observation['images'],base_directory);body=self.body_encoder.sample(observation)
        raw=self.response(retinal,body,visual_enabled,observation['simulation_time'])
        encoded=self.encode_responses(raw);self.state[self.circuit.kc]=encoded
        return encoded


class TemporalFeedbackEncoder:
    """Batch independent episodes, preserving order and elapsed time inside each."""
    def __init__(self,core):self.core=core

    def responses(self,retinal,body,examples,batch_size=8,progress=None):
        if len(retinal)!=len(body) or len(retinal)!=len(examples) or not len(examples) or batch_size<1:raise ValueError('Aligned nonempty temporal sequences required')
        groups=OrderedDict()
        for i,e in enumerate(examples):groups.setdefault(e['episode'],[]).append(i)
        sequences=list(groups.values());c=self.core.circuit;raw=np.empty((len(examples),len(c.kc)),dtype=np.float32)
        total=0
        for start in range(0,len(sequences),batch_size):
            chunk=sequences[start:start+batch_size];width=len(chunk)
            visual=np.zeros((len(c.ids),width),dtype=np.float32);bodily=np.zeros_like(visual)
            previous=np.full(width,np.nan)
            for step in range(max(map(len,chunk))):
                active=np.asarray([j for j,seq in enumerate(chunk) if step<len(seq)])
                indices=np.asarray([chunk[j][step] for j in active])
                times=np.asarray([examples[i]['observation']['simulation_time'] for i in indices])
                dt=np.where(np.isnan(previous[active]),self.core.temporal_config['nominal_interval_seconds'],times-previous[active])
                alpha=integration_alpha(dt,self.core.temporal_config)
                drive=np.clip((np.asarray(retinal)[indices]-self.core.retinal_mean)/self.core.retinal_std,-1,1).T
                visual[:,active]=advance(c.sensory,visual[:,active],drive,c.retina,alpha,self.core.updates)
                bodily[:,active]=advance(c.sensory,bodily[:,active],np.asarray(body)[indices].T,c.body_indices,alpha,self.core.updates)
                raw[indices]=np.concatenate([visual[c.visual_kc][:,active].T,bodily[c.ascending][:,active].T],axis=1)
                previous[active]=times;total+=len(active)
                if progress and (step%8==0 or total==len(examples)):progress(total,len(examples))
        return raw


def sequence_responses(core,retinal,body,examples,root,progress=None):
    """Cache complete episode sequences; individual frame caches are not valid here."""
    groups=OrderedDict()
    for i,e in enumerate(examples):groups.setdefault(e['episode'],[]).append(i)
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    recipe={'schema':1,**calibration_recipe(core,'cpu'),'temporal':core.temporal_config,'temporal_source':sha256_file(__file__)}
    hashes=input_keys(retinal,body);raw=np.empty((len(examples),len(core.circuit.kc)),dtype=np.float32)
    records=[];missing=[]
    for episode,indices in groups.items():
        times=[examples[i]['observation']['simulation_time'] for i in indices]
        # Validate even a cached sequence, including its reset boundary.
        integration_alpha([core.temporal_config['nominal_interval_seconds']]+np.diff(times).tolist(),core.temporal_config)
        key=canonical_hash({**recipe,'inputs':[hashes[i].decode('ascii') for i in indices],'times':times})
        path=root/f'{key}.npz';record={'episode':episode,'key':key,'frames':len(indices)};records.append(record)
        if path.exists():
            with np.load(path,allow_pickle=False) as saved:values=saved['raw_responses'].copy()
            if values.shape!=(len(indices),len(core.circuit.kc)) or not np.isfinite(values).all():raise ValueError('Invalid temporal sequence cache')
            raw[indices]=values
        else:missing.append((indices,path))
    if missing:
        encoder=TemporalFeedbackEncoder(core)
        def calculate(indices):
            return encoder.responses(np.asarray(retinal)[indices],np.asarray(body)[indices],[examples[i] for i in indices],batch_size=1)
        executor=ThreadPoolExecutor(max_workers=2,thread_name_prefix='sensory-episode')
        futures={executor.submit(calculate,indices):(indices,path) for indices,path in missing}
        done=0;total=sum(len(indices) for indices,_ in missing)
        try:
            for future in as_completed(futures):
                indices,path=futures[future];raw[indices]=future.result()
                with tempfile.NamedTemporaryFile(dir=root,prefix='.sequence-',suffix='.npz',delete=False) as temp:temporary=Path(temp.name)
                try:np.savez_compressed(temporary,raw_responses=raw[indices]);temporary.replace(path)
                finally:temporary.unlink(missing_ok=True)
                done+=len(indices)
                if progress:progress(done,total)
        finally:executor.shutdown(wait=True,cancel_futures=True)
    return raw,{'kind':'temporal_sequences','workers':2,'sequences':records,'encoded_episodes':len(missing),'reused_episodes':len(groups)-len(missing),'frames':len(examples)}
