"""Ephemeral SSH/CUDA sensory worker; arm actions and learned motor decoding stay local."""
import argparse,base64,json,os,re,select,shlex,subprocess,sys,time,uuid,zlib
from pathlib import Path
import numpy as np
from ..assets import canonical_hash,home


def calibration_identity(core):
    return canonical_hash({'circuit':core.circuit.feedback_manifest['feedback_circuit_id'],'updates':core.updates,
                           **{k:getattr(core,k).tolist() for k in ['retinal_mean','retinal_std','kc_mean','kc_std']}})


class RemoteFeedbackCore:
    def __init__(self,core,host,project='/home/monomyth/code/codex/fly-brain',checkpoint='data/benchmark-bundle/checkpoint',root=None):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.@-]*',host):raise ValueError('Invalid sensory worker SSH host')
        self.local=core;self.state=core.state;self.request_id=0;self.closed=False
        logs=home(root)/'runs/sensory-workers';logs.mkdir(parents=True,exist_ok=True)
        self.log=(logs/(str(uuid.uuid4())+'.log')).open('w')
        argv=['.venv-cuda/bin/python','-m','fly_brain.visual_dopamine.remote_features','--serve','--checkpoint',checkpoint,'--root','data/benchmark-bundle']
        command='cd '+shlex.quote(project)+' && PYTHONPATH=src XDG_CACHE_HOME='+shlex.quote(project+'/.cache')+' CUDA_CACHE_PATH='+shlex.quote(project+'/.cache/cuda')+' '+shlex.join(argv)
        self.process=subprocess.Popen(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=8',host,command],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1)
        try:
            ready=self._read(40)
            if ready.get('feedback_circuit_id')!=core.circuit.feedback_manifest['feedback_circuit_id']:raise ValueError('Remote sensory circuit does not match this checkpoint')
            self.process.stdin.write(json.dumps({'updates':core.updates,**{k:getattr(core,k).tolist() for k in ['retinal_mean','retinal_std','kc_mean','kc_std']}},allow_nan=False)+'\n');self.process.stdin.flush()
            calibrated=self._read(10)
            if calibrated.get('calibration_identity')!=calibration_identity(core):raise ValueError('Remote sensory calibration does not match this checkpoint')
        except BaseException:self.close();raise

    def __getattr__(self,key):return getattr(self.local,key)

    def _read(self,timeout):
        if not select.select([self.process.stdout],[],[],timeout)[0]:raise TimeoutError('Remote sensory worker timed out')
        line=self.process.stdout.readline(8_000_001)
        if not line or len(line)>8_000_000:raise RuntimeError('Remote sensory worker disconnected or returned an oversized result')
        value=json.loads(line)
        if 'error' in value:raise RuntimeError(value['error'])
        return value

    def encode_observation(self,observation,base_directory=None,visual_enabled=True):
        if not visual_enabled:raise ValueError('Use the CPU core for visual ablation')
        retinal=self.encoder.sample(observation['images'],base_directory);body=self.body_encoder.sample(observation)
        self.request_id+=1
        self.process.stdin.write(json.dumps({'id':self.request_id,'retinal':retinal.tolist(),'body':body.tolist()},allow_nan=False)+'\n');self.process.stdin.flush()
        result=self._read(5)
        if result.get('id')!=self.request_id:raise RuntimeError('Mismatched remote observation')
        payload=base64.b64decode(result['display_activity'],validate=True)
        decoder=zlib.decompressobj();raw=decoder.decompress(payload,len(self.circuit.ids)+1)
        if not decoder.eof or len(raw)!=len(self.circuit.ids):raise ValueError('Invalid neural state length')
        state=np.frombuffer(raw,dtype=np.uint8).astype(np.float32)/255;features=np.asarray(result['features'],dtype=np.float32)
        if not np.isfinite(state).all() or features.shape!=(len(self.circuit.kc),) or not np.isfinite(features).all():raise ValueError('Invalid remote neural activity')
        # These values feed display telemetry only; control features stay lossless.
        self.state[:]=state;self.state[self.circuit.kc]=features
        return features

    def close(self):
        if self.closed:return
        self.closed=True
        if hasattr(self,'process'):
            if self.process.stdin:self.process.stdin.close()
            try:self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:self.process.terminate();self.process.wait(timeout=3)
            if self.process.stdout:self.process.stdout.close()
        self.log.close()


def attach(policy,host,project='/home/monomyth/code/codex/fly-brain',checkpoint='data/benchmark-bundle/checkpoint'):
    if not policy.metadata.get('feedback_circuit_id'):raise ValueError('Remote sensory computation requires a camera/body checkpoint')
    policy.core=RemoteFeedbackCore(policy.core,host,project,checkpoint,policy.root)
    return policy


def configure_runtime(policy,host=None):
    settings_path=policy.root/'runtime.json'
    settings=json.loads(settings_path.read_text()) if settings_path.exists() else {}
    selected=host or os.environ.get('FLY_BRAIN_SENSORY_HOST') or settings.get('sensory_host')
    if not selected or selected=='local':return policy
    if policy.metadata.get('temporal_sensory'):raise ValueError('Temporal sensory runtime currently requires --sensory-host local; remote stateless encoding is incompatible')
    if not policy.metadata.get('feedback_circuit_id'):
        if host:raise ValueError('This checkpoint does not support CUDA sensory offload')
        return policy
    print(f'Sensory graph: {selected} over SSH/CUDA; motor decoding and simulator stay local',flush=True)
    return attach(policy,selected,settings.get('sensory_project','/home/monomyth/code/codex/fly-brain'),settings.get('sensory_checkpoint','data/benchmark-bundle/checkpoint'))


def close_runtime(policy):
    core=getattr(policy,'core',None)
    if isinstance(core,RemoteFeedbackCore):core.close()


def serve(checkpoint,root):
    from .motor_policy import MotorPolicy
    from .feedback import TorchFeedbackEncoder
    import torch
    torch.set_num_threads(2)
    policy=MotorPolicy(checkpoint,root);core=policy.core
    print(json.dumps({'feedback_circuit_id':core.circuit.feedback_manifest['feedback_circuit_id'],'device':torch.cuda.get_device_name(0)}),flush=True)
    configuration=json.loads(sys.stdin.readline(1_000_001))
    if type(configuration['updates']) is not int or not 4<=configuration['updates']<=40:raise ValueError('Invalid integration updates')
    core.updates=configuration['updates']
    for key in ['retinal_mean','retinal_std','kc_mean','kc_std']:
        value=np.asarray(configuration[key],dtype=np.float32)
        if value.shape!=getattr(core,key).shape or not np.isfinite(value).all() or (key.endswith('_std') and (value<=0).any()):raise ValueError('Invalid sensory calibration')
        setattr(core,key,value)
    encoder=TorchFeedbackEncoder(core,'cuda')
    print(json.dumps({'calibration_identity':calibration_identity(core)}),flush=True)
    while True:
        line=sys.stdin.readline(1_000_001)
        if not line:break
        try:
            if len(line)>1_000_000:raise ValueError('Oversized sensory request')
            value=json.loads(line);retinal=np.asarray(value['retinal'],dtype=np.float32);body=np.asarray(value['body'],dtype=np.float32)
            if retinal.shape!=(len(core.circuit.retina),) or body.shape!=(len(core.circuit.body_indices),) or not np.isfinite(retinal).all() or not np.isfinite(body).all():raise ValueError('Invalid sensory request')
            features=core.encode_responses(encoder.responses(retinal[None],body[None],return_state=True))[0]
            state=encoder.last_state;state[core.circuit.kc]=features
            print(json.dumps({'id':value['id'],'features':features.tolist(),'display_activity':base64.b64encode(zlib.compress(np.rint(np.clip(np.abs(state),0,1)*255).astype(np.uint8).tobytes(),1)).decode()},allow_nan=False),flush=True)
        except Exception as error:print(json.dumps({'error':str(error)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--serve',action='store_true',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--root',required=True);a=p.parse_args();serve(a.checkpoint,a.root)
