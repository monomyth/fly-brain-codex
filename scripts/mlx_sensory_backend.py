"""Experimental Metal acceleration of the unchanged fixed sensory calculation.

Not enabled by the application. Validate against the CPU implementation before
using this wrapper in a separately recorded runtime comparison.
"""
import numpy as np

KERNEL_SOURCE=r'''
    #pragma clang fp contract(off)
    uint row = thread_position_in_grid.x;
    if (row >= uint(state_shape[0])) return;
    float visual = 0.0f;
    float body = 0.0f;
    for (int edge = row_ptr[row]; edge < row_ptr[row + 1]; ++edge) {
        int index = columns[edge] * 2;
        float weight = weights[edge];
        visual += weight * state[index];
        body += weight * state[index + 1];
    }
    int vi = retinal_lookup[row];
    int bi = body_lookup[row];
    float next_visual = mixing[0] * state[row * 2] + mixing[1] * metal::precise::tanh(0.95f * visual);
    float next_body = mixing[0] * state[row * 2 + 1] + mixing[1] * metal::precise::tanh(0.95f * body);
    out[row * 2] = enabled[0] == 0 ? 0.0f : (vi >= 0 ? retinal[vi] : next_visual);
    out[row * 2 + 1] = bi >= 0 ? afferents[bi] : next_body;
'''


class MLXFeedbackCore:
    def __init__(self,base):
        import mlx.core as mx
        if not mx.metal.is_available():raise RuntimeError('Apple Metal GPU is unavailable')
        if base.circuit.feedback_manifest['recipe']['version']!=2:raise ValueError('MLX wrapper requires separated sensory modalities')
        self.base=base;self.mx=mx;self.n=len(base.circuit.ids)
        matrix=base.circuit.sensory.tocsr()
        if matrix.dtype!=np.float32 or matrix.shape!=(self.n,self.n) or matrix.nnz>=2**31:raise ValueError('Unsupported sensory matrix for Metal')
        self.row_ptr=mx.array(matrix.indptr.astype(np.int32));self.columns=mx.array(matrix.indices.astype(np.int32));self.weights=mx.array(matrix.data)
        r=np.full(self.n,-1,dtype=np.int32);b=r.copy()
        r[base.circuit.retina]=np.arange(len(base.circuit.retina),dtype=np.int32)
        b[base.circuit.body_indices]=np.arange(len(base.circuit.body_indices),dtype=np.int32)
        self.retinal_lookup=mx.array(r);self.body_lookup=mx.array(b)
        self.kernel=mx.fast.metal_kernel(name='malecns_sensory_csr_v1',input_names=['row_ptr','columns','weights','state','retinal_lookup','body_lookup','retinal','afferents','mixing','enabled'],output_names=['out'],source=KERNEL_SOURCE,compile_options={'math_mode':'safe'})
        mx.eval(self.row_ptr,self.columns,self.weights,self.retinal_lookup,self.body_lookup)
        self.reset()

    def __getattr__(self,name):return getattr(self.base,name)

    @property
    def state(self):return self.base.state

    def reset(self):
        self.gpu_state=self.mx.zeros((self.n,2),dtype=self.mx.float32);self.last_time=None
        if hasattr(self.base,'reset'):self.base.reset()
        else:self.base.state.fill(0)

    def response(self,retinal,body,visual_enabled=True,simulation_time=None):
        mx=self.mx
        config=getattr(self.base,'temporal_config',None)
        if config:
            from fly_brain.visual_dopamine.temporal import integration_alpha
            if simulation_time is None or not np.isfinite(simulation_time):raise ValueError('Temporal sensing requires simulation time')
            dt=config['nominal_interval_seconds'] if self.last_time is None else simulation_time-self.last_time
            alpha=integration_alpha([dt],config)[0]
            mixing=np.array([np.float32(1)-alpha,alpha],dtype=np.float32)
        else:
            self.gpu_state=mx.zeros((self.n,2),dtype=mx.float32)
            mixing=np.array([.35,.65],dtype=np.float32)
        drive=np.clip((np.asarray(retinal,dtype=np.float32)-self.base.retinal_mean)/self.base.retinal_std,-1,1)
        body=np.asarray(body,dtype=np.float32)
        if drive.shape!=(len(self.circuit.retina),) or body.shape!=(len(self.circuit.body_indices),) or not np.isfinite(drive).all() or not np.isfinite(body).all():raise ValueError('Invalid Metal sensory input')
        inputs=[mx.array(drive),mx.array(body),mx.array(mixing),mx.array([int(visual_enabled)],dtype=mx.int32)]
        state=self.gpu_state
        for _ in range(self.base.updates):
            state=self.kernel(inputs=[self.row_ptr,self.columns,self.weights,state,self.retinal_lookup,self.body_lookup,*inputs],grid=(self.n,1,1),threadgroup=(min(self.n,128),1,1),output_shapes=[(self.n,2)],output_dtypes=[mx.float32],stream=mx.gpu)[0]
        mx.eval(state);self.gpu_state=state;values=np.array(state)
        if not np.isfinite(values).all():raise RuntimeError('Nonfinite Metal sensory state')
        if config:self.last_time=float(simulation_time)
        self.base.state=values.sum(axis=1)
        self.base.state[self.circuit.visual_kc]=values[self.circuit.visual_kc,0]
        self.base.state[self.circuit.ascending]=values[self.circuit.ascending,1]
        return self.base.state[self.circuit.kc].copy()

    def encode_observation(self,observation,base_directory=None,visual_enabled=True):
        retinal=self.encoder.sample(observation['images'],base_directory);body=self.body_encoder.sample(observation)
        raw=self.response(retinal,body,visual_enabled,observation.get('simulation_time'))
        features=self.base.encode_responses(raw);self.base.state[self.circuit.kc]=features
        return features
