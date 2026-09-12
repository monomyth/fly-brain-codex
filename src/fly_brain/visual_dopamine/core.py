"""Retinal stimulation, anatomical motor routing and reward-gated eligibility learning.

This is an engineering rate-deviation model, not a molecular simulation of PAM neurons.
"""
import hashlib
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from .circuit import Circuit
from ..adapters import decode_image
from ..cameras import CURRENT_CAMERAS, CURRENT_RIG, LEGACY_CAMERAS, LEGACY_RIG


class RetinalEncoder:
    # Legacy default preserves offline reproducibility; current models store their registration.
    camera_names = LEGACY_CAMERAS
    camera_rig_revision = LEGACY_RIG
    def __init__(self, circuit):
        self.circuit = circuit
        self.profile = None

    def configure_cameras(self, names=LEGACY_CAMERAS, revision=LEGACY_RIG):
        names = tuple(names)
        if (names, revision) not in ((LEGACY_CAMERAS, LEGACY_RIG), (CURRENT_CAMERAS, CURRENT_RIG)):
            raise ValueError("Unsupported retinal camera registration")
        self.camera_names = names
        self.camera_rig_revision = revision

    def configure(self,profile):
        if profile is None:self.profile=None;return
        lower=np.asarray(profile.get("lower_m"),dtype=float);upper=np.asarray(profile.get("upper_m"),dtype=float)
        if profile.get("kind")!="fixed_workspace" or lower.shape!=(3,) or upper.shape!=(3,) or not np.isfinite(np.r_[lower,upper]).all() or np.any(upper<=lower):raise ValueError("Invalid retinal workspace profile")
        names=profile.get("cameras",list(self.camera_names))
        if not isinstance(names,list) or not names or any(x not in self.camera_names for x in names):raise ValueError("Invalid retinal workspace cameras")
        self.profile={"kind":"fixed_workspace","lower_m":lower.tolist(),"upper_m":upper.tolist(),"cameras":list(names)}

    def image_region(self,rgb,camera):
        if self.profile is None or camera.get("name") not in self.profile["cameras"]:return rgb
        from itertools import product
        lower=self.profile["lower_m"];upper=self.profile["upper_m"]
        points=np.array([list(p)+[1.] for p in product(*zip(lower,upper))])
        transform=np.asarray(camera["world_from_camera"],dtype=float);intrinsics=np.asarray(camera["intrinsics"],dtype=float)
        if transform.shape!=(4,4) or intrinsics.shape!=(3,3) or not np.isfinite(transform).all() or not np.isfinite(intrinsics).all() or camera.get("world_units")!="meters":raise ValueError("Workspace retina requires metric camera calibration")
        view=(np.linalg.inv(transform)@points.T)[:3];view[1:]*=-1
        if np.any(view[2]<=.001):raise ValueError("Retinal workspace is behind the camera")
        projected=intrinsics@view;uv=projected[:2]/projected[2]
        a=uv.min(axis=1);b=uv.max(axis=1);padding=.05*(b-a);a=np.floor(a-padding).astype(int);b=np.ceil(b+padding).astype(int)
        h,w=rgb.shape[:2];x0,y0=np.maximum(a,[0,0]);x1,y1=np.minimum(b,[w,h])
        if x1-x0<8 or y1-y0<8:raise ValueError("Retinal workspace is outside the camera image")
        return rgb[y0:y1,x0:x1]

    def sample(self, images, base_directory=None):
        cameras = {image["name"]: image for image in images}
        if not all(name in cameras for name in self.camera_names):
            raise ValueError("Retinal control needs recorded " + " and ".join(self.camera_names) + " camera images")
        if len(cameras) != len(images):
            raise ValueError("Duplicate camera images")
        if any(cameras[name].get("rig_revision", LEGACY_RIG) != self.camera_rig_revision for name in self.camera_names):
            raise ValueError("Camera viewpoint changed: collect new data and train a model for this camera rig")
        result = np.zeros(len(self.circuit.retina), dtype=np.float32)
        for side,name in enumerate(self.camera_names):
            rgb = decode_image(cameras[name], base_directory).astype(np.float32) / 255
            rgb = self.image_region(rgb,cameras[name])
            luminance = rgb @ np.asarray([.2126,.7152,.0722],dtype=np.float32)
            h,w = luminance.shape
            # Spatial receptive fields prevent a small cube falling between samples.
            filtered = gaussian_filter(luminance, sigma=(h/39*.55,w/36*.55), mode="nearest")
            selected = self.circuit.retina_side == side
            uv = self.circuit.retina_uv[selected]
            result[selected] = map_coordinates(filtered, [uv[:,1]*(h-1),uv[:,0]*(w-1)],order=1,mode="nearest")
        return result


class VisualCore:
    def __init__(self,circuit,updates=12):
        if updates < 4 or updates > 40: raise ValueError("Visual circuit requires 4–40 abstract integration steps")
        self.circuit = circuit
        self.encoder = RetinalEncoder(circuit)
        self.updates = updates
        self.retinal_mean = np.zeros(len(circuit.retina),dtype=np.float32)
        self.retinal_std = np.full(len(circuit.retina),.125,dtype=np.float32)
        self.kc_mean = np.zeros(len(circuit.kc),dtype=np.float32)
        self.kc_std = np.ones(len(circuit.kc),dtype=np.float32)
        self.state = np.zeros(len(circuit.ids),dtype=np.float32)

    def visual_response(self,retinal, enabled=True):
        # Each one-step curriculum trial is a new stimulus presentation.
        self.state.fill(0)
        drive = np.clip((np.asarray(retinal)-self.retinal_mean)/self.retinal_std,-1,1) if enabled else np.zeros_like(retinal)
        for _ in range(self.updates):
            message = self.circuit.sensory @ self.state if enabled else np.zeros_like(self.state)
            self.state = .35*self.state + .65*np.tanh(.95*message)
            self.state[self.circuit.retina] = drive
        return self.state[self.circuit.kc].copy()

    def fit_adaptation(self, retinal_samples, local_contrast=False):
        # Unsupervised tonic/gain calibration: no task labels, positions or rewards.
        self.retinal_mean = np.asarray(retinal_samples,dtype=np.float32).mean(axis=0)
        # Local contrast adaptation equalizes small cues and large moving surfaces.
        self.retinal_std = (np.maximum(np.asarray(retinal_samples,dtype=np.float32).std(axis=0),.01)
                            if local_contrast else np.full(len(self.circuit.retina),.125,dtype=np.float32))
        responses = np.asarray([self.visual_response(x) for x in retinal_samples])
        self.kc_mean = responses.mean(axis=0)
        std = responses.std(axis=0)
        self.kc_std = np.maximum(std, max(float(std.max())*.005,1e-12))
        return self.encode_responses(responses)

    def encode_responses(self,responses):
        return np.tanh((np.asarray(responses)-self.kc_mean)/self.kc_std).astype(np.float32)

    def encode(self,retinal,enabled=True):
        raw = self.visual_response(retinal,enabled)
        features = self.encode_responses(raw) if enabled else np.zeros(len(self.circuit.kc),dtype=np.float32)
        self.state[self.circuit.kc] = features
        return features


class DopamineLearner:
    """Only real KC→MBON edges learn. Fixed motor decoding reads real leg MN states.

    Eligibility is presynaptic activity times exploratory postsynaptic perturbation.
    A PAM01 rate deviation gates the update through its recorded MBON innervation.
    """
    def __init__(self,circuit,seed=0,learning_rate=.002,noise=.35,eligibility_tau=6.):
        self.circuit=circuit
        self.rng=np.random.default_rng(seed)
        self.learning_rate=float(learning_rate); self.noise=float(noise)
        self.eligibility_tau=float(eligibility_tau)
        if not np.isfinite([self.learning_rate,self.noise,self.eligibility_tau]).all() or min(self.learning_rate,self.noise,self.eligibility_tau)<=0: raise ValueError("Positive learner parameters required")
        self.mask=circuit.plastic_base>0
        self.base=circuit.plastic_base.astype(np.float32)
        self.weight_multiple=4.0
        self.weights=self.base.copy()
        self.eligibility=np.zeros_like(self.weights)
        self.baseline=.5
        self.dopamine=np.zeros(len(circuit.dan),dtype=np.float32)
        self.last_reward=0.;self.last_update_l1=0.
        self._initialize_output()

    def _initialize_output(self):
        circuit=self.circuit
        types=[x["type"] or "" for x in circuit.neurons["motor"]]
        positive=np.array([x=="Ti extensor MN" for x in types])
        negative=np.array([x in ["Ti flexor MN","Acc. ti flexor MN"] for x in types])
        if not positive.any() or not negative.any(): raise ValueError("Both annotated tibial motor pools must reach the decoder")
        self.motor_readout=positive.astype(np.float32)/positive.sum()-negative.astype(np.float32)/negative.sum()
        # Fixed body-interface gain, calibrated on neural perturbations only.
        transfer=circuit.motor_weights@circuit.dn_weights
        sensitivity=self.motor_readout@transfer
        self.motor_gain=1/max(float(np.linalg.norm(sensitivity)),1e-6)
        self.mbon=np.zeros(len(circuit.mbon),dtype=np.float32)
        self.descending=np.zeros(len(circuit.dn),dtype=np.float32)
        self.motor=np.zeros(len(circuit.motor),dtype=np.float32)
        self.last_action=0

    def choose(self,features,explore=False):
        features=np.asarray(features,dtype=np.float32)
        perturbation=self.rng.normal(0,self.noise,len(self.circuit.mbon)).astype(np.float32) if explore else np.zeros(len(self.circuit.mbon),dtype=np.float32)
        self.mbon=np.tanh(self.weights@features+perturbation)
        self.descending=np.tanh(self.circuit.dn_weights@self.mbon)
        self.motor=np.tanh(self.circuit.motor_weights@self.descending)
        score=float(self.motor_gain*(self.motor_readout@self.motor))
        self.last_action=int(score>=0)  # -1 degree or +1 degree, an explicit cross-body adapter.
        self.eligibility[:]=(perturbation[:,None]/(self.noise*self.noise))*features[None,:]*self.mask if explore else 0
        return self.last_action,score

    def reinforce(self,reward,delay_s=.3,dopamine_enabled=True,plasticity_enabled=True):
        if not np.isfinite(reward) or not 0<=reward<=1: raise ValueError("Reward must be in [0,1]")
        if not np.isfinite(delay_s) or delay_s<0: raise ValueError("Nonnegative finite delay required")
        self.eligibility*=np.exp(-delay_s/self.eligibility_tau)
        self.last_reward=float(reward)
        # Signed deviations represent increases/dips about tonic dopamine activity.
        self.dopamine[:]=np.tanh(reward-self.baseline) if dopamine_enabled else 0
        modulation=self.circuit.dopamine_weights@self.dopamine
        before=self.weights.copy()
        if plasticity_enabled:
            self.weights+=self.learning_rate*modulation[:,None]*self.eligibility
            np.clip(self.weights,0,self.weight_multiple*self.base,out=self.weights)
            self.weights*=self.mask
        self.last_update_l1=float(abs(self.weights-before).sum())
        self.baseline=.98*self.baseline+.02*reward
        self.eligibility.fill(0)
        return {"reward":self.last_reward,"dopamine_mean":float(self.dopamine.mean()),"weight_change_l1":self.last_update_l1}

    def state_into(self,state):
        state[self.circuit.mbon]=self.mbon
        state[self.circuit.dn]=self.descending
        state[self.circuit.motor]=self.motor
        state[self.circuit.dan]=self.dopamine
        return state

    def weight_digest(self):
        return hashlib.sha256(self.weights.tobytes()).hexdigest()
