"""Portable retinal policy. Cube coordinates are never an input to this class."""
import json
import os
import tempfile
import shutil
from pathlib import Path
import numpy as np
from ..assets import home, local_artifact, checked_files, artifact_manifest, canonical_hash, sha256_file
from .circuit import Circuit
from .core import VisualCore, DopamineLearner
from .experiment import response_action


class VisualDopaminePolicy:
    def __init__(self,checkpoint,root=None,visual_enabled=True):
        self.root=home(root);checkpoint=local_artifact(checkpoint);self.checkpoint=checkpoint.resolve()
        self.metadata=json.loads((checkpoint/"manifest.json").read_text())
        if self.metadata.get("schema")!="visual-dopamine-checkpoint-v1":raise ValueError("Not a visual dopamine checkpoint")
        checked_files(checkpoint,self.metadata)
        identity=self.metadata["circuit_id"]
        if len(identity)!=64 or any(c not in "0123456789abcdef" for c in identity):raise ValueError("Invalid circuit identity")
        self.circuit=Circuit(self.root/"circuits"/identity)
        if self.circuit.manifest["graph_id"]!=self.metadata["graph_id"]:raise ValueError("Checkpoint graph mismatch")
        self.core=VisualCore(self.circuit,self.metadata["architecture"]["updates"])
        rule=self.metadata.get("learning_rule",{})
        self.learner=DopamineLearner(self.circuit,seed=self.metadata["seed"],
            learning_rate=rule.get("learning_rate",.002),noise=rule.get("exploration_sigma",.35),
            eligibility_tau=rule.get("eligibility_tau_seconds",6.))
        with np.load(checkpoint/"model.npz",allow_pickle=False) as data:
            for name in ["retinal_mean","kc_mean","kc_std"]:
                value=data[name]
                if value.shape!=getattr(self.core,name).shape or not np.isfinite(value).all():raise ValueError("Invalid visual calibration")
                setattr(self.core,name,value.copy())
            if "retinal_std" in data:
                value=data["retinal_std"]
                if value.shape!=self.core.retinal_std.shape or not np.isfinite(value).all() or (value<=0).any():raise ValueError("Invalid retinal gain calibration")
                self.core.retinal_std=value.copy()
            if (self.core.kc_std<=0).any():raise ValueError("Invalid neural gain calibration")
            weights=data["weights"]
            if weights.shape!=self.learner.weights.shape or not np.isfinite(weights).all() or (weights<0).any() or (weights>4*self.learner.base+1e-6).any():raise ValueError("Invalid plastic synapse weights")
            if (weights[~self.learner.mask]!=0).any():raise ValueError("Checkpoint introduced non-anatomical synapses")
            if not np.array_equal(data["motor_readout"],self.learner.motor_readout):raise ValueError("Motor bridge does not match anatomical motor pools")
            self.learner.weights[:]=weights;self.learner.baseline=float(data["baseline"])
            if not np.isfinite(self.learner.baseline) or not 0 <= self.learner.baseline <= 1: raise ValueError("Invalid reward expectation")
        if "rng_state" in self.metadata:
            self.learner.rng.bit_generator.state=self.metadata["rng_state"]
        self.visual_enabled=visual_enabled;self.mode="vision";self.hz=1.;self.ablation="none" if visual_enabled else "block-retina"
        self.state=np.zeros((1,len(self.circuit.ids),1),dtype=np.float32)
        self.episode=None;self.last_frame=None;self.last_choice=None;self.last_score=0.

    def reset(self,observation):
        self.episode=observation["episode_id"];self.last_frame=None
        self.core.state.fill(0);self.learner.eligibility.fill(0);self.learner.dopamine.fill(0)
        self.learner.last_reward=0.;self.learner.last_update_l1=0.
        self.state.fill(0)

    def act(self,observation,explore=False):
        if self.episode!=observation["episode_id"]:raise ValueError("Episode changed without a policy reset")
        if self.last_frame is not None and observation["frame_id"]<=self.last_frame:raise ValueError("Camera frame did not advance")
        self.last_frame=observation["frame_id"]
        retinal=self.core.encoder.sample(observation["images"])
        features=self.core.encode(retinal,enabled=self.visual_enabled)
        self.last_choice,self.last_score=self.learner.choose(features,explore=explore)
        self.learner.state_into(self.core.state);self.state[0,:,0]=self.core.state
        return response_action(observation,self.last_choice)

    def feedback(self,reward,delay_s=.3,learn=False):
        result=self.learner.reinforce(reward,delay_s,dopamine_enabled=self.metadata.get("dopamine_enabled",True),plasticity_enabled=learn)
        self.learner.state_into(self.core.state);self.state[0,:,0]=self.core.state
        return result

    def activity_metadata(self):
        return {"dopamine":{"signal":float(self.learner.dopamine.mean()),"reward":self.learner.last_reward,
                            "cells":len(self.circuit.dan),"weight_change_l1":self.learner.last_update_l1}}

    def save(self,destination,training_report):
        destination=Path(destination)
        if destination.exists():raise FileExistsError("Online learning creates a new checkpoint")
        destination.parent.mkdir(parents=True,exist_ok=True)
        temp=Path(tempfile.mkdtemp(prefix=".retinal-checkpoint-",dir=destination.parent))
        try:
            np.savez(temp/"model.npz",weights=self.learner.weights,retinal_mean=self.core.retinal_mean,
                     retinal_std=self.core.retinal_std,kc_mean=self.core.kc_mean,kc_std=self.core.kc_std,
                     baseline=np.array(self.learner.baseline),motor_readout=self.learner.motor_readout)
            metadata={k:v for k,v in self.metadata.items() if k not in ["files","test_result"]}
            metadata.update(name="retinal-online-"+destination.parent.name[:40],parent_checkpoint=str(self.checkpoint),
                            training_method="online PAM01-gated node perturbation",online_training_report=training_report,
                            closed_loop_validated=False,rng_state=self.learner.rng.bit_generator.state,
                            package_source_hash=canonical_hash({str(p.relative_to(Path(__file__).parents[1])):sha256_file(p) for p in Path(__file__).parents[1].rglob("*.py")}))
            artifact_manifest(temp,metadata);os.rename(temp,destination)
        finally:
            if temp.exists():shutil.rmtree(temp)
        return str(destination.resolve())
