"""Factory compatible with the simulator's reset(observation)/act(observation) API."""
import os
import numpy as np
import torch
from .adapters import ObservationAdapter, ActionAdapter, feature_schema
from .checkpoint import load_checkpoint
from .assets import home


class Policy:
    def __init__(self, checkpoint, root=None, ablation="none"):
        self.root = home(root)
        torch.set_num_threads(4)
        self.model, self.metadata = load_checkpoint(checkpoint, root)
        self.model.eval()
        self.mode = self.metadata["observation_schema"]["mode"]
        if self.metadata["observation_schema"] != feature_schema(self.mode):
            raise ValueError("Observation schema version does not match checkpoint")
        self.hz = self.metadata["hz"]
        self.encoder = ObservationAdapter(self.mode, self.hz)
        self.decoder = ActionAdapter(self.hz)
        self.mean = np.array(self.metadata["normalization"]["mean"], dtype=np.float32)
        self.std = np.array(self.metadata["normalization"]["std"], dtype=np.float32)
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or np.any(self.std <= 0):
            raise ValueError("Invalid checkpoint normalization")
        self.state = None
        self.ablation = ablation
        if ablation != "none":
            if self.metadata["architecture"]["kind"] != "malecns":
                raise ValueError("Graph ablations require a MaleCNS model")
            if ablation == "disconnect-inputs":
                self.model.inputs_enabled = False
            elif ablation == "disable-graph":
                self.model.graph_enabled = False
            else:
                raise ValueError("Unknown graph ablation")

    def reset(self, observation):
        self.encoder.reset(observation)
        self.state = self.model.initial_state()

    def act(self, observation):
        if self.state is None:
            raise ValueError("Reset policy before the episode")
        features = self.encoder.encode(observation)
        inputs = torch.from_numpy((features - self.mean) / self.std).unsqueeze(0)
        with torch.inference_mode():
            output, self.state = self.model.step(inputs, self.state)
        delta = output[0, :6].tanh().cpu().numpy()
        opened = output[0, 6].sigmoid().item()
        return self.decoder.decode(observation, delta, opened)


def create_policy():
    checkpoint = os.environ.get("FLY_BRAIN_CHECKPOINT")
    if not checkpoint:
        raise ValueError("Set FLY_BRAIN_CHECKPOINT to a trained model directory")
    return Policy(checkpoint)
