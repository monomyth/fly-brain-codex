"""Record encoder outputs before motor activity updates the display state."""
import numpy as np


class SensoryFeatureTap:
    def __init__(self, base):
        self.base = base
        self.last_features = None
        self.last_frame = None

    def __getattr__(self, name):
        return getattr(self.base, name)

    def reset(self):
        self.base.reset()
        self.last_features = None
        self.last_frame = None

    def encode_observation(self, observation, *args, **kwargs):
        value = self.base.encode_observation(observation, *args, **kwargs)
        self.last_features = np.asarray(value, dtype=np.float32).copy()
        self.last_frame = (observation['episode_id'], observation['frame_id'])
        return value

    def features_for(self, observation):
        if self.last_frame != (observation['episode_id'], observation['frame_id']):
            raise ValueError('Encoded features do not match the recorded observation')
        if self.last_features is None or not np.isfinite(self.last_features).all():
            raise ValueError('Missing or invalid encoded features')
        return self.last_features.copy()
