import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from sensory_feature_tap import SensoryFeatureTap


class Core:
    def __init__(self): self.state = np.array([1., 2.], dtype=np.float32)
    def reset(self): self.state.fill(0)
    def encode_observation(self, observation): return self.state


def test_capture_survives_motor_display_state_mutation():
    core = Core(); tap = SensoryFeatureTap(core)
    observation = {'episode_id':'episode-a', 'frame_id':4}
    output = tap.encode_observation(observation)
    assert output is core.state
    core.state *= .1
    assert np.array_equal(tap.features_for(observation), [1.,2.])
    assert np.array_equal(tap.state, core.state)


def test_capture_requires_exact_frame_and_clears_at_reset():
    tap=SensoryFeatureTap(Core()); observation={'episode_id':'a','frame_id':1}
    tap.encode_observation(observation)
    with pytest.raises(ValueError):tap.features_for({**observation,'frame_id':2})
    with pytest.raises(ValueError):tap.features_for({**observation,'episode_id':'b'})
    tap.reset()
    with pytest.raises(ValueError):tap.features_for(observation)
