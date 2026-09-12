import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_runtime_recovery import merge_episode_splits


def test_new_collection_preserves_every_source_episode_split():
    original=[{'episode':'old-val','split':'validation','cube_xy_mm':[345,8]},
              {'episode':'old-test','split':'test','cube_xy_mm':[355,4]}]
    cases=[{'id':'new','split':'train','task':{'cube_xy_mm':[352,-9]}}]
    result=merge_episode_splits(original,cases)
    assert result['episode_splits']=={'old-val':'validation','old-test':'test'}
    assert result['train']==[[352,-9]]
    assert result['validation']==[[345,8]]
    assert result['test']==[[355,4]]


def test_refuses_to_train_on_previously_held_out_configuration():
    original=[{'episode':'held-out','split':'test','cube_xy_mm':[355,4]}]
    with pytest.raises(ValueError,match='held-out placement'):
        merge_episode_splits(original,[{'id':'new','split':'train','task':{'cube_xy_mm':[355,4]}}])


def test_collection_head_may_differ_but_sensory_calibration_must_match():
    from types import SimpleNamespace
    import numpy as np
    import pytest
    from prepare_runtime_recovery import require_matching_sensory_contract
    def policy():return SimpleNamespace(metadata={'graph_id':'same'},core=SimpleNamespace(**{k:np.array([1.]) for k in ['retinal_mean','retinal_std','kc_mean','kc_std']}))
    source=policy();collector=policy()
    require_matching_sensory_contract(source,collector)
    collector.core.kc_std=np.array([2.])
    with pytest.raises(ValueError,match='calibration'):
        require_matching_sensory_contract(source,collector)
    collector=policy();collector.metadata['graph_id']='different'
    with pytest.raises(ValueError,match='contract'):
        require_matching_sensory_contract(source,collector)
