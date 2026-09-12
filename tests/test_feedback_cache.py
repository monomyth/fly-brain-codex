import numpy as np
import pytest
from fly_brain.visual_dopamine.feedback_cache import cached_responses


def test_exact_inputs_reuse_responses_in_any_order_and_encode_only_new_rows(tmp_path):
    r=np.array([[1,2],[3,4]],dtype=np.float32);b=np.array([[5],[6]],dtype=np.float32);calls=[]
    def encode(r,b):
        calls.append(len(r));return np.c_[r.sum(1),b[:,0]]
    path=tmp_path/'calibration-a.npz'
    first,_=cached_responses(path,r,b,2,encode)
    second,stats=cached_responses(path,r[::-1],b[::-1],2,encode)
    assert np.array_equal(second,first[::-1]) and calls==[2] and stats['encoded_unique_rows']==0
    expanded=np.vstack([r,r[0]]);body=np.vstack([b,[[7]]])
    actual,stats=cached_responses(path,expanded,body,2,encode)
    assert calls==[2,1] and stats['encoded_unique_rows']==1
    assert np.array_equal(actual,np.array([[3,5],[7,6],[3,7]]))
    cached_responses(tmp_path/'calibration-b.npz',r,b,2,encode)
    assert calls==[2,1,2]


def test_legacy_seed_must_be_consistent_for_identical_inputs(tmp_path):
    r=np.ones((2,2));b=np.zeros((2,1))
    def unexpected(*args):raise AssertionError('Seed should avoid encoding')
    result,_=cached_responses(tmp_path/'a.npz',r,b,2,unexpected,precomputed=np.ones((2,2)))
    assert np.array_equal(result,np.ones((2,2)))
    with pytest.raises(ValueError,match='Conflicting'):
        cached_responses(tmp_path/'a.npz',r,b,2,unexpected,precomputed=np.array([[1,1],[2,2]]))
    with pytest.raises(ValueError,match='Invalid sensory response'):
        cached_responses(tmp_path/'b.npz',r,b,2,unexpected,precomputed=np.ones((1,2)))
