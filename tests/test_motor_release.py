from copy import deepcopy
import pytest
from fly_brain.visual_dopamine import motor_release as release


class Client:
    def __init__(self):
        self.tick = 0
        self.commands = []
    def observe(self, images=False):
        return {'episode_id': 'one', 'joints_deg': [1,-100,-90,70,2,3],
                'gripper_mm': min(90,20+self.tick*5), 'simulation_time': self.tick*.2,
                'finger_contacts': {'left': self.tick<4, 'right': self.tick<4}}
    def action(self, observation, action):
        self.commands.append(deepcopy(action)); self.tick += 1
    def call(self, name):
        return {'episode_id': 'one', 'success': True, 'held': self.tick<4,
                'contacts': ['floor'] if self.tick>=5 else [],
                'clearance_mm': 0 if self.tick>=5 else 100,
                'linear_speed_mm_s': 0, 'angular_speed_deg_s': 0}


def patch_clock(monkeypatch):
    clock=[0.]
    monkeypatch.setattr(release.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(release.time,'sleep',lambda duration:clock.__setitem__(0,clock[0]+duration))


def test_opening_waits_for_fingers_and_settled_landing(monkeypatch,tmp_path):
    patch_clock(monkeypatch);client=Client()
    result=release.open_and_drop(client,{'success':True,'held':True,'episode_id':'one'},tmp_path)
    assert result['success']
    assert len(client.commands)>=16
    assert all(a=={'joints_deg':[1,-100,-90,70,2,3],'gripper_mm':90.} for a in client.commands)
    assert (tmp_path/'release.json').exists()


def test_does_not_open_after_unsuccessful_hold(tmp_path):
    client=Client()
    with pytest.raises(ValueError,match='verified'):
        release.open_and_drop(client,{'success':False,'held':True},tmp_path)
    assert client.commands==[]


def test_episode_change_prevents_command(monkeypatch,tmp_path):
    patch_clock(monkeypatch);client=Client()
    with pytest.raises(RuntimeError,match='Episode changed'):
        release.open_and_drop(client,{'success':True,'held':True,'episode_id':'old'},tmp_path)
    assert client.commands==[]


def test_falling_without_opening_is_not_release_success(monkeypatch,tmp_path):
    patch_clock(monkeypatch);client=Client();original=client.observe
    client.observe=lambda **kw:{**original(**kw),'gripper_mm':20.}
    result=release.open_and_drop(client,{'success':True,'held':True,'episode_id':'one'},tmp_path)
    assert not result['success'] and 'error' in result


def test_disconnect_is_preserved(monkeypatch,tmp_path):
    patch_clock(monkeypatch);client=Client()
    def disconnected(*args):raise ConnectionRefusedError('closed')
    client.action=disconnected
    with pytest.raises(ConnectionRefusedError):
        release.open_and_drop(client,{'success':True,'held':True,'episode_id':'one'},tmp_path)


@pytest.mark.parametrize('release_success',[True,False])
def test_runner_reports_whole_task_and_releases_ownership_last(monkeypatch,tmp_path,release_success):
    import numpy as np
    from types import SimpleNamespace
    from fly_brain.visual_dopamine import motor_run as runner
    events=[]
    observation={'joints_deg':[0]*6,'gripper_mm':20,'simulation_time':1.,'episode_id':'one'}
    evaluation={'success':True,'held':True,'episode_id':'one','hold_seconds':5.}
    class LiveClient:
        def observe(self,**kwargs):return observation
        def call(self,*args,**kwargs):return evaluation
        def connect(self,**kwargs):pass
        def action(self,*args):events.append('neural_action')
        def release(self):events.append('ownership_released')
    policy=SimpleNamespace(metadata={'name':'fake'},checkpoint=tmp_path,hz=2,
        reset=lambda *a:None,act=lambda *a,**kw:{'joints_deg':[0]*6,'gripper_mm':0},
        feedback=lambda *a:None,last_scores=np.zeros(7),last_choices=np.zeros(7))
    monkeypatch.setattr(runner,'configure_episode',lambda *a:{'hold_seconds':5})
    monkeypatch.setattr(runner.ActivityPublisher,'for_policy',lambda *a:None)
    monkeypatch.setattr(runner,'save_observation',lambda *a:None)
    monkeypatch.setattr(runner,'reward_metrics',lambda *a:{'pose_cost':0,'grip_error_mm':0,'transport':True})
    monkeypatch.setattr(runner.time,'sleep',lambda *a:None)
    def drop(*args):
        events.append('open_and_drop')
        return {'success':release_success,'error':'Drop failed' if not release_success else None}
    monkeypatch.setattr(runner,'open_and_drop',drop)
    report=runner.run(LiveClient(),policy,[{}],tmp_path/'run')
    assert report['hold_successes']==1 and report['successes']==int(release_success)
    assert report['release_successes']==int(release_success) and report['release_required']
    assert events==['neural_action','open_and_drop','ownership_released']
