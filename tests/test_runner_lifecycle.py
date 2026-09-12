from types import SimpleNamespace
import json
import pytest
from fly_brain import evaluate as module
from fly_brain.simulator import current_episode_task, release_for_cleanup, Simulator


class CurrentClient:
    def __init__(self, observation):
        self.observation = observation
        self.calls = []
    def call(self, tool, **args):
        self.calls.append(tool)
        return {'task': {'input_mode':'state','cube_xy_mm':[340,-20],'cube_size_mm':40,'timeout_seconds':90},
                'state': {'phase':'ready','episode_id':self.observation['episode_id']}}
    def observe(self, images=False): return self.observation
    def connect(self, **kw): pass
    def release(self): raise ConnectionRefusedError('simulator was closed')


def test_current_episode_is_read_only_and_preserves_placement(observation):
    client = CurrentClient(observation)
    task = current_episode_task(client, 'state', observation['episode_id'])
    assert task['cube_xy_mm'] == [340,-20] and task['cube_size_mm'] == 40
    assert client.calls == ['rebot_get_task']
    with pytest.raises(ValueError, match='changed'):
        current_episode_task(client, 'state', 'other-episode')
    with pytest.raises(ValueError, match='needs vision'):
        current_episode_task(client, 'vision', observation['episode_id'])


def test_cleanup_keeps_the_primary_error_and_finishes_report(tmp_path, observation):
    client = CurrentClient(observation)
    policy = SimpleNamespace(mode='state', hz=5, ablation='none',
                             metadata={'name':'test','architecture':{'kind':'malecns'}}, reset=lambda obs:None)
    def fail(obs): raise RuntimeError('original inference failure')
    policy.act = fail
    report = module.evaluate(client, policy, [{}], tmp_path, current_episode=True)
    episode = report['episodes'][0]
    assert episode['error'] == 'original inference failure'
    assert episode['cleanup_error']['type'] == 'ConnectionRefusedError'
    assert (tmp_path/'evaluation.json').exists()


def test_closed_app_after_recording_does_not_leave_partial_episode(tmp_path, observation):
    client = CurrentClient(observation)
    count = 0
    def observe(images=False):
        nonlocal count
        count += 1
        if count > 1: raise ConnectionRefusedError('original disconnect')
        return observation
    client.observe = observe
    policy = SimpleNamespace(mode='state', hz=5, ablation='none',
        metadata={'name':'test','architecture':{'kind':'malecns'}}, reset=lambda obs:None, act=lambda obs:{})
    result = module.evaluate(client, policy, [{}], tmp_path, current_episode=True)
    assert result['episodes'][0]['error'] == 'original disconnect'
    manifest = next(tmp_path.glob('episode-*/manifest.json'))
    assert json.loads(manifest.read_text())['complete']
    assert not list(tmp_path.glob('episode-*/steps.jsonl.partial'))


def test_keyboard_interrupt_is_preserved_and_saved(tmp_path, observation):
    client = CurrentClient(observation)
    policy = SimpleNamespace(mode='state', hz=5, ablation='none',
        metadata={'name':'test','architecture':{'kind':'malecns'}}, reset=lambda obs:None)
    def interrupt(obs): raise KeyboardInterrupt()
    policy.act=interrupt
    result=module.evaluate(client,policy,[{},{}],tmp_path,current_episode=True)
    assert result['attempts'] == 1
    assert result['episodes'][0]['interrupted']
    assert result['episodes'][0]['error_type'] == 'KeyboardInterrupt'


def test_context_cleanup_does_not_mask_original_exception():
    sim=Simulator()
    sim.client=SimpleNamespace(release=lambda:(_ for _ in ()).throw(ConnectionRefusedError('closed')))
    primary=RuntimeError('primary')
    sim.__exit__(type(primary),primary,None)
    assert sim.cleanup_error['type']=='ConnectionRefusedError'


def test_explicit_release_remains_strict(observation):
    client=CurrentClient(observation)
    with pytest.raises(ConnectionRefusedError): client.release()
    assert release_for_cleanup(client)['type']=='ConnectionRefusedError'


def test_interrupt_during_model_loading_is_clean(monkeypatch, capsys):
    from fly_brain import cli
    def interrupted(): raise KeyboardInterrupt()
    monkeypatch.setattr(cli, '_main', interrupted)
    with pytest.raises(SystemExit) as stopped:
        cli.main()
    assert stopped.value.code == 130
    assert capsys.readouterr().err == "Interrupted.\n"
