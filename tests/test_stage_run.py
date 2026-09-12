import pytest
from fly_brain.visual_dopamine.stage_run import actor_view,unsupported_grasp,validate_handoff


def test_stage_actor_excludes_setup_targets_and_cube_geometry():
    observation={k:None for k in ('images','joints_deg','gripper_mm','finger_contacts','simulation_time','frame_id','episode_id')}
    observation.update(cube_pose={'position_mm':[350,0,9]},expert_action=[99]*7,stage='hold',reward=1)
    result=actor_view(observation)
    assert result['input_mode']=='vision'
    assert not {'cube_pose','expert_action','stage','reward'} & result.keys()


def test_stage_grasp_requires_bilateral_unsupported_lift():
    e={'held':True,'clearance_mm':20,'contacts':['finger_left_link','finger_right_link'],'dropped':False}
    assert unsupported_grasp(e)
    assert not unsupported_grasp({**e,'contacts':e['contacts']+['floor']})
    assert not unsupported_grasp({**e,'clearance_mm':0})
    assert not unsupported_grasp({**e,'dropped':True})
    validate_handoff('lift_hold',e)
    with pytest.raises(ValueError,match='hold credit'):validate_handoff('lift_hold',{**e,'success':True})
    with pytest.raises(ValueError,match='hold credit'):validate_handoff('lift_hold',{**e,'hold_seconds':.2})
    with pytest.raises(ValueError,match='verified unsupported'):validate_handoff('lift_hold',{**e,'held':False})


def test_temporal_fixture_warmup_preserves_order_and_avoids_duplicate_frames():
    from types import SimpleNamespace
    from fly_brain.visual_dopamine.stage_run import warmup_history
    p=SimpleNamespace(core=SimpleNamespace(temporal_config={'enabled':True}),hz=2.)
    history=[({'simulation_time':t,'frame_id':f},None) for t,f in [(0,0),(.2,1),(.5,2),(.5,2),(.7,3),(.9,4)]]
    selected=warmup_history(p,history)
    assert [x[0]['frame_id'] for x in selected]==[0,2,4]
    with pytest.raises(ValueError,match='chronological'):warmup_history(p,list(reversed(history)))


def test_deployment_wait_is_used_unless_explicitly_overridden():
    from fly_brain.visual_dopamine.motor_run import resolve_settle_seconds
    assert resolve_settle_seconds({},None) is None
    assert resolve_settle_seconds({'deployment_settle_seconds':.1},None)==.1
    assert resolve_settle_seconds({'deployment_settle_seconds':.1},.2)==.2
    with pytest.raises(ValueError,match='Settle override'):resolve_settle_seconds({'deployment_settle_seconds':2},None)


def test_failed_qualification_stops_without_inflating_attempts(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import fly_brain.visual_dopamine.motor_run as runner
    class Client:
        started=False
        def call(self,*args,**kwargs):return {}
        def observe(self,**kwargs):return {}
        def connect(self,**kwargs):self.started=True
        def release(self):self.started=False
    client=Client()
    def configure(client,requested):client.started=False;return requested
    def act(observation,**kwargs):
        if client.started:raise RuntimeError('Failure during an actual trial')
        return {}
    monkeypatch.setattr(runner,'configure_episode',configure)
    monkeypatch.setattr(runner,'reward_metrics',lambda *args:{'pose_cost':0,'grip_error_mm':0,'transport':False})
    monkeypatch.setattr(runner.ActivityPublisher,'for_policy',staticmethod(lambda *args:None))
    policy=SimpleNamespace(metadata={'name':'test'},checkpoint=tmp_path,reset=lambda obs:None,act=act)
    report=runner.run(client,policy,[{}]*5,tmp_path/'limited',stop_after_failures=2)
    assert report['attempts']==2 and report['successes']==0 and report['startup_failures']==0
    assert report['planned_attempts']==5 and report['stopped_early_after_failures']==2
    default=runner.run(client,policy,[{}]*5,tmp_path/'default')
    assert default['attempts']==5 and 'stopped_early_after_failures' not in default


def test_locked_screen_does_not_count_as_a_pickup_failure(tmp_path):
    from types import SimpleNamespace
    from fly_brain.visual_dopamine.motor_run import run
    class LockedClient:
        def call(self,*args,**kwargs):raise RuntimeError('Unlock the Mac before starting live simulation; RealityKit cannot advance reliably while the screen is locked')
        def release(self):pass
    result=run(LockedClient(),SimpleNamespace(metadata={},checkpoint=tmp_path),[{}]*20,tmp_path/'blocked',stop_after_failures=3)
    assert result['attempts']==0 and result['startup_failures']==1 and result['records']==1
    assert result['execution_blocked']=='screen_locked'
    assert 'stopped_early_after_failures' not in result
