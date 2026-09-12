"""Explicit task-ending release, separate from the learned pickup controller."""
import time
from ..assets import write_json


def open_and_drop(client, hold, directory, timeout_seconds=10.):
    """Keep the reached joint pose, open fully, and verify a settled floor landing.

    Native pickup/hold success must already be recorded. This deterministic task
    completion command is never described as a learned neural motor response.
    The controller lease remains active until opening and landing are observed.
    """
    if not hold.get('success') or not hold.get('held'):
        raise ValueError('Opening requires a verified held-cube success')
    start = time.monotonic()
    observation = client.observe()
    episode = hold['episode_id']
    joints = list(observation['joints_deg'])
    result = {'success': False, 'command_source': 'task_completion_open_gripper',
              'hold_evaluation': hold, 'steps': []}
    settled_since = None
    write_json(directory/'release.json', result)
    while time.monotonic()-start < timeout_seconds:
        if observation['episode_id'] != episode:
            raise RuntimeError('Episode changed during release; no further command sent')
        action = {'joints_deg': joints, 'gripper_mm': 90.}
        client.action(observation, action)
        time.sleep(.2)
        observation = client.observe()
        evaluation = client.call('rebot_get_evaluation')
        if observation['episode_id'] != episode or evaluation['episode_id'] != episode:
            raise RuntimeError('Episode changed during release')
        result['steps'].append({'action': action, 'observation': observation,
                                'evaluation': evaluation})
        landed = (observation['gripper_mm'] >= 89.5 and not evaluation.get('held', True)
                  and not any(observation['finger_contacts'].values())
                  and 'floor' in evaluation.get('contacts', [])
                  and abs(evaluation.get('clearance_mm', float('inf'))) < 2
                  and evaluation.get('linear_speed_mm_s', float('inf')) < 15
                  and evaluation.get('angular_speed_deg_s', float('inf')) < 10)
        now = observation['simulation_time']
        settled_since = (now if settled_since is None else settled_since) if landed else None
        result['success'] = settled_since is not None and now-settled_since >= .3
        result['final_evaluation'] = evaluation
        result['elapsed_seconds'] = time.monotonic()-start
        write_json(directory/'release.json', result)
        if result['success']:
            return result
    result['error'] = 'Gripper opening and settled floor landing were not verified within ten seconds'
    write_json(directory/'release.json', result)
    return result
