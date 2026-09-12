"""Summarize saved comparison outcomes without starting the simulator."""
import json
from pathlib import Path

import numpy as np
from fly_brain.assets import write_json


def summarize(directory):
    report = json.loads((directory / 'evaluation.json').read_text())
    episodes = report['episodes']
    steps = [step for episode in episodes for step in episode['steps']]
    timing = {
        key: {'median': float(np.median([s['timings_ms'][key] for s in steps])),
              'p95': float(np.percentile([s['timings_ms'][key] for s in steps], 95))}
        for key in steps[0]['timings_ms']
    } if steps else {}
    failures = [e for e in episodes if not e['success']]
    schema_errors = []
    observations = 0
    image_count = 0
    allowed = {'images', 'joints_deg', 'gripper_mm', 'finger_contacts', 'simulation_time', 'frame_id', 'episode_id'}
    for path in sorted(directory.glob('episode-*/step-*/observation.json')):
        value = json.loads(path.read_text())
        observations += 1
        image_count += len(value['images'])
        if set(value) - allowed:
            schema_errors.append({'path': str(path), 'unexpected_fields': sorted(set(value) - allowed)})
        if ([im['name'] for im in value['images']] != ['Front', 'Gripper']
                or any(im['rig_revision'] != 'front336l-gripper305-rgb-v4' for im in value['images'])):
            schema_errors.append({'path': str(path), 'error': 'Unexpected camera pair or rig'})
    return {
        'attempts': report['attempts'], 'successes': report['successes'],
        'failed_without_any_native_grip': sum(not any(s['evaluation']['held'] for s in e['steps']) for e in failures),
        'failed_after_some_native_grip': sum(any(s['evaluation']['held'] for s in e['steps']) for e in failures),
        'failed_with_floor_limited_commands': sum(any(s['floor_limited'] for s in e['steps']) for e in failures),
        'runtime_errors': [{'episode': i, 'error': e['error']} for i, e in enumerate(episodes) if 'error' in e],
        'timings_ms': timing, 'recorded_observations': observations, 'recorded_images': image_count,
        'actor_observation_schema_errors': schema_errors,
        'outcomes': [{'episode': i, 'success': e['success'], 'steps': len(e['steps']),
                      'cube_xy_mm': e.get('task_for_evaluator_only', {}).get('cube_xy_mm'),
                      'max_hold_seconds': max([s['evaluation']['hold_seconds'] for s in e['steps']] or [0]),
                      'max_clearance_mm': max([s['evaluation']['clearance_mm'] for s in e['steps']] or [0]),
                      'stop_reason': e.get('error') or e.get('stop_reason') or 'Native hold completed'}
                     for i, e in enumerate(episodes)],
    }


if __name__ == '__main__':
    batch = Path(__file__).resolve().parents[1] / 'reports/temporal-control-20260911'
    result = {name: summarize(batch / name) for name in ['full-temporal-20', 'full-temporal-mlx-20']}
    write_json(batch / 'failure-analysis.json', result)
    for name, value in result.items():
        print(name, json.dumps({k: v for k, v in value.items() if k not in ['outcomes', 'timings_ms']}))
