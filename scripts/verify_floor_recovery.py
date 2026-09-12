"""Live regression: a controller can push into the floor, stop, and raise again."""
import argparse
import json
from pathlib import Path
import time
from fly_brain.assets import write_json
from fly_brain.simulator import Simulator, configure_episode


def verify(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    samples = []
    with Simulator(log=output / 'native.log') as client:
        configure_episode(client, {'cube_size_mm': 20, 'cube_xy_mm': [700, 300],
            'initial_joints_deg': [0, -95, -95, 10, 0, 90], 'initial_gripper_mm': 90,
            'input_mode': 'state', 'timeout_seconds': 60})
        client.call('rebot_experiment_control', action='start')
        client.connect(model_id='floor-recovery-check')
        token = client.session['token']
        down = {'joints_deg': [0, -179, -95, 10, 0, 90], 'gripper_mm': 90}
        for number in range(15):
            client.action(client.observe(), down)
            assert client.last_action_result['floor_limited']
            time.sleep(.2)
            state = client.call('rebot_get_state')
            assert state['experiment']['phase'] == 'running'
            assert state['experiment']['owner'] == 'conventional'
            clearance = state['floor']['minimum_robot_height_mm'] - state['floor']['height_mm']
            assert clearance >= -0.0001, state['floor']
            samples.append({'stage': 'down', 'clearance_mm': clearance, 'joints_deg': state['joints_deg']})
        assert samples[-1]['clearance_mm'] < .005, samples[-1]
        up = {'joints_deg': list(state['joints_deg']), 'gripper_mm': 89}
        up['joints_deg'][0] += 10
        up['joints_deg'][1] += 10
        for number in range(8):
            client.action(client.observe(), up)
            time.sleep(.2)
            state = client.call('rebot_get_state')
            clearance = state['floor']['minimum_robot_height_mm'] - state['floor']['height_mm']
            assert clearance >= -0.0001
            samples.append({'stage': 'up', 'clearance_mm': clearance, 'joints_deg': state['joints_deg']})
        assert samples[-1]['clearance_mm'] > 10, samples[-1]
        assert state['experiment']['phase'] == 'running'
        assert client.session['token'] == token
        result = {'passed': True, 'same_controller_session': True,
                  'contact_clearance_mm': samples[14]['clearance_mm'],
                  'raised_clearance_mm': samples[-1]['clearance_mm'], 'samples': samples}
        write_json(output / 'result.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='reports/floor-recovery-verification')
    args = parser.parse_args()
    result = verify(args.output)
    print(json.dumps({k: v for k, v in result.items() if k != 'samples'}, indent=2))
