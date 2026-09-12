"""Record neural prefixes followed by privileged, explicitly labeled recovery.

Only this collection script uses cube geometry and IK. It is never imported by
MotorPolicy. Every episode begins folded and qualifies only through native hold.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from fly_brain.assets import write_json, sha256_file
from fly_brain.dataset import EpisodeWriter
from fly_brain.simulator import EpisodeSimulator, configure_episode, release_for_cleanup
from fly_brain.teacher import grasp_pose
from fly_brain.visual_dopamine.motor_policy import MotorPolicy
from fly_brain.visual_dopamine.stage_run import actor_view


class RecoveryTeacher:
    def __init__(self, client, task, setup=None, hold_in_place=False):
        self.client, self.task = client, task
        self.hold_in_place = hold_in_place
        self.setup = setup
        self.setup_active = setup is not None
        self.label_path = {}
        if setup is not None:
            values = [setup.get('roll_deg', 0), setup.get('pitch_deg', 0), setup.get('height_mm', 31)]
            if not np.isfinite(values).all() or max(abs(values[0]), abs(values[1])) > 10 or not 27 <= values[2] <= 36:
                raise ValueError('Angled grasp setup needs finite angles within 10 degrees and height 27-36 mm')
        self.stage = 'approach'
        self.cache = {}
        self.open_pose = None
        self.approach = grasp_pose([350, 0], 20, 0)
        self.approach['position_mm'][2] = 181.
        self.canonical_approach = {**self.approach, 'position_mm': list(self.approach['position_mm'])}
        self.path = {'approach': self.solve(self.approach, 90)}
        self.label_path['approach'] = self.solve(self.canonical_approach, 90)
        self.recoveries = 0

    def with_setup_orientation(self, pose):
        rotation = Rotation.from_euler('xy', [self.setup.get('roll_deg', 0), self.setup.get('pitch_deg', 0)], degrees=True)
        return {**pose, 'quaternion_xyzw': (rotation * Rotation.from_quat(pose['quaternion_xyzw'])).as_quat().tolist()}

    def finish_setup(self):
        self.setup_active = False
        self.approach = self.canonical_approach
        self.path = dict(self.label_path)

    def supervision(self, observation, action):
        # Angled setup is a collection perturbation, not a hidden desired grasp angle.
        if not self.setup_active or self.stage in ('lift', 'hold', 'recovery_open'):
            self.label_path[self.stage] = list(action['joints_deg'])
            return action, self.label_path if self.setup is not None else self.path
        if self.stage == 'approach':
            pose, grip = self.canonical_approach, 90.
        else:
            pose = grasp_pose(observation['cube_pose']['position_mm'][:2], 20, 0)
            pose['position_mm'][2] = 55. if self.stage == 'recovery_align' else 27.
            grip = 0. if self.stage == 'close' else 90.
        label = {'joints_deg': list(self.solve(pose, grip)), 'gripper_mm': grip}
        self.label_path[self.stage] = label['joints_deg']
        return label, self.label_path

    def solve(self, pose, grip):
        key = tuple(np.round(pose['position_mm'], 2)) + tuple(np.round(pose['quaternion_xyzw'], 5)) + (float(grip),)
        if key not in self.cache:
            self.cache[key] = self.client.call('rebot_solve_pose', pose=pose, frame='grasp', gripper_mm=grip)['joints_deg']
        return self.cache[key]

    def reopen(self, observation):
        self.stage = 'recovery_open'
        self.recoveries += 1
        self.open_pose = {**observation['grasp_pose'], 'position_mm': list(observation['grasp_pose']['position_mm'])}
        self.open_pose['position_mm'][2] = max(65., self.open_pose['position_mm'][2] + 25.)

    def target(self, observation, evaluation):
        cube = observation['cube_pose']
        tool = observation['grasp_pose']
        aperture = observation['gripper_mm']
        contacts = observation['finger_contacts']
        # A yawed cube spans more than its side length between the fingers.
        bilateral = all(contacts.values())
        if evaluation.get('success'):
            self.stage = 'hold'
            return {'joints_deg': observation['joints_deg'], 'gripper_mm': 0.}
        if bilateral:
            self.stage = 'lift'
        elif self.stage in ('lift', 'hold'):
            self.reopen(observation)
        if evaluation.get('clearance_mm', 0) < 2 and evaluation.get('tilt_deg', 0) > 30 and not bilateral:
            raise RuntimeError('Overturned cube is outside this recovery teacher domain')
        desired = grasp_pose(cube['position_mm'][:2], 20, 0)
        desired['position_mm'][2] = self.setup.get('height_mm', 31) if self.setup_active else 27.
        if self.setup_active:
            desired = self.with_setup_orientation(desired)
        delta = np.asarray(tool['position_mm']) - desired['position_mm']
        angle = float((Rotation.from_quat(desired['quaternion_xyzw']).inv() * Rotation.from_quat(tool['quaternion_xyzw'])).magnitude())
        if self.stage == 'approach':
            if np.max(np.abs(np.asarray(observation['joints_deg']) - self.path['approach'])) < 1 and aperture > 89:
                self.stage = 'recovery_align'
        elif self.stage == 'recovery_open':
            if tool['position_mm'][2] >= 60 and aperture > 89:
                self.stage = 'recovery_align'
        elif self.stage == 'recovery_align':
            if np.linalg.norm(delta[:2]) < 2.5 and angle < .04 and aperture > 89 and abs(tool['position_mm'][2] - 55) < 3:
                self.stage = 'descend'
        elif self.stage == 'descend':
            if aperture < 70 and np.linalg.norm(delta[:2]) > 5:
                self.reopen(observation)
            elif np.linalg.norm(delta) < 3 and angle < .04:
                self.stage = 'close'
        elif self.stage == 'close':
            if aperture < 17 or np.linalg.norm(delta[:2]) > 5 or angle > .12:
                self.reopen(observation)
        if self.stage == 'approach':
            pose, grip = self.approach, 90.
        elif self.stage == 'recovery_open':
            pose, grip = self.open_pose, 90.
        elif self.stage == 'recovery_align':
            pose = {**desired, 'position_mm': [*desired['position_mm'][:2], 55.]}
            grip = 90.
        elif self.stage in ('descend', 'close'):
            pose, grip = desired, 0. if self.stage == 'close' else 90.
        else:
            # Preserve the actual grip-to-cube transform while leveling/lifting.
            def matrix(value):
                m = np.eye(4)
                m[:3, :3] = Rotation.from_quat(value['quaternion_xyzw']).as_matrix()
                m[:3, 3] = value['position_mm']
                return m
            target_cube = np.eye(4)
            target_cube[:3, 3] = [*cube['position_mm'][:2], 134.]
            goal = target_cube @ np.linalg.inv(matrix(cube)) @ matrix(tool)
            if not self.hold_in_place:
                goal[:2, 3] = [350., 0.]
            pose = {'position_mm': goal[:3, 3].tolist(), 'quaternion_xyzw': Rotation.from_matrix(goal[:3, :3]).as_quat().tolist()}
            grip = 0.
            if evaluation.get('clearance_mm', 0) >= 110:
                self.stage = 'hold'
        q = self.solve(pose, aperture if grip == 0 and bilateral else grip)
        self.path[self.stage] = q
        return {'joints_deg': list(q), 'gripper_mm': grip}


def collect_one(client, policy, case, dataset, source_hash, max_steps=190, runtime_matched=False):
    actual = configure_episode(client, {**case['task'], 'input_mode': 'state'})
    first = client.observe(images=True)
    if first['joints_deg'] != [0]*6 or first['gripper_mm'] != 0:
        raise ValueError('Recovery collection must start folded')
    teacher = RecoveryTeacher(client, actual, case.get('grasp_setup'), case.get('hold_in_place',False))
    policy_start_step = None if case.get('grasp_setup') else 0
    policy.reset(actor_view(first)); policy.act(actor_view(first)); policy.reset(actor_view(first))
    writer = EpisodeWriter(dataset, actual, 'teacher_assisted', 2, {
        'teacher_target_recorded': True, 'teacher_privileged_state': True,
        'vision_actor_cube_pose': False, 'execution_mode': 'absolute_targets',
        'supervision_contract': 'recorded-recovery-v1', 'teacher_version': 'policy_recovery_v2_contact_grip',
        'collector_source_sha256': source_hash, 'policy_checkpoint': str(policy.checkpoint),
        'policy_prefix_steps': case['prefix_steps'], 'case_id': case['id'],
        'hold_in_place':teacher.hold_in_place,
        'simulator': getattr(client, 'provenance', {}),
    })
    feature_rows = []; feature_frames = []; reflex_rows = []
    if runtime_matched:
        writer.manifest.update(runtime_profile='mlx-continuous-grip-2hz',
                               policy_model_sha256=sha256_file(policy.checkpoint/'model.npz'),
                               live_feature_contract='normalized-encoder-output-v1',
                               teacher_version='policy_recovery_v4_angled_setup' if case.get('grasp_setup') else 'policy_recovery_v3_local_takeover',
                               grasp_setup=case.get('grasp_setup'),
                               policy_schedule='teacher setup then neural recovery' if case.get('grasp_setup') else 'neural prefix then teacher recovery')
    result = {'success': False, 'hold_success': False, 'release_success': False, 'case_id': case['id']}
    try:
        client.call('rebot_experiment_control', action='start')
        client.connect(provenance='teacher_assisted', model_id='policy-recovery-v1')
        for step in range(max_steps):
            started = time.monotonic()
            observation = client.observe(images=True)
            evaluation = client.call('rebot_get_evaluation')
            if observation['phase'] not in ('running', 'completed'):
                raise RuntimeError('Recovery episode ended: '+observation['phase'])
            if policy_start_step is None and all(observation['finger_contacts'].values()) and evaluation.get('clearance_mm', 0) >= 110:
                policy_start_step = step
                teacher.finish_setup()
            if runtime_matched and policy_start_step is not None and step == policy_start_step + case['prefix_steps'] and not all(observation['finger_contacts'].values()):
                desired = grasp_pose(observation['cube_pose']['position_mm'][:2], 20, 0)
                angle = (Rotation.from_quat(desired['quaternion_xyzw']).inv() * Rotation.from_quat(observation['grasp_pose']['quaternion_xyzw'])).magnitude()
                if angle < .15:
                    if observation['gripper_mm'] < 70:
                        teacher.reopen(observation)
                    else:
                        teacher.stage = 'recovery_align'
            expert = teacher.target(observation, evaluation)
            learned = policy.act(actor_view(observation))
            selected = policy_start_step is not None and step < policy_start_step + case['prefix_steps'] and not evaluation.get('success')
            action = learned if selected else expert
            # No teacher override is hidden in the policy prefix.
            if not evaluation.get('success'):
                client.action(observation, action)
            label_target, label_path = teacher.supervision(observation, expert)
            writer.append(observation, action, stage=teacher.stage, evaluation=evaluation,
                          latency_ms=1000*(time.monotonic()-started), expert_target=label_target, expert_path=label_path,
                          intervention={'policy_selected': selected, 'policy_proposal': learned,
                                        'hold_in_place':teacher.hold_in_place,
                                        'action_applied': not bool(evaluation.get('success')),
                                        'teacher_correction': not selected and not teacher.setup_active,
                                        'setup_perturbation': teacher.setup_active,
                                        'policy_start_step': policy_start_step,
                                        'floor_limited': getattr(client, 'last_action_result', {}).get('floor_limited', False)})
            if runtime_matched:
                feature_rows.append(policy.core.features_for(observation))
                feature_frames.append(observation['frame_id'])
                if getattr(policy,'reflex_capture',None) is not None:
                    capture=policy.reflex_capture
                    raw=np.array(policy.core.gpu_state)[capture['indices'],0]
                    reflex_rows.append(np.tanh((raw-capture['mean'])/capture['std']).astype(np.float32))
            if evaluation.get('success'):
                result.update(success=True, hold_success=True, final_evaluation=evaluation)
                break
            if step % 10 == 0:
                print({'case': case['id'], 'step': step, 'teacher_stage': teacher.stage, 'policy_selected': selected,
                       'held': evaluation['held'], 'gripper_mm': round(observation['gripper_mm'], 1)}, flush=True)
            if runtime_matched:
                # Match the evaluation loop's post-command observation and minimum settling wait.
                time.sleep(.1)
                client.observe()
                client.call('rebot_get_evaluation')
                time.sleep(max(0., .5-(time.monotonic()-started)))
            else:
                time.sleep(max(.1, .5-(time.monotonic()-started)))
        else:
            result['error'] = 'Recovery decision limit reached'
    except (OSError, RuntimeError, ValueError, TimeoutError) as error:
        result['error'] = str(error)
    except KeyboardInterrupt:
        result['error'] = 'Interrupted'; result['interrupted'] = True
    finally:
        result['recoveries'] = teacher.recoveries
        result['policy_start_step'] = policy_start_step
        result['cleanup_error'] = release_for_cleanup(client)
        if runtime_matched:
            if len(feature_rows) != writer.manifest['steps']:
                result.update(success=False, hold_success=False, error='Recorded feature/observation alignment failed')
            feature_path = writer.directory/'live-features.npz'
            np.savez(feature_path, features=np.asarray(feature_rows, dtype=np.float32), frame_ids=np.asarray(feature_frames, dtype=np.int64))
            writer.manifest['live_features_sha256'] = sha256_file(feature_path)
        if getattr(policy,'reflex_capture',None) is not None:
            if len(reflex_rows)!=writer.manifest['steps']:
                result.update(success=False,hold_success=False,error='Reflex capture/observation alignment failed')
            extra=writer.directory/'reflex-features.npz'
            np.savez(extra,features=np.asarray(reflex_rows,dtype=np.float32),frame_ids=np.asarray(feature_frames,dtype=np.int64))
            writer.manifest['reflex_features_sha256']=sha256_file(extra)
            writer.manifest['reflex_input_manifest_sha256']=policy.reflex_capture['manifest_sha256']
        result['episode_directory'] = str(writer.finish(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text()); run = args.plan.parent; dataset = Path(plan['dataset'])
    dataset.mkdir(parents=True, exist_ok=True)
    path = dataset/'collection-report.json'
    results = json.loads(path.read_text())['episodes'] if path.exists() else []
    policy = MotorPolicy(plan['warm_start'], Path(plan['project'])/'data')
    count = 0
    for case in plan['collection_cases']:
        if any(r['case_id'] == case['id'] for r in results):
            continue
        if args.limit is not None and count >= args.limit:
            break
        with EpisodeSimulator(log=run/f"native-{case['id']}.log") as client:
            result = collect_one(client, policy, case, dataset, sha256_file(__file__))
        results.append(result); count += 1
        write_json(path, {'episodes': results, 'successes': sum(r['success'] for r in results)})
        print(result, flush=True)
        if result.get('interrupted') or 'Unlock the Mac' in result.get('error', ''):
            break


if __name__ == '__main__':
    main()
