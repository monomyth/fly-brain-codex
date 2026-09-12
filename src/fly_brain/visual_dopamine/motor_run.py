"""Live seven-motor control, with independent geometry/reward evaluation."""
import time
from pathlib import Path
import numpy as np
from ..assets import write_json
from ..simulator import configure_episode,current_episode_task,release_for_cleanup
from ..activity import ActivityPublisher
from .experiment import save_observation
from .motor_experiment import reward_metrics
from .motor_release import open_and_drop


def update_stagnation(previous,movement,evaluator,now):
    """A stable supported hold is intended behaviour, not a motion failure."""
    holding=bool(evaluator.get('held')) or float(evaluator.get('hold_seconds',0))>0
    if movement>=.03 or holding:return None
    return now if previous is None else previous


def resolve_settle_seconds(metadata,override):
    value=override if override is not None else metadata.get('deployment_settle_seconds')
    if value is not None and (not isinstance(value,(int,float)) or not np.isfinite(value) or not .05<=value<=.5):raise ValueError('Settle override must be between 0.05 and 0.5 seconds')
    return value


def run(client,policy,tasks,output,current_episode=False,expected_episode_id=None,max_steps=120,learn=False,settle_seconds=None,stop_after_failures=None,release=True):
    if learn and getattr(policy,'metadata',{}).get('touch_gated_hold'):raise ValueError('Touch-gated banks are trained offline; disable Learn from reward')
    if not tasks or not 1<=max_steps<=1000:raise ValueError('At least one task and 1–1000 steps required')
    if stop_after_failures is not None and (not isinstance(stop_after_failures,int) or not 1<=stop_after_failures<=len(tasks)):raise ValueError('Invalid failure stopping limit')
    settle_seconds=resolve_settle_seconds(getattr(policy,'metadata',{}),settle_seconds)
    output=Path(output);output.mkdir(parents=True,exist_ok=True);episodes=[]
    for number,requested in enumerate(tasks):
        directory=output/f'episode-{number:03d}';directory.mkdir();row={'success':False,'steps':[],'trial_started':False};activity=None
        try:
            actual=current_episode_task(client,'vision',expected_episode_id) if current_episode else configure_episode(client,{**requested,'input_mode':'vision'})
            observation=client.observe(images=True);policy.reset(observation);policy.act(observation);policy.reset(observation)
            activity=ActivityPublisher.for_policy(policy,client)
            row['task_for_evaluator_only']=actual;row['initial_metrics']=reward_metrics(observation,client.call('rebot_get_evaluation'))
            client.call('rebot_experiment_control',action='start');client.connect(provenance='malecns',model_id=policy.metadata['name']);row['trial_started']=True
            stagnant_since=None;last_command_time=time.monotonic()
            for step in range(max_steps):
                started=time.monotonic();before=client.observe(images=True);observed_at=time.monotonic();evaluator_before=client.call('rebot_get_evaluation')
                metrics_before=reward_metrics(before,evaluator_before)
                inference_started=time.monotonic();action=policy.act(before,explore=learn);inference_finished=time.monotonic()
                if activity:activity.publish(policy.state,before)
                record_started=time.monotonic();folder=directory/f'step-{step:03d}';folder.mkdir();save_observation(folder,before);record_finished=time.monotonic()
                command_started=time.monotonic();client.action(before,action);command_finished=time.monotonic()
                inter_command_ms=1000*(command_finished-last_command_time);last_command_time=command_finished
                settle=policy.metadata.get('architecture',{}).get('settle_before_observation',False)
                pause=settle_seconds if settle_seconds is not None else (max(.25,1/policy.hz+.05) if settle else .25)
                time.sleep(pause)
                after=client.observe();evaluator=client.call('rebot_get_evaluation');metrics=reward_metrics(after,evaluator)
                progress=metrics_before['pose_cost']-metrics['pose_cost']+.1*(metrics_before['grip_error_mm']-metrics['grip_error_mm'])
                reward=float(np.clip(.5+progress/10,0,1))
                if metrics['transport'] and not metrics_before['transport']:reward=1.
                if metrics_before['transport'] and not metrics['transport']:reward=0.
                if evaluator.get('success'):reward=1.
                feedback=policy.feedback(reward,after['simulation_time']-before['simulation_time'],learn)
                if activity:activity.publish(policy.state,after)
                decision={'step':step,'response_bank':getattr(policy,'active_motor_bank','single'),'scores':policy.last_scores.tolist(),'choices':policy.last_choices.tolist(),'action':action,
                    'raw_neural_action':getattr(policy,'raw_neural_action',action),'actuator_filter':getattr(getattr(policy,'motor_output_filter',None),'parameters',None),
                    'actual_joints_deg':after['joints_deg'],'actual_gripper_mm':after['gripper_mm'],'metrics':metrics,
                    'evaluation':evaluator,'feedback':feedback,'floor_limited':getattr(client,'last_action_result',{}).get('floor_limited',False),
                    'timings_ms':{'observation':1000*(observed_at-started),'inference':1000*(inference_finished-inference_started),'recording':1000*(record_finished-record_started),'command':1000*(command_finished-command_started),'between_commands':inter_command_ms},'settle_seconds':pause,
                    'loop_ms':1000*(time.monotonic()-started)}
                row['steps'].append(decision);write_json(folder/'decision.json',decision)
                row['final_metrics']=metrics;row['final_evaluation']=evaluator
                # Only the independent native hold evaluator can qualify pickup/hold.
                row['hold_success']=bool(evaluator.get('success'))
                if row['hold_success']:
                    if release:
                        print('Five-second hold completed; opening gripper and verifying drop',flush=True)
                        row['release']=open_and_drop(client,evaluator,directory)
                        row['release_success']=row['release']['success']
                        row['success']=row['release_success']
                        if not row['success']:row['error']=row['release']['error']
                    else:row['success']=True
                    break
                movement=np.linalg.norm(np.r_[np.asarray(after['joints_deg'])-before['joints_deg'],after['gripper_mm']-before['gripper_mm']])
                stagnant_since=update_stagnation(stagnant_since,movement,evaluator,time.monotonic())
                if stagnant_since is not None and time.monotonic()-stagnant_since>=8:
                    row['stop_reason']='No effective motion for eight seconds without a held cube';break
                if step%10==0:print(f"Episode {number+1}, step {step+1}: distance {metrics['position_error_mm']:.1f} mm, orientation {metrics['orientation_error_rad']:.2f} rad",flush=True)
                time.sleep(max(0,1/policy.hz-(time.monotonic()-started)))
            else:row['stop_reason']='Decision limit reached'
        except KeyboardInterrupt:row.update(error='Interrupted',interrupted=True)
        except (OSError,ValueError,RuntimeError,TimeoutError) as error:
            row['error']=str(error)
            if not row['trial_started']:
                row['startup_error']='screen_locked' if 'Unlock the Mac before starting live simulation' in str(error) else 'startup_error'
        finally:
            cleanup=release_for_cleanup(client)
            if activity:activity.close()
            if cleanup:row['cleanup_error']=cleanup
            episodes.append(row)
            report={'task_kind':'seven_motor_control','task_scope':'Camera-driven pickup and native hold; deterministic opening with verified floor landing' if release else 'Camera-driven pickup and native hold only',
                'release_required':release,'hold_successes':sum(e.get('hold_success',False) for e in episodes),'release_successes':sum(e.get('release_success',False) for e in episodes),
                'checkpoint':str(policy.checkpoint),'settle_seconds_override':settle_seconds,'learning_enabled':learn,'attempts':sum(e['trial_started'] and not e.get('interrupted',False) for e in episodes),
                'successes':sum(e['success'] and not e.get('interrupted',False) for e in episodes),
                'startup_failures':sum(bool(e.get('startup_error')) for e in episodes),'interrupted_attempts':sum(bool(e.get('interrupted')) for e in episodes),
                'planned_attempts':len(tasks),'records':len(episodes),'episodes':episodes}
            if row.get('startup_error'):report['execution_blocked']=row['startup_error']
            write_json(output/'evaluation.json',report)
        if row.get('interrupted') or row.get('startup_error'):break
        if stop_after_failures is not None and sum(e['trial_started'] and not e['success'] and not e.get('interrupted',False) for e in episodes)>=stop_after_failures:
            report.update(stopped_early_after_failures=stop_after_failures,planned_attempts=len(tasks))
            write_json(output/'evaluation.json',report)
            break
    if learn and report['attempts']:
        report['learned_checkpoint']=policy.save(output/'checkpoint',{'attempts':report['attempts'],'successes':report['successes']})
        write_json(output/'evaluation.json',report)
    return report
