"""Stage diagnostics with recorded conventional setup and a clean neural handoff.

Stage successes are never reported as full autonomous pickups. The native
five-second evaluator must start with no credit at the handoff.
"""
import base64
import time
from pathlib import Path
import numpy as np
from ..assets import write_json
from ..simulator import configure_episode,release_for_cleanup
from ..teacher import grasp_pose
from ..activity import ActivityPublisher
from .experiment import save_observation


def actor_view(observation):
    keys=('images','joints_deg','gripper_mm','finger_contacts','simulation_time','frame_id','episode_id')
    return {**{k:observation[k] for k in keys},'input_mode':'vision'}


def unsupported_grasp(evaluation):
    contacts=set(evaluation.get('contacts',[]))
    return (bool(evaluation.get('held')) and evaluation.get('clearance_mm',0)>=10
            and contacts=={'finger_left_link','finger_right_link'} and not evaluation.get('dropped',False))


def validate_handoff(stage,evaluation):
    if evaluation.get('success') or evaluation.get('hold_seconds',0)>0:
        raise ValueError('Teacher setup has already earned hold credit')
    if stage=='lift_hold' and not unsupported_grasp(evaluation):
        raise ValueError('Lift/hold setup lacks a verified unsupported grasp')


def warmup_history(policy,history):
    """Sample recorded sensory history at the nominal cadence, retaining endpoints."""
    if not getattr(policy.core,'temporal_config',None):return history[-1:]
    selected=[];last_time=None;last_frame=None;seen_time=None;seen_frame=None
    for index,item in enumerate(history):
        observation=item[0];now=observation['simulation_time'];frame=observation['frame_id']
        if seen_time is not None and (now<seen_time or frame<seen_frame):raise ValueError('Setup history is not chronological')
        seen_time=now;seen_frame=frame
        if last_time is not None and (now<=last_time or frame<=last_frame):continue
        if last_time is None or now-last_time>=1/policy.hz or index==len(history)-1:
            selected.append(item);last_time=now;last_frame=frame
    return selected


def run_stage(client,policy,requested,stage,output,offset_mm=(0.,0.),max_steps=60):
    if stage not in ('grasp','lift_hold'):raise ValueError('Unknown diagnostic stage')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    result={'stage':stage,'stage_success':False,'full_task_success':False,'scope':'Conventional setup followed by neural stage control; not a full autonomous pickup',
            'setup_steps':[],'neural_steps':[],'checkpoint':str(policy.checkpoint)}
    history=[];activity=None
    def solve(xy,z,opening):
        pose=grasp_pose(xy,20,0);pose['position_mm'][2]=z
        return client.call('rebot_solve_pose',pose=pose,frame='grasp',gripper_mm=opening)['joints_deg']
    def drive(q,opening,label,predicate,timeout=25):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            started=time.monotonic();observation=client.observe(images=True);evaluation=client.call('rebot_get_evaluation')
            folder=output/'setup'/f'{len(history):03d}';folder.mkdir(parents=True)
            record=save_observation(folder,observation);history.append((record,folder))
            action={'joints_deg':list(q),'gripper_mm':opening}
            result['setup_steps'].append({'frame_id':observation['frame_id'],'stage':label,'action':action,'evaluation':evaluation})
            if predicate(observation,evaluation):return observation,evaluation
            client.action(observation,action)
            time.sleep(max(.05,.25-(time.monotonic()-started)))
        raise TimeoutError('Conventional stage setup did not settle: '+label)
    def settled(q,opening):
        return lambda o,e: np.max(np.abs(np.asarray(o['joints_deg'])-q))<.2 and abs(o['gripper_mm']-opening)<.2
    try:
        task=configure_episode(client,{**requested,'input_mode':'state'})
        first=client.observe()
        if first['joints_deg']!=[0]*6 or first['gripper_mm']!=0:raise ValueError('Stage fixtures must begin folded')
        result['task']=task;result['starts_folded']=True;result['offset_mm']=list(offset_mm)
        xy=np.asarray(task['cube_xy_mm']);approach=solve(xy,181,90)
        near=solve(xy+np.asarray(offset_mm),47,90);grasp=solve(xy,27,90)
        client.call('rebot_experiment_control',action='start');client.connect(provenance='conventional',model_id='stage-fixture-v1')
        drive(approach,90,'approach',settled(approach,90))
        if stage=='grasp':drive(near,90,'above_cube',settled(near,90))
        else:
            drive(grasp,90,'align',settled(grasp,90))
            observation,evaluation=drive(grasp,0,'close',lambda o,e: all(o['finger_contacts'].values()) and o['gripper_mm']<=20.5)
            pose={**observation['grasp_pose'],'position_mm':list(observation['grasp_pose']['position_mm'])}
            pose['position_mm'][2]+=20
            low=client.call('rebot_solve_pose',pose=pose,frame='grasp',gripper_mm=observation['gripper_mm'])['joints_deg']
            drive(low,0,'verify_grasp',lambda o,e: unsupported_grasp(e) and np.max(np.abs(np.asarray(o['joints_deg'])-low))<.2)
        evaluation=client.call('rebot_get_evaluation');validate_handoff(stage,evaluation)
        result['handoff_evaluation']=evaluation
        client.release();client.call('rebot_experiment_control',action='pause')
        # Pausing during neural warmup cannot accumulate physical hold time.
        policy.reset(actor_view(history[0][0]))
        warm=warmup_history(policy,history);result['warmup_observations']=len(warm)
        for observation,directory in warm:
            observation=actor_view(observation)
            observation['images']=[{**image,'jpeg_base64':base64.b64encode((directory/image['file']).read_bytes()).decode('ascii')} for image in observation['images']]
            policy.act(observation)
        client.call('rebot_experiment_control',action='resume');time.sleep(.05)
        client.connect(provenance='malecns',model_id=policy.metadata['name'])
        activity=ActivityPublisher.for_policy(policy,client)
        result['neural_handoff_frame']=evaluation['frame_id'];stable_grasp_samples=0
        for index in range(max_steps):
            started=time.monotonic();before=client.observe(images=True)
            action=policy.act(actor_view(before))
            if activity:activity.publish(policy.state,before)
            client.action(before,action)
            folder=output/'neural'/f'{index:03d}';folder.mkdir(parents=True);save_observation(folder,before)
            time.sleep(.1);after=client.observe();evaluation=client.call('rebot_get_evaluation')
            stable_grasp_samples=stable_grasp_samples+1 if unsupported_grasp(evaluation) else 0
            qualified=bool(evaluation.get('success')) if stage=='lift_hold' else stable_grasp_samples>=2
            decision={'step':index,'action':action,'evaluation':evaluation,'loop_ms':1000*(time.monotonic()-started)}
            result['neural_steps'].append(decision);write_json(folder/'decision.json',decision)
            result['stage_success']=qualified;result['final_evaluation']=evaluation
            if qualified:break
            if index%10==0:print({'stage':stage,'step':index,'held':evaluation.get('held'),'clearance_mm':evaluation.get('clearance_mm')},flush=True)
        else:result['stop_reason']='Neural stage decision limit reached'
    except KeyboardInterrupt:result['error']='Interrupted';result['interrupted']=True
    except (OSError,ValueError,RuntimeError,TimeoutError) as error:result['error']=str(error)
    finally:
        cleanup=release_for_cleanup(client)
        if activity:activity.close()
        if cleanup:result['cleanup_error']=cleanup
        write_json(output/'result.json',result)
    return result
