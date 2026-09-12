"""Physical motor calibration and visual-response reward collection."""
import json
import time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from ..assets import write_json,home
from ..simulator import Simulator,configure_episode,release_for_cleanup
from .motor_circuit import prepare,MotorCircuit
from .motor_core import motor_action,stimulation
from .experiment import save_observation


def move_and_wait(client,action,timeout=8):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        obs=client.observe();client.action(obs,action)
        time.sleep(.12);after=client.observe()
        error=max(abs(np.asarray(after['joints_deg'])-action['joints_deg']))
        grip_error=abs(after['gripper_mm']-action['gripper_mm'])
        if error<.02 and grip_error<.05:return after
        if getattr(client,'last_action_result',{}).get('floor_limited'):
            return after
    raise TimeoutError(f"Motor pose did not settle: target={action}, actual_joints={after['joints_deg']}, actual_gripper={after['gripper_mm']}, phase={after['phase']}, frame={after['frame_id']}")


def task(x=350.,y=12.,seed=9200):
    return {'cube_size_mm':20,'cube_xy_mm':[x,y],'cube_yaw_deg':0,'initial_joints_deg':[0]*6,'initial_gripper_mm':0,
            'input_mode':'vision','hold_seconds':5,'timeout_seconds':180,'seed':seed}


def calibrate(output,root=None,circuit_directory=None):
    root=home(root);directory,meta=prepare(root);circuit=MotorCircuit(directory,root)
    output=Path(output)
    if output.exists():raise FileExistsError('Choose a new motor calibration directory')
    output.mkdir(parents=True);results=[]
    with Simulator(log=output/'native.log') as client:
        configure_episode(client,task())
        for axis in range(7):
            for direction in [-1,1]:
                if results:client.call('rebot_reset_episode');client.wait_ready()
                initial=client.observe()
                assert initial['joints_deg']==[0]*6 and initial['gripper_mm']==0
                client.call('rebot_experiment_control',action='start');client.connect(provenance='conventional',model_id='identified-motor-pool-stimulation-calibration')
                # A recorded setup pose permits both signs at joints whose folded pose is a limit.
                baseline=move_and_wait(client,{'joints_deg':[0,-10,-10,0,0,0],'gripper_mm':40})
                scores,motor=stimulation(circuit,axis,direction)
                before=client.observe(images=True)
                folder=output/(circuit.mapping[axis]['axis']+('-negative' if direction<0 else '-positive'));folder.mkdir()
                save_observation(folder,before)
                action=motor_action(before,scores,axis=axis,step_degrees=1,step_mm=2)
                after=move_and_wait(client,action)
                delta=np.r_[np.asarray(after['joints_deg'])-before['joints_deg'],after['gripper_mm']-before['gripper_mm']]
                expected=np.zeros(7);expected[axis]=direction*(2 if axis==6 else 1)
                success=bool(np.allclose(delta,expected,atol=.03))
                row={'axis':axis,'name':circuit.mapping[axis]['axis'],'direction':direction,'scores':scores.tolist(),
                     'stimulated_body_ids':circuit.ids[circuit.motor[motor>0]].tolist(),'actual_delta':delta.tolist(),
                     'expected_delta':expected.tolist(),'success':success,'starts_folded':True,
                     'conventional_setup_pose':{'joints_deg':baseline['joints_deg'],'gripper_mm':baseline['gripper_mm']}}
                results.append(row);write_json(folder/'result.json',row)
                client.release()
                write_json(output/'calibration.json',{'schema':'motor-pool-calibration-v1','experiment':'Direct stimulation of named MN pools; no image learning or pickup claimed',
                    'motor_circuit_id':meta['motor_circuit_id'],'simulator':client.provenance,'results':results,'successes':sum(r['success'] for r in results),'attempts':len(results)})
                print(f"{row['name']} {direction:+}: {'passed' if success else 'FAILED'} {delta.tolist()}",flush=True)
    return {'output':str(output),'successes':sum(r['success'] for r in results),'attempts':len(results)}


def reward_metrics(observation,evaluation):
    """Privileged task scoring, deliberately separate from neural policy inputs."""
    from ..teacher import grasp_pose
    cube=evaluation['cube_pose']['position_mm']
    size=float(evaluation.get('cube_size_mm',20.))
    yaw=float(Rotation.from_quat(evaluation['cube_pose']['quaternion_xyzw']).as_euler('xyz',degrees=True)[2])
    transport=all(observation['finger_contacts'].values()) and observation['gripper_mm']<=size+.5
    clearance=float(observation.get('goal',{}).get('clearance_mm',100.))+15 if transport else 0.
    target=grasp_pose(cube[:2],size,yaw,clearance=clearance)
    grasp=observation['grasp_pose']
    position_error=float(np.linalg.norm(np.asarray(grasp['position_mm'])-target['position_mm']))
    angle_error=float((Rotation.from_quat(target['quaternion_xyzw']).inv()*Rotation.from_quat(grasp['quaternion_xyzw'])).magnitude())
    # Close near the aligned cube; open while approaching. No stage is passed to the actor.
    near=position_error<12 and angle_error<.15
    desired_grip=0. if near or transport else 90.
    return {'position_error_mm':position_error,'orientation_error_rad':angle_error,
            'pose_cost':position_error+25*angle_error,'grip_error_mm':abs(observation['gripper_mm']-desired_grip),
            'desired_grip_evaluator_only':desired_grip,'cube_position_mm':cube,'transport':transport}



def fresh_physics_episode(client,configuration):
    """Remove old contact bodies before folding, then build a clean physics scene."""
    client.release()
    client.call('rebot_experiment_control',action='disable')
    client.call('rebot_playback',action='reset')
    client.call('rebot_configure_task',task=configuration)
    client.wait_ready()
    observation=client.observe()
    if observation['joints_deg']!=[0]*6 or observation['gripper_mm']!=0:
        raise RuntimeError('Fresh physics did not start folded')
    return observation


def collect_responses(output,root=None,limit=None,resume=False):
    """Conventional exploration collects camera inputs and physical directional rewards.

    IK initializes curriculum poses, but no IK target or action label enters the actor.
    Every scene starts folded. Most non-contact probes return to their measured pose;
    a displaced cube invalidates the scene rather than silently changing its inputs.
    """
    from ..teacher import Teacher
    root=home(root);directory,meta=prepare(root)
    output=Path(output)
    if output.exists() and not resume:raise FileExistsError('Choose a new motor response dataset or explicitly resume')
    output.mkdir(parents=True,exist_ok=True);records=[]
    if resume:
        previous=json.loads((output/'collection.json').read_text())
        if previous['motor_circuit_id']!=meta['motor_circuit_id']:raise ValueError('Cannot resume with a different motor circuit')
        records=previous['cases']
        for record in records:record.setdefault('simulator',previous['simulator'])
    scenarios=[(338.,-12.,'train'),(338.,12.,'train'),(354.,-12.,'train'),(354.,12.,'train'),(346.,-10.,'test'),(346.,10.,'test')]
    total=len(scenarios)*5
    with Simulator(log=output/'native.log') as client:
        for scene,(x,y,split) in enumerate(scenarios):
            requested=task(x,y,seed=9300+scene)
            if scene>0:
                fresh_physics_episode(client,requested);configured=client.call('rebot_get_task')['task']
            else:configured=configure_episode(client,requested)
            teacher=Teacher(client,configured,release=False)
            approach=np.asarray(teacher.path['approach']);pregrasp=np.asarray(teacher.path['pregrasp']);grasp=np.asarray(teacher.path['descend'])
            prefixes=[('folded',np.zeros(6),0.),('unfolding',approach*.5,45.),('approach',approach,75.),('above_cube',pregrasp,45.),('near_cube',grasp,24.)]
            for stage_index,(stage,q,grip) in enumerate(prefixes):
                if scene*len(prefixes)+stage_index<len(records):continue
                if limit is not None and len(records)>=limit:return {'dataset':str(output),'cases':len(records)}
                client.release();client.call('rebot_reset_episode');client.wait_ready()
                first=client.observe();assert first['joints_deg']==[0]*6 and first['gripper_mm']==0
                client.call('rebot_experiment_control',action='start');client.connect(provenance='conventional',model_id='motor-response-reward-calibration')
                # Follow the conventional approach before low poses to avoid floor tunnelling.
                if stage in ['above_cube','near_cube']:
                    move_and_wait(client,{'joints_deg':approach.tolist(),'gripper_mm':90.},timeout=40)
                if stage=='near_cube':move_and_wait(client,{'joints_deg':pregrasp.tolist(),'gripper_mm':90.},timeout=40)
                baseline=move_and_wait(client,{'joints_deg':q.tolist(),'gripper_mm':grip},timeout=40)
                if max(abs(np.asarray(baseline['joints_deg'])-q))>.1:raise RuntimeError('Curriculum setup was floor-limited')
                baseline_action={'joints_deg':baseline['joints_deg'],'gripper_mm':baseline['gripper_mm']}
                observation=client.observe(images=True);initial_eval=client.call('rebot_get_evaluation')
                initial_metrics=reward_metrics(observation,initial_eval)
                folder=output/f'case-{len(records):03d}'
                if folder.exists():
                    import uuid
                    invalid=output/('invalid-'+folder.name+'-'+uuid.uuid4().hex[:8]);folder.rename(invalid)
                    write_json(invalid/'INVALID.json',{'reason':'Interrupted/incomplete counterfactual scene; excluded from training'})
                folder.mkdir();save_observation(folder,observation)
                outcomes=[]
                for axis in range(7):
                    directions=[]
                    for direction in [-1,1]:
                        if stage=='near_cube' and (axis>0 or direction>0):
                            reset=fresh_physics_episode(client,configured)
                            client.call('rebot_experiment_control',action='start');client.connect(provenance='conventional',model_id='motor-response-reward-calibration')
                            move_and_wait(client,{'joints_deg':approach.tolist(),'gripper_mm':90.},timeout=40)
                            move_and_wait(client,{'joints_deg':pregrasp.tolist(),'gripper_mm':90.},timeout=40)
                            move_and_wait(client,baseline_action,timeout=40)
                        before=client.observe();before_eval=client.call('rebot_get_evaluation');a=reward_metrics(before,before_eval)
                        scores=np.zeros(7);scores[axis]=direction
                        action=motor_action(before,scores,axis=axis,step_degrees=1,step_mm=2)
                        actual=move_and_wait(client,action)
                        after_eval=client.call('rebot_get_evaluation');b=reward_metrics(actual,after_eval)
                        metric='grip_error_mm' if axis==6 else 'pose_cost'
                        progress=a[metric]-b[metric]
                        reward=1. if progress>.01 else 0. if progress<-.01 else .5
                        directions.append({'axis':axis,'direction':direction,'action':action,'reward':reward,'progress':progress,
                            'before':a,'after':b,'actual_joints_deg':actual['joints_deg'],'actual_gripper_mm':actual['gripper_mm'],
                            'delay_s':actual['simulation_time']-before['simulation_time'],'floor_limited':getattr(client,'last_action_result',{}).get('floor_limited',False)})
                        write_json(folder/f'probe-{axis}-{direction:+}.json',directions[-1])
                        if stage!='near_cube':
                            move_and_wait(client,baseline_action)
                            reset_eval=client.call('rebot_get_evaluation')
                            if np.linalg.norm(np.asarray(reset_eval['cube_pose']['position_mm'])-initial_metrics['cube_position_mm'])>1:
                                raise RuntimeError('Probe displaced cube; camera/reward pairing is invalid')
                    outcomes.append(directions)
                record={'id':folder.name,'split':split,'simulator':client.provenance,'stage_for_audit_only':stage,'task_for_evaluator_only':configured,
                        'starts_folded':True,'conventional_prefix':baseline_action,'initial_metrics':initial_metrics,'outcomes':outcomes}
                write_json(folder/'outcomes.json',record);records.append(record);client.release()
                write_json(output/'collection.json',{'schema':'seven-motor-response-dataset-v1','motor_circuit_id':meta['motor_circuit_id'],
                    'cases':records,'simulator':client.provenance,'images_recorded':True,'task_scope':'Single-joint progress toward grasp pose and aperture; no successful pickup labels',
                    'collection_method':'IK-seeded body exploration with physically measured action rewards; actor inputs are camera pixels only'})
                print(f'Recorded motor response {len(records)}/{total}: {stage}, cube={x,y}',flush=True)
    return {'dataset':str(output),'cases':len(records),'physical_actions':len(records)*14}
