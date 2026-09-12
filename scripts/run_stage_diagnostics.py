"""Run a declared stage test battery against one frozen checkpoint."""
import argparse,json
from pathlib import Path
from fly_brain.assets import write_json
from fly_brain.simulator import EpisodeSimulator
from fly_brain.visual_dopamine.motor_policy import MotorPolicy
from fly_brain.visual_dopamine.motor_experiment import task
from fly_brain.visual_dopamine.stage_run import run_stage

p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--max-steps',type=int,default=60);a=p.parse_args()
root=Path(__file__).resolve().parents[1];a.output.mkdir(parents=True,exist_ok=False)
cases=[{'stage':'grasp','xy':[350,0],'offset':o} for o in [[0,0],[3,0],[0,3],[-3,0]]]+[{'stage':'lift_hold','xy':xy,'offset':[0,0]} for xy in [[350,0],[340,-12]]]
write_json(a.output/'plan.json',{'checkpoint':str(a.checkpoint.resolve()),'cases':cases,'scope':'Isolated stage diagnostics with explicitly conventional setup','max_steps':a.max_steps})
policy=MotorPolicy(a.checkpoint,root/'data');results=[]
for i,case in enumerate(cases):
    with EpisodeSimulator(log=a.output/f'native-{i}.log') as client:
        result=run_stage(client,policy,task(*case['xy'],seed=43000+i),case['stage'],a.output/f'case-{i:02d}',case['offset'],a.max_steps)
    results.append(result)
    report={'checkpoint':str(a.checkpoint.resolve()),'scope':'Stage successes after conventional setup, not full autonomous pickups','results':results}
    write_json(a.output/'stages.json',report)
    print({'case':i,'stage':case['stage'],'stage_success':result['stage_success'],'error':result.get('error')},flush=True)
    if result.get('interrupted'):break
