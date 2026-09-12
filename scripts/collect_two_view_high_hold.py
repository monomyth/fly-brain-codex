"""Verified higher-lift demonstrations with measured recovery around the new setpoint."""
import argparse,importlib,json
from pathlib import Path
from datetime import datetime
from fly_brain.teacher import Teacher
from fly_brain.simulator import EpisodeSimulator,SimulatorError
from fly_brain.visual_dopamine.motor_experiment import task
from fly_brain.assets import home,write_json,sha256_file

class HighHoldTeacher(Teacher):
    version='cube_feedback_v7_high_hold'
    stages=('approach','descend','close','lift','hold','release','done')
    def __init__(self,*a,**k):super().__init__(*a,**k);self.curriculum_start=None;self.last_intervention=None
    def act(self,observation,evaluation):
        result=super().act(observation,evaluation);self.last_intervention=None
        if self.stage=='hold':
            if self.curriculum_start is None:self.curriculum_start=observation['simulation_time']
            index=int((observation['simulation_time']-self.curriculum_start)/1.2)
            heights=[120.,170.,125.,165.,130.]
            if index<len(heights):
                requested=heights[index];accepted=None
                for height in dict.fromkeys([requested,160.,155.,150.] if requested>150 else [requested]):
                    try: q=self._feedback_lift_target(observation,height)
                    except SimulatorError as error:
                        if 'No pose IK solution' not in str(error):raise
                        continue
                    accepted=height;break
                if accepted is not None:
                    self.last_execution_target={'joints_deg':q,'gripper_mm':0.}
                    self.last_intervention={'reason':'higher-hold recovery curriculum; only IK-verified perturbations are executed','requested_clearance_mm':requested,'executed_clearance_mm':accepted,'expert_clearance_mm':150.}
                else:self.last_intervention={'reason':'unreachable optional curriculum height skipped; expert goal retained','requested_clearance_mm':requested}
        return result

parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
project=Path(__file__).resolve().parents[1];run=Path(json.loads((project/'reports/two-view-current.json').read_text())['run']);base=json.loads((run/'plan.json').read_text())
if args.resume:
    plan=json.loads((run/'high-hold-plan.json').read_text());dataset=Path(plan['dataset']);positions=plan['positions']
    plan['resume_source_sha256']=sha256_file(__file__);write_json(run/'high-hold-plan.json',plan)
else:
    dataset=home()/'datasets/rebot-pick'/('two-view-high-hold-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
    positions=base['configurations']['train']+base['configurations']['validation']
    write_json(run/'high-hold-plan.json',{'dataset':str(dataset),'positions':positions,'native_demo_minimum_clearance_mm':135,'teacher_target_clearance_mm':150,'recovery_clearances_mm':[120,170,125,165,130],'source_sha256':sha256_file(__file__),'benchmark_minimum_unchanged_mm':100})
collection=importlib.import_module('fly_brain.collect');collection.Teacher=HighHoldTeacher
for i,xy in enumerate(positions):
    file=dataset/'collection-report.json';attempts=json.loads(file.read_text())['episodes'] if file.exists() else []
    if any(r['success'] and r['requested_task']['cube_xy_mm']==xy for r in attempts):continue
    configuration={**task(*xy,seed=35000+i),'lift_clearance_mm':135}
    with EpisodeSimulator(log=run/f'high-hold-native-{i}-attempt-{len(attempts)}.log') as client:
        result=collection.collect(client,[configuration],dataset,hz=5,images=True,seed=200+i,absolute_targets=True)
    print({'xy':xy,**result},flush=True)
    latest=json.loads(file.read_text())['episodes'][-1]
    if not latest['success']:raise RuntimeError('Higher-hold reference failed; inspect before continuing')
