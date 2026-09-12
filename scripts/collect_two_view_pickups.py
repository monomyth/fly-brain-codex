"""Repeatable Front/Gripper pickup dataset with whole-position held-out splits."""
import argparse,importlib,json
from pathlib import Path
from datetime import datetime
from collect_direct_pickups import DirectTeacher
from collect_visual_grounding import GroundingTeacher
from fly_brain.simulator import EpisodeSimulator
from fly_brain.visual_dopamine.motor_experiment import task
from fly_brain.assets import home,write_json,sha256_file
from fly_brain.cameras import CURRENT_RIG


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path);p.add_argument('--limit',type=int);p.add_argument('--hz',type=float,default=5.)
    a=p.parse_args();project=Path(__file__).resolve().parents[1]
    run=a.run or project/'reports'/('two-view-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
    run=run.resolve();run.mkdir(parents=True,exist_ok=True);planfile=run/'plan.json'
    if planfile.exists():plan=json.loads(planfile.read_text())
    else:
        configurations={'train':[[350,0],[340,-12],[340,12],[360,-12],[360,12]],'validation':[[345,8],[355,-8]],'test':[[345,-4],[355,4]]}
        cases=[{'xy':xy,'split':split,'prefix':0} for split,positions in configurations.items() for xy in positions]
        cases += [{'xy':xy,'split':'train','prefix':direction} for xy in configurations['train'] for direction in [-1,1]]
        plan={'camera_rig_revision':CURRENT_RIG,'dataset':str(home()/'datasets/rebot-pick'/run.name),'configurations':configurations,'cases':cases,'hz':a.hz,'cube_size_mm':20,'initial_joints_deg':[0]*6,'initial_gripper_mm':0,
              'collection_source_sha256':sha256_file(__file__),'direct_teacher_source_sha256':sha256_file(Path(__file__).with_name('collect_direct_pickups.py')),'grounding_teacher_source_sha256':sha256_file(Path(__file__).with_name('collect_visual_grounding.py'))}
        write_json(planfile,plan)
    assert plan['camera_rig_revision']==CURRENT_RIG
    write_json(project/'reports/two-view-current.json',{'run':str(run),'dataset':plan['dataset']})
    statusfile=run/'collection-status.json';status=json.loads(statusfile.read_text()) if statusfile.exists() else {'completed_cases':[],'attempts':[]}
    collection=importlib.import_module('fly_brain.collect');attempts=0
    for i,case in enumerate(plan['cases']):
        if i in status['completed_cases']:continue
        if a.limit is not None and attempts>=a.limit:break
        collection.Teacher=DirectTeacher if not case['prefix'] else GroundingTeacher
        d=case['prefix'];GroundingTeacher.perturbation=[d*6,-6,-8,2,d*3,-d*4]
        print({'case':i,**case,'dataset':plan['dataset']},flush=True)
        with EpisodeSimulator(log=run/f'native-case-{i:02d}-attempt-{len(status["attempts"])}.log') as client:
            result=collection.collect(client,[task(*case['xy'],seed=21000+i)],plan['dataset'],hz=plan['hz'],images=True,seed=i,absolute_targets=True)
        latest=json.loads((Path(plan['dataset'])/'collection-report.json').read_text())['episodes'][-1]
        status['attempts'].append({'case':i,'success':latest['success'],'hold_success':latest.get('hold_success',False),'error':latest.get('error'),'episode_directory':latest.get('episode_directory')})
        if latest['success'] or latest.get('hold_success'):status['completed_cases'].append(i)
        write_json(statusfile,status);attempts+=1
        if not(latest['success'] or latest.get('hold_success')):raise RuntimeError('Reference pickup failed; inspect this attempt before proceeding')
    print(json.dumps({'run':str(run),'completed':len(status['completed_cases']),'planned':len(plan['cases'])}),flush=True)

if __name__=='__main__':main()
