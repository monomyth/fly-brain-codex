"""Clearly labeled teacher-assisted corrections around states visited by the image policy."""
import importlib,json
from datetime import datetime
from pathlib import Path
from collect_direct_pickups import DirectTeacher
from fly_brain.simulator import EpisodeSimulator
from fly_brain.visual_dopamine.motor_experiment import task
from fly_brain.visual_dopamine.motor_policy import MotorPolicy
from fly_brain.assets import write_json,home

project=Path(__file__).resolve().parents[1];s=json.loads((project/'reports/two-view-current.json').read_text());run=Path(s['run'])
dataset=home()/'datasets/rebot-pick'/('two-view-corrections-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
policy=MotorPolicy(s['checkpoint'],home())
collection=importlib.import_module('fly_brain.collect');collection.Teacher=DirectTeacher
schedule={'close':1.,'lift':1.,'hold':1.,'release':1.,'done':1.}
write_json(run/'correction-plan.json',{'dataset':str(dataset),'checkpoint':s['checkpoint'],'teacher_probability':.8,'teacher_mixing_schedule':schedule,'positions':[[350,0],[350,0],[340,-12],[360,12]],'purpose':'Model visits some approach/descent states; privileged teacher completes grasp/hold and supplies labels. Not autonomous evaluation.'})
for i,(x,y) in enumerate([[350,0],[350,0],[340,-12],[360,12]]):
    with EpisodeSimulator(log=run/f'correction-native-{i}.log') as client:
        print(collection.collect(client,[task(x,y,seed=32000+i)],dataset,hz=2,images=True,policy=policy,teacher_probability=.8,seed=100+i,absolute_targets=True,mixing_schedule=schedule),flush=True)
print('Corrections:',dataset,flush=True)
