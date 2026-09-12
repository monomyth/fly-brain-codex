import json
from pathlib import Path
from datetime import datetime
from fly_brain.visual_dopamine.feedback_training import train
spec=json.loads(Path('direct-training-inputs.json').read_text());output=Path('data/checkpoints/rebot')/('observable-goals-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
result=train(spec['dataset'],output,spec['configurations'],root=Path('data/benchmark-bundle'),iterations=10000,device='cuda',train_motor=True,output_mode='joint_targets',goal_supervision='observable',warm_start=Path('data/checkpoints/rebot/direct-goals-20260910-204558'))
Path('reports/observable-model-path.txt').write_text(str(output)+'\n');print(result,flush=True)
