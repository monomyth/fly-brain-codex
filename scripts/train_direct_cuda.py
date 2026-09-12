import json
from pathlib import Path
from datetime import datetime
from fly_brain.visual_dopamine.feedback_training import train
spec=json.loads(Path('direct-training-inputs.json').read_text());output=Path('data/checkpoints/rebot')/('direct-goals-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
result=train(spec['dataset'],output,spec['configurations'],root=Path('data/benchmark-bundle'),iterations=12000,device='cuda',train_motor=True,output_mode='joint_targets')
Path('reports/direct-model-path.txt').write_text(str(output)+'\n');print(result,flush=True)
