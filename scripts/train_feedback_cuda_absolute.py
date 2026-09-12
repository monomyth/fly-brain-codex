import json
from pathlib import Path
from datetime import datetime
from fly_brain.visual_dopamine.feedback_training import train
spec=json.loads(Path('reports/cuda-recovery-training-inputs.json').read_text());out=Path('data/checkpoints/rebot')/('cuda-absolute-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
result=train(spec['datasets'],out,spec['configurations'],root=Path('data/benchmark-bundle'),iterations=12000,warm_start=Path('data/checkpoints/rebot/cuda-recovery-20260910-185128'),train_motor=True,device='cuda',output_mode='joint_targets')
Path('reports/cuda-absolute-model-path.txt').write_text(str(out)+'\n');print(result,flush=True)
