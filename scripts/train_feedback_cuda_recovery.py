import json
from pathlib import Path
from datetime import datetime
from fly_brain.visual_dopamine.feedback_training import train
spec=json.loads(Path('reports/cuda-recovery-training-inputs.json').read_text());out=Path('data/checkpoints/rebot')/('cuda-recovery-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
result=train(spec['datasets'],out,spec['configurations'],root=Path('data/benchmark-bundle'),iterations=8000,warm_start=Path('data/checkpoints/rebot/cuda-feedback-20260910-183546'),train_motor=True,device='cuda')
Path('reports/cuda-recovery-model-path.txt').write_text(str(out)+'\n');print(result,flush=True)
