import argparse,json
from pathlib import Path
from datetime import datetime
from fly_brain.visual_dopamine.feedback_training import train
p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--iterations',type=int,default=5000);a=p.parse_args();spec=json.loads(a.spec.read_text())
out=Path('data/checkpoints/rebot')/('cuda-feedback-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
result=train(spec['datasets'],out,spec['configurations'],root=Path('data/benchmark-bundle'),iterations=a.iterations,warm_start=Path('data/benchmark-bundle/checkpoint'),train_motor=True,device='cuda')
Path('reports/cuda-model-path.txt').write_text(str(out)+'\n');print(result,flush=True)
