import json
from pathlib import Path
from datetime import datetime
from fly_brain.visual_dopamine.feedback_training import train
spec=json.loads(Path('grounding-training-inputs.json').read_text());profile=json.loads(Path('retina-workspace.json').read_text())
output=Path('data/checkpoints/rebot')/('retinal-grounding-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
result=train(spec['datasets'],output,spec['configurations'],root=Path('data/benchmark-bundle'),iterations=12000,device='cuda',train_motor=True,output_mode='joint_targets',goal_supervision='observable',precondition=True,initial_weight=8.,first_frame_weight=32.,refit_adaptation=True,retinal_profile=profile,warm_start=Path('data/checkpoints/rebot/preconditioned-goals-20260910-211738'))
Path('reports/grounded-model-path.txt').write_text(str(output)+'\n');print(result,flush=True)
