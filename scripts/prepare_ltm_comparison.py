"""Build and physically calibrate the controlled LTM gripper variant."""
from pathlib import Path
from fly_brain.assets import write_json
from fly_brain.visual_dopamine.ltm_circuit import prepare
from fly_brain.visual_dopamine.motor_experiment import calibrate
root=Path(__file__).resolve().parents[1];run=root/'reports/temporal-control-20260911'
path,meta=prepare(root/'data');write_json(run/'ltm-circuit.json',{'path':str(path),'metadata':meta})
print({'ltm_feedback_circuit':str(path),'motor_circuit_id':meta['motor_circuit_id']},flush=True)
result=calibrate(run/'ltm-calibration',root/'data',root/'data/motor-circuits'/meta['motor_circuit_id'])
write_json(run/'ltm-calibration-summary.json',result);print(result,flush=True)
