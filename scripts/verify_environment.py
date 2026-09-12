import json, time
from pathlib import Path
from fly_brain.simulator import EpisodeSimulator, configure_episode
from fly_brain.assets import write_json
from fly_brain.collect import collect

report={'reset_checks':[]}
poses=[]
for path in sorted(Path('/Users/monomyth/code/codex/fly-brain/reports/gru-evaluation').glob('episode-*/steps.jsonl')):
    lines=path.read_text().splitlines()
    if lines:
        observation=json.loads(lines[-1])['observation']
        poses.append({'joints_deg':observation['joints_deg'],'gripper_mm':observation['gripper_mm']})
with EpisodeSimulator() as client:
    task={'input_mode':'state','cube_xy_mm':[350,0],'cube_size_mm':50}
    configure_episode(client,task)
    for index,pose in enumerate(poses[:3]):
        client.call('rebot_experiment_control',action='start');client.connect(model_id='reset-regression')
        end=time.monotonic()+4
        while time.monotonic()<end:
            observation=client.observe()
            try: client.action(observation,pose)
            except RuntimeError: break
            time.sleep(.1)
        configure_episode(client,task)
        observation=client.observe(images=True)
        position=observation['cube_pose']['position_mm']
        assert abs(position[0]-350)<2 and abs(position[1])<2 and abs(position[2]-24)<2,position
        report['reset_checks'].append({'trial':index+1,'phase':observation['phase'],'cube_position_mm':position})
    tasks=[{'input_mode':'state','cube_xy_mm':[350,0],'cube_size_mm':size,'hold_seconds':5,'timeout_seconds':60} for size in [10,90]]
    report['size_checks']=collect(client,tasks,'/Users/monomyth/code/data/malecns/datasets/rebot-pick/size-boundary-final')
write_json('/Users/monomyth/code/codex/fly-brain/reports/final-native-check.json',report)
print(json.dumps(report,indent=2))
