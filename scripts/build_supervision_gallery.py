"""Display the actual saved optimization targets for verified recovery observations."""
import argparse
import json
import os
from pathlib import Path
from urllib.parse import quote
import numpy as np
from scipy.spatial.transform import Rotation

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
m=json.loads((a.checkpoint/'manifest.json').read_text());rows=json.loads((a.checkpoint/'feature-rows.json').read_text())
safe_targets=m.get('supervision_contract','').startswith('safe-observable-grasp-')
wanted={r['episode'] for r in rows} if safe_targets else {e['episode'] for e in m['episodes'] if e.get('supervision_contract')=='recorded-recovery-v1'}
folders={}
for dataset in m['datasets']:
    for folder in Path(dataset).glob('episode-*'):
        if folder.name in folders:raise ValueError('Duplicate episode identifier')
        folders[folder.name]=folder
with np.load(a.checkpoint/'training-targets.npz',allow_pickle=False) as saved:targets={k:saved[k].copy() for k in saved.files}
assert len(rows)==len(targets['motor_targets'])
orientation=None
if (a.checkpoint/'orientation-supervision.npz').exists():
    with np.load(a.checkpoint/'orientation-supervision.npz',allow_pickle=False) as saved:orientation={k:saved[k].copy() for k in saved.files}
    if len(orientation['held_rows'])!=len(rows):raise ValueError('Orientation supervision does not match feature rows')
retention=None
if (a.checkpoint/'retention-supervision.npz').exists():
    with np.load(a.checkpoint/'retention-supervision.npz',allow_pickle=False) as saved:retention={k:saved[k].copy() for k in saved.files}
cache={};episodes={}
for index,item in enumerate(rows):
    name=item['episode']
    if name not in wanted:continue
    folder=folders[name]
    if name not in cache:cache[name]=[json.loads(line) for line in (folder/'steps.jsonl').read_text().splitlines()]
    row=cache[name][item['step']];obs=row['observation']
    applied=(row.get('intervention') or {}).get('action_applied',not bool(row.get('evaluation',{}).get('success')))
    held_up=orientation is not None and bool(orientation['held_rows'][index])
    held_clearance=held_up and m.get('training_loss_recipe',{}).get('held_position_objective')=='clearance'
    anchored=retention is not None and bool(retention['anchor_rows'][index])
    record={'views':{im['name']:quote(os.path.relpath(folder/im['file'],a.output.parent),safe='/') for im in obs['images']},
            'goal_role':m['goal_role_names'][int(targets['goal_role_code'][index])] if 'goal_role_code' in targets else None,
            'frame':item['step'],'stage':item['stage'],'time':obs['simulation_time'],'split':item['split'],
            'goal_position_mm':None if held_clearance else targets['goal_position_mm'][index].tolist(),
            'minimum_cube_clearance_mm':m.get('training_loss_recipe',{}).get('held_clearance_target_mm') if held_clearance else None,
            'baseline_joint_targets_deg':(retention['reference_scores'][index,:6]*m['architecture']['joint_gain']).tolist() if anchored else None,
            'baseline_gripper_target_mm':float(90/(1+np.exp(-np.clip(retention['reference_scores'][index,6]*2000,-60,60)))) if anchored else None,
            'baseline_retention_strength':m.get('training_loss_recipe',{}).get('pregrasp_retention_strength') if anchored else None,
            'goal_quaternion_xyzw':None if held_up else Rotation.from_matrix(targets['goal_rotation'][index]).as_quat().tolist(),
            'orientation_mode':'held cube upright' if held_up else 'tool rotation',
            'cube_up_target_world':[0,0,1] if held_up else None,
            'cube_up_in_gripper_frame':orientation['relative_cube_up'][index].tolist() if held_up else None,
            'posture_weight':m.get('training_loss_recipe',{}).get('held_joint_posture_weight',.01) if held_up else .04,
            'joint_posture_reference_deg':orientation['current_joints_deg'][index].tolist() if held_up else targets['motor_targets'][index,:6].tolist(),
            'gripper_label':'OPEN' if targets['motor_targets'][index,6]>.5 else 'CLOSE',
            'sample_weight':float(targets['sample_weights'][index]) if 'sample_weights' in targets else None,
            'recorded_target_preserved':not safe_targets and bool(targets['preserve_expert_target'][index]),
            'actual_executed_action':row['action'] if applied else None,'recorded_teacher_target':row['expert_target'],
            'executed_by':('angled grasp setup' if (row.get('intervention') or {}).get('setup_perturbation') else 'neural policy' if (row.get('intervention') or {}).get('policy_selected') else 'teacher correction') if applied else 'no command (episode complete)'}
    episodes.setdefault(name,[]).append(record)
data=json.dumps([{'episode':key,'frames':value} for key,value in episodes.items()]).replace('<','\\u003c')
page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Actual recovery training targets</title><style>
:root{color-scheme:dark}body{max-width:1100px;margin:28px auto;padding:0 20px;background:#171d1a;color:#e4ede7;font:16px system-ui;line-height:1.5}select{width:100%;padding:12px;background:#2b3630;color:inherit;border:1px solid #64766a;border-radius:6px}input{width:100%;accent-color:#c1ddcb}.views,.targets{display:grid;grid-template-columns:1fr 1fr;gap:16px}figure{margin:0}img{width:100%;aspect-ratio:1.6;object-fit:contain;background:#0e120f}pre{font:13px ui-monospace;white-space:pre-wrap;background:#253029;padding:14px;border-radius:6px}p{color:#bbcec2}h2{font-size:18px}@media(max-width:700px){.views,.targets{grid-template-columns:1fr}}</style>
<h1>Actual recovery training targets</h1><p>These targets were saved by the optimizer for this checkpoint. They may differ from the original teacher commands: ordinary targets use the observable-goal recipe, while selected escape corrections remain unchanged. This page contains verified recovery examples only; the fit also includes the earlier demonstrations.</p>
<p>XYZ and orientation below are supervision, not inputs to the autonomous actor. Joint values are the posture reference used by this recipe. When the orientation mode is held cube upright, the target is the world vertical axis; heading is free, and the posture reference is the recorded current pose. A minimum-clearance target replaces an exact held position when shown. Baseline retention fields identify the additional training-only constraint on approach behavior. The gripper label is a separate open/close target.</p>
<select id="episode" aria-label="Training episode"></select><input id="frame" type="range" min="0" value="0" aria-label="Recorded frame"><p id="caption"></p>
<div class="views"><figure><figcaption>Front</figcaption><img id="front" alt="Recorded front camera"></figure><figure><figcaption>Gripper</figcaption><img id="gripper" alt="Recorded gripper camera"></figure></div>
<div class="targets"><section><h2>Targets actually optimized</h2><pre id="target"></pre></section><section><h2>Recorded execution and original teacher command</h2><pre id="recorded"></pre></section></div>
<script>const episodes=__DATA__;const $=id=>document.getElementById(id);for(let i=0;i<episodes.length;i++){const e=episodes[i],o=document.createElement('option');o.value=i;o.textContent=`${e.frames[0].split} · ${e.episode} · ${e.frames.length} observations`;$('episode').appendChild(o)}
function draw(){const e=episodes[Number($('episode').value)],f=e.frames[Number($('frame').value)];$('front').src=f.views.Front;$('gripper').src=f.views.Gripper;$('caption').textContent=`Frame ${f.frame} · ${f.stage} · ${f.time.toFixed(2)} simulation seconds · executed by ${f.executed_by}`;const round=a=>a.map(x=>Number(x.toFixed(3)));$('target').textContent=JSON.stringify({goal_role:f.goal_role,goal_position_mm:f.goal_position_mm?round(f.goal_position_mm):null,minimum_cube_clearance_mm:f.minimum_cube_clearance_mm,baseline_joint_targets_deg:f.baseline_joint_targets_deg?round(f.baseline_joint_targets_deg):null,baseline_gripper_target_mm:f.baseline_gripper_target_mm,baseline_retention_strength:f.baseline_retention_strength,goal_quaternion_xyzw:f.goal_quaternion_xyzw?round(f.goal_quaternion_xyzw):null,orientation_mode:f.orientation_mode,cube_up_target_world:f.cube_up_target_world,cube_up_in_gripper_frame:f.cube_up_in_gripper_frame,posture_weight:f.posture_weight,joint_posture_reference_deg:round(f.joint_posture_reference_deg),gripper_label:f.gripper_label,sample_weight:f.sample_weight,recorded_target_preserved:f.recorded_target_preserved},null,2);$('recorded').textContent=JSON.stringify({actual_executed_action:f.actual_executed_action,original_teacher_target:f.recorded_teacher_target},null,2)}
function select(){$('frame').value=0;$('frame').max=episodes[Number($('episode').value)].frames.length-1;draw()}$('episode').addEventListener('change',select);$('frame').addEventListener('input',draw);if(episodes.length)select();</script></html>'''.replace('__DATA__',data)
if safe_targets:page=page.replace('Actual recovery training targets','Corrected training targets').replace('These targets were saved by the optimizer for this checkpoint. They may differ from the original teacher commands: ordinary targets use the observable-goal recipe, while selected escape corrections remain unchanged. This page contains verified recovery examples only; the fit also includes the earlier demonstrations.','These are the actual corrected optimization targets for every recorded training, validation and test frame. The original commands are retained alongside them for comparison. The new recipe raises before large lateral moves, lowers the approach target as alignment improves, and closes only near the grasp. All target poses passed inverse-kinematics and native floor checks. These are supervision targets; autonomous success is measured separately.')
a.output.write_text(page);print(json.dumps({'path':str(a.output),'episodes':len(episodes),'observations':sum(map(len,episodes.values()))}))
