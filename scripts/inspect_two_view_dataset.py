"""Create a local gallery of the actual synchronized training images and labels."""
import argparse,html,json,os
from pathlib import Path
from collections import Counter
from fly_brain.assets import write_json

project=Path(__file__).resolve().parents[1];settings=json.loads((project/'reports/two-view-current.json').read_text());run=Path(settings['run']);dataset=Path(settings['dataset'])
plan=json.loads((run/'plan.json').read_text());groups={tuple(x):name for name,points in plan['configurations'].items() for x in points}
parser=argparse.ArgumentParser();parser.add_argument('--training-plan',type=Path);args=parser.parse_args()
expanded=args.training_plan or run/'supported-training-plan.json'
training_plan=json.loads(expanded.read_text()) if expanded.exists() else {}
datasets=[Path(x) for x in training_plan.get('datasets',[dataset])]
cards=[];stats=Counter();latencies=[]
for manifest in sorted(p for d in datasets for p in d.glob('episode-*/manifest.json')):
    meta=json.loads(manifest.read_text())
    if not meta['complete']:continue
    rows=[json.loads(x) for x in (manifest.parent/'steps.jsonl').read_text().splitlines()]
    split=groups[tuple(meta['task']['cube_xy_mm'])];stats[split+'_episodes']+=1;stats['raw_frames']+=len(rows)
    stats['successful_episodes']+=bool(meta['success']);latencies.append(meta['result'].get('p95_loop_ms'))
    selected=[]
    for stage in ['approach','descend','close','lift','hold','release']:
        options=[(i,r) for i,r in enumerate(rows) if r['stage']==stage]
        if options:selected.append(options[0] if stage=='approach' else options[len(options)//2])
    frames=[]
    for i,row in selected:
        images={c['name']:c for c in row['observation']['images']}
        views=[]
        for name in ['Front','Gripper']:
            file=manifest.parent/images[name]['file'];url=html.escape(os.path.relpath(file,run),quote=True)
            views.append(f'<figure><img loading="lazy" src="{url}"><figcaption>{name}</figcaption></figure>')
        target=row['expert_target'];q=', '.join(f'{x:.1f}' for x in target['joints_deg'])
        frames.append(f'<div class="sample"><h3>{html.escape(row["stage"])} · frame {i}</h3><div class="pair">{"".join(views)}</div><p>Expert joints (degrees): {q}; gripper command: {target["gripper_mm"]:.0f} mm</p></div>')
    xy=meta['task']['cube_xy_mm'];label=f'{split} · cube {xy[0]:g}, {xy[1]:g} mm · {meta["teacher_version"]} · '+('success' if meta['success'] else 'failed; excluded from fitting')
    cards.append(f'<details data-split="{split}"><summary>{html.escape(label)}</summary>{"".join(frames)}<p>{html.escape(manifest.parent.name)}</p></details>')
summary={'camera_rig_revision':plan['camera_rig_revision'],'counts':dict(stats),'episode_p95_loop_ms':latencies,'raw_images':2*stats['raw_frames'],'teacher_source':'conventional direct goals plus explicitly recorded recovery curriculum','model_input':'Front/Gripper RGB luminance, joints, gripper opening, finger contacts','cube_ground_truth':'teacher labels and evaluation only'}
write_json(run/'image-dataset-summary.json',summary)
document='''<!doctype html><meta charset="utf-8"><title>Two-view training observations</title><style>
body{background:#161c19;color:#e5ebe7;font:16px system-ui;margin:30px auto;max-width:1080px;padding:0 20px}a{color:#c0df66}h1{font-size:28px}.note{color:#b9c5bf;line-height:1.6}summary{padding:15px;background:#26302a;cursor:pointer;border-radius:5px;margin:9px 0}.pair{display:flex;gap:16px}figure{margin:0;flex:1}img{width:100%;aspect-ratio:1.6;object-fit:contain;background:#101511}figcaption{padding:5px}.sample{padding:10px;border-bottom:1px solid #39483d}.sample p{font:13px ui-monospace;color:#b9c5bf}select{padding:8px;margin:15px 0;background:#26302a;color:inherit;border:1px solid #718173}
</style><h1>Front + Gripper training observations</h1><p class="note">These are recorded teacher demonstrations and explicitly teacher-assisted corrections. They are not autonomous model results. Failed attempts are retained for inspection and excluded from fitting. Labels shown are the recorded teacher commands; the selected training plan may standardize lift targets as described below. Every trial starts folded with a 20 mm cube. The learned controller receives camera pixels, joint positions, jaw opening and finger contacts. True cube coordinates are used only by the teacher and evaluator.</p>'''
if training_plan.get('approach_center_mm') is not None:
    document+=f'<p>Planned raised approach center: XY {html.escape(str(training_plan["approach_center_mm"]))} mm. Grasp targets remain specific to each observed cube.</p>'
if training_plan.get('hold_center_mm') is not None:
    document+=f'<p>Planned tool holding center: XY {html.escape(str(training_plan["hold_center_mm"]))} mm. This is a fixed training target to teach correction of horizontal drift.</p>'
if training_plan.get('grasp_goal_height_mm') is not None:
    document+=f'<p>Planned training grasp target: world Z={training_plan["grasp_goal_height_mm"]:g} mm. The joint commands below are the original recorded teacher commands; the training target is adjusted separately.</p>'
if training_plan.get('desired_lift_clearance_mm') is not None:
    document+=f'<p>Planned training lift target: {training_plan["desired_lift_clearance_mm"]:g} mm clearance. This is supervision, not a measured autonomous result.</p>'
document+=f'<p>{stats["successful_episodes"]} successful demonstrations · {stats["raw_frames"]} synchronized pairs · rig {html.escape(plan["camera_rig_revision"])}</p>'
document+='''<label>Show <select id="split"><option value="all">All positions</option><option>train</option><option>validation</option><option>test</option></select></label>'''+''.join(cards)+'''<script>document.querySelector('#split').onchange=e=>{document.querySelectorAll('details').forEach(x=>x.hidden=e.target.value!=='all'&&x.dataset.split!==e.target.value)};</script>'''
(run/'training-images.html').write_text(document)
print(json.dumps({'gallery':str(run/'training-images.html'),**summary},indent=2))
