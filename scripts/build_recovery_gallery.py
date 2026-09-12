"""Show actual neural-prefix observations and recorded corrective teacher targets."""
import argparse
import html
import json
import os
from pathlib import Path
from urllib.parse import quote

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--plan',type=Path,required=True)
args=parser.parse_args();run=args.plan.resolve().parent;plan=json.loads(args.plan.read_text())
cases={c['id']:c for c in plan['collection_cases']}
cards=[]
for path in sorted(Path(plan['dataset']).glob('episode-*/manifest.json')):
    meta=json.loads(path.read_text())
    if not meta['complete']:continue
    case=cases[meta['case_id']]
    rows=[json.loads(s) for s in (path.parent/'steps.jsonl').read_text().splitlines()]
    selected={0,len(rows)-1}
    for i,row in enumerate(rows):
        if i and (row['stage']!=rows[i-1]['stage'] or row['intervention']['policy_selected']!=rows[i-1]['intervention']['policy_selected']):
            selected.update([i-1,i])
    frames=[]
    for i in sorted(selected):
        if i<0 or i>=len(rows):continue
        row=rows[i];images=[]
        for im in row['observation']['images']:
            source=path.parent/im['file']
            if not source.is_file():raise ValueError('Missing recorded image: '+str(source))
            url=quote(os.path.relpath(source,run),safe='/')
            images.append(f'<figure><img loading="lazy" src="{url}" alt="Recorded {html.escape(im["name"])} observation"><figcaption>{html.escape(im["name"])}</figcaption></figure>')
        applied=row['intervention'].get('action_applied',not bool(row.get('evaluation',{}).get('success')))
        actor=('Angled grasp setup' if row['intervention'].get('setup_perturbation') else 'Neural policy' if row['intervention']['policy_selected'] else 'Teacher correction') if applied else 'Completed observation; no command sent'
        text=json.dumps({'executed_by':actor,'executed_action':row['action'] if applied else None,'supervised_target':row['expert_target'],
                         'observed_aperture_mm':row['observation']['gripper_mm'],'observed_contacts':row['observation']['finger_contacts']},indent=2)
        frames.append(f'<article><h3>Frame {i} · {html.escape(row["stage"])} · {actor}</h3><div class="views">{"".join(images)}</div><pre>{html.escape(text)}</pre></article>')
    status='Verified recovery' if meta['success'] else 'Failed; excluded from training'
    label=f"Case {case['id']} · {case['split']} · cube {meta['task']['cube_xy_mm']} · {status}"
    cards.append(f'<details><summary>{html.escape(label)}</summary><p>{html.escape(meta["result"].get("error", "Native lift and five-second hold completed"))}</p>{"".join(frames)}</details>')
if cards:cards[0]=cards[0].replace('<details>','<details open>',1)
page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Recorded recovery training data</title><style>
:root{color-scheme:dark}body{max-width:1100px;margin:28px auto;padding:0 20px;background:#171d1a;color:#e4ede7;font:16px system-ui;line-height:1.5}summary{cursor:pointer;background:#2b3630;padding:14px;border-radius:6px;margin:12px 0}article{border-bottom:1px solid #536158;padding:12px}.views{display:flex;gap:16px}figure{flex:1;margin:0}img{width:100%;aspect-ratio:1.6;object-fit:contain;background:#0d100e}h3{font-size:17px}pre{font-size:13px;white-space:pre-wrap;color:#c8d8ce}p{color:#b9ccc0}</style><h1>Recorded recovery training data</h1><p>Each attempt starts folded with a 20 mm cube. The plan can use a neural prefix followed by teacher corrections, or teacher commands throughout. Executed commands are identified below; the recorded neural proposals are separate. These are assisted data-collection runs, not autonomous qualification results. Only verified native lift/hold episodes enter the train, validation or test split.</p><p>Images are the actual observations before each command. The executed command and teacher target are shown separately. Commands below are the recorded teacher targets. The selected fitting recipe may derive different observable goals; saved training-target arrays document the targets actually optimized. Cube geometry is available to the teacher only.</p>'''+''.join(cards)+'</html>'
(run/'recovery-data.html').write_text(page);print(run/'recovery-data.html')
