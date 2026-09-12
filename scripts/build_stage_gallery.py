"""Inspect actual recorded views before and after the neural stage handoff."""
import argparse,html,json,os
from pathlib import Path
from urllib.parse import quote
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
report=json.loads((a.directory/'stages.json').read_text())
blocks=[]
for index,result in enumerate(report['results']):
    case=a.directory/f'case-{index:02d}'
    folders=sorted((case/'setup').glob('*'))[-1:]+sorted((case/'neural').glob('*'))
    frames=[]
    for folder in folders:
        path=folder/'observation.json'
        if not path.exists():continue
        o=json.loads(path.read_text())
        decision=json.loads((folder/'decision.json').read_text()) if (folder/'decision.json').exists() else None
        urls={im['name']:quote(os.path.relpath(folder/im['file'],a.directory),safe='/') for im in o['images']}
        frames.append({'views':urls,'caption':f'{folder.parent.name} {folder.name}; aperture {o["gripper_mm"]:.1f} mm; contacts {o["finger_contacts"]}',
                       'action':decision['action'] if decision else 'Conventional setup, before neural handoff'})
    title=f'Case {index}: {result["stage"]}; '+('stage passed' if result['stage_success'] else 'stage failed')
    if result.get('error'):title+='; '+result['error']
    data=html.escape(json.dumps(frames),quote=True)
    blocks.append(f'<section data-frames="{data}"><h2>{html.escape(title)}</h2><input type="range" min="0" max="{max(0,len(frames)-1)}" value="0"><p class="caption"></p><div class="views"><figure><img class="front"><figcaption>Front</figcaption></figure><figure><img class="gripper"><figcaption>Gripper</figcaption></figure></div><pre></pre></section>')
page='''<!doctype html><meta charset="utf-8"><title>Isolated neural-control stages</title><style>body{max-width:1000px;margin:30px auto;padding:0 20px;background:#17201b;color:#e0e8e2;font:16px system-ui}h1{font-size:28px}h2{font-size:19px}section{padding:18px;background:#253129;margin:20px 0;border-radius:8px}input{width:100%}.views{display:flex;gap:15px}figure{margin:0;flex:1}img{width:100%;aspect-ratio:1.6;object-fit:contain;background:#121712}pre{white-space:pre-wrap;font-size:13px}.caption{font-size:13px}p{line-height:1.5}</style><h1>Isolated neural-control stages</h1><p>The first frame in each slider is the final conventional setup. Following frames are observations used by the neural controller. A stage pass is not a full autonomous pickup: setup began folded but was performed by the conventional controller.</p>'''+''.join(blocks)+'''<script>for(const section of document.querySelectorAll('section')){const frames=JSON.parse(section.dataset.frames),slider=section.querySelector('input');const draw=()=>{const f=frames[Number(slider.value)];if(!f)return;section.querySelector('.caption').textContent=f.caption;section.querySelector('.front').src=f.views.Front;section.querySelector('.gripper').src=f.views.Gripper;section.querySelector('pre').textContent=JSON.stringify(f.action,null,2)};slider.addEventListener('input',draw);draw()}</script>'''
(a.directory/'gallery.html').write_text(page);print(a.directory/'gallery.html')
