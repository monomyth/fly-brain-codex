"""Deterministic, evidence-bound SVG infographics for the September 12 snapshot."""
from pathlib import Path
import json,html,textwrap
from PIL import ImageFont
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/implementation-infographics-20260912'
s=json.loads((OUT/'snapshot.json').read_text())
assert s['baseline']['successes']==19 and s['baseline']['attempts']==20
assert s['latest_native']['successes']==12 and s['latest_native']['attempts']==15
STAMP='12 Sep 2026 · 13:27 PDT'
C={'bg':'#F7F5EF','ink':'#14343C','muted':'#526A70','fixed':'#E1ECF0','learn':'#FFE5B9','plain':'#E9E6DD','line':'#527985','orange':'#AC6826','green':'#197452','red':'#B04738','softred':'#F5E0DA','grey':'#8B9697'}
FONTS='/System/Library/Fonts/Supplemental/Arial'
class Page:
 def __init__(self,num,title,subtitle):
  self.parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="1700" viewBox="0 0 1400 1700"><title>{html.escape(title)}</title><desc>{html.escape(subtitle)}</desc><defs>']
  for name,color in [('a',C['line']),('b',C['orange'])]:self.parts.append(f'<marker id="{name}" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M0,0 L10,5 L0,10 Z" fill="{color}"/></marker>')
  self.parts+=['</defs>',f'<rect width="1400" height="1700" fill="{C["bg"]}"/>']
  self.text(70,61,f'MALECNS / REBOT     {num:02d} OF 03',22,C['muted'],True)
  self.text(70,125,title,47,bold=True)
  self.lines(70,165,[subtitle],25,C['muted'])
 def text(self,x,y,t,size=27,color=None,bold=False,anchor='start'):
  self.parts.append(f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color or C["ink"]}" text-anchor="{anchor}">{html.escape(str(t))}</text>')
 def lines(self,x,y,items,size=26,color=None,bold=False,step=None):
  for i,t in enumerate(items):self.text(x,y+i*(step or size*1.35),t,size,color,bold)
 def rect(self,x,y,w,h,fill,stroke=None,r=14):
  self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}"'+(f' stroke="{stroke}" stroke-width="2"' if stroke else '')+'/>')
 def box(self,x,y,w,h,title,lines=(),kind='fixed',size=26):
  self.rect(x,y,w,h,C[kind])
  title_size=29
  while ImageFont.truetype(FONTS+' Bold.ttf',title_size).getlength(title)>w-42:title_size-=1
  body_size=size
  while lines and max(ImageFont.truetype(FONTS+'.ttf',body_size).getlength(line) for line in lines)>w-44:body_size-=1
  assert title_size>=22 and body_size>=21, (title, title_size, body_size)
  self.text(x+22,y+39,title,title_size,bold=True);self.lines(x+22,y+78,lines,body_size)
 def arrow(self,points,train=False):
  path='M'+' L'.join(f'{x},{y}' for x,y in points)
  self.parts.append(f'<path d="{path}" fill="none" stroke="{C["orange"] if train else C["line"]}" stroke-width="3" stroke-linejoin="round" marker-end="url(#{"b" if train else "a"})"/>')
 def footer(self):
  self.text(70,1633,'Snapshot: '+STAMP+'  ·  Code and completed-run evidence: sources.md',21,C['muted'])
  self.text(70,1666,'An engineering rate model derived from MaleCNS wiring. Biological fidelity is unvalidated.',22,C['muted'])
 def save(self,name):
  self.footer();(OUT/(name+'.svg')).write_text(''.join(self.parts)+'</svg>')

p=Page(1,'How the configured controller moves the arm','UI default: retain-grasp-20260911 · one learned response set · 20 mm cube · folded start')
p.box(70,205,590,170,'Rendered camera views',['Front: Gemini 336L nominal RGB FOV','Gripper: Gemini 305 nominal RGB FOV','Both RGB streams → luminance → R1–R6'],size=24)
p.box(700,205,630,170,'Simulated robot feedback',['Six joint angles, jaw opening, finger contacts','Engineered stimulation of leg sensory cells','Time drives integration; IDs guard resets/frames'],size=24)
p.arrow([(365,375),(365,400)]);p.arrow([(1015,375),(1015,400)])
p.box(70,410,1260,135,'Fixed sensory propagation · MLX / Metal',['Prepared graph: 167,124 nodes; 3,316 assigned retinal inputs. Separate persistent visual/body states.','206 visual-side features + 59 ascending-body features = 265 normalized inputs.'],size=24)
p.arrow([(700,545),(700,571),(225,571),(225,616)],True)
p.text(750,586,'Learned motor response: NumPy + SciPy on CPU',24,C['muted'])
p.box(70,625,310,164,'29 MBONs',['Mushroom-body outputs','Fit input weights','and response offsets'],kind='learn',size=25)
p.box(445,620,260,122,'4,977 relay nodes',['Central-brain pool'],size=23)
p.box(445,827,260,125,'1,129 descending',['neurons / nodes'],size=23)
p.box(800,659,250,125,'1,995 premotor',['neurons / nodes'],size=23)
p.box(1080,827,250,164,'75 motor neurons',['Fit input strengths','from DN + premotor','Existing edges only'],kind='learn',size=23)
p.arrow([(380,679),(436,679)])
p.arrow([(380,747),(408,747),(408,885),(436,885)])
p.arrow([(575,742),(575,817)])
p.arrow([(705,850),(750,850),(750,721),(790,721)])
p.arrow([(705,916),(1070,916)],True)
p.arrow([(1050,721),(1205,721),(1205,817)],True)
p.text(75,875,'Blue: fixed pathways',23,C['muted'])
p.text(75,911,'Amber: fitted inputs',23,C['orange'])
p.arrow([(1205,991),(1205,1024),(314,1024),(314,1060)])
p.box(70,1070,490,150,'Fixed muscle-pool readout',['Engineering mapping from leg-motor pools','75 motor rates → seven control scores'],kind='plain',size=24)
p.arrow([(560,1135),(662,1135)])
p.box(673,1070,657,150,'Commands · nominal 2 Hz',['Six joint targets (degrees) + jaw aperture (mm)','Local Unix socket → native control tools','Native rate / floor limits → RealityKit'],kind='plain',size=24)
p.text(75,1254,'RealityKit: kinematic arm + dynamic cube. The control tools are also exposed through MCP.',24)
p.text(75,1290,'Each step returns new observations; the 3D overlay displays simulated neural activity.',24,C['muted'])
p.rect(70,1323,1260,245,C['plain'])
p.text(94,1362,'Task completion is a separate, explicit control step',29,bold=True)
p.lines(94,1406,['Native evaluator requires: ≥100 mm cube clearance, ≤5° tilt and a stable ≥5 s hold.','Then the runner keeps the reached joint pose, commands the jaw open to 90 mm,','and verifies that the cube has landed. This opening/drop step is scripted.'],26)
p.text(94,1534,'Cube geometry is reserved for teacher supervision and evaluation, outside the neural policy.',24,C['muted'])
p.save('01-runtime')

p=Page(2,'How training and touch selection work','Latest tested implementation: touch-direct-20260912 · experimental · not the configured UI default')
p.box(70,215,490,145,'One shared sensory core',['Front + Gripper images and body feedback','Same fixed sensory mapping and calibration'],size=24)
p.box(870,215,460,145,'Observed finger contacts',['Both contact flags true → holding weights','Otherwise → pickup weights'],kind='plain',size=23)
p.box(70,418,550,145,'Stored pickup weights',['direct-pickup-20260912','Fit using the unheld training subset'],kind='learn',size=25)
p.box(780,418,550,145,'Stored holding weights',['level-hold-response-20260912','Fit using the bilateral-contact training subset'],kind='learn',size=25)
p.arrow([(1100,360),(1100,391),(700,391),(700,590)])
p.arrow([(620,490),(680,490),(680,590)],True)
p.arrow([(780,490),(720,490),(720,590)],True)
p.rect(615,600,170,76,C['plain']);p.text(700,647,'SELECT',27,bold=True,anchor='middle')
p.arrow([(310,360),(310,385),(36,385),(36,788),(68,788)])
p.box(80,723,1250,150,'One shared learned motor route',['The selected bank supplies MBON input weights/offsets and motor-input strengths.','The cells, fixed pathways and seven-channel readout are shared. This is not two brains running in parallel.'],size=24)
p.arrow([(700,676),(700,713)],True)
p.text(80,924,'The contact selector is an engineered rule. Seven motor commands still come from the neural route.',25)
p.rect(70,970,1260,394,C['plain'])
p.text(94,1012,'Offline training: images are used, through cached neural features',29,bold=True)
p.box(94,1046,360,154,'1  Record on the Mac',['Rendered RGB + body feedback','Separate teacher-state records','6,967 records / 122 episodes'],kind='fixed',size=23)
p.box(514,1046,340,154,'2  Create supervision',['Teacher geometry + IK targets','Train / validation / test','Only training rows fit weights'],kind='fixed',size=23)
p.box(914,1046,392,154,'3  Fit response weights',['Neural features + teacher targets','PyTorch: MPS / RTX 4090 CUDA','Fixed topology and output map'],kind='learn',size=23)
p.arrow([(454,1123),(504,1123)]);p.arrow([(854,1123),(904,1123)])
p.arrow([(274,1200),(274,1240),(1110,1240),(1110,1204)])
p.text(515,1229,'cached neural features',22,C['muted'])
p.lines(96,1290,['Counts include training, validation and test records. The teacher supplies geometry-based targets.','The policy receives images and robot feedback; native simulator trials decide deployment.'],24)
p.rect(70,1402,1260,166,C['fixed'])
p.text(94,1444,'Dopamine: distinguish activity display from biological learning',28,bold=True)
p.lines(94,1486,['Offline fits use teacher-loss gradients with a fixed dopamine multiplier.','Run-time rewards update the simulated dopamine display; default evaluation keeps weights fixed.','Touch-selected response banks currently require offline training.'],25)
p.save('02-training-and-touch-selection')

p=Page(3,'What works — and what still fails','Completed simulator evidence · 20 mm cube, folded start · larger-placement reliability remains unresolved')
p.box(70,212,600,187,'Configured UI model: retain-grasp',['19 / 20 passed pickup + hold on its small grid','X: 343–357 mm; Y: −10–8 mm','A separate normal-UI test verified hold + drop.'],kind='fixed',size=26)
p.box(710,212,620,187,'Latest tested candidate: touch-direct',['12 / 15 completed pickup + hold + drop','20 positions planned; stopped after 3 failures','5 remaining positions were not attempted.'],kind='plain',size=26)
p.text(75,450,'Latest candidate: actual floor placements and outcomes',30,bold=True)
p.text(75,488,'Different grids and completion criteria: the two scores above are not a head-to-head comparison.',23,C['muted'])
left,top,scale=130,556,1140/70
# Horizontal is Y (sideways); vertical is X (forward), with equal millimetre scale.
def point(x,y):return left+(y+35)*scale, top+(357-x)*scale
for y in [-30,-15,0,15,30]:
 xx,_=point(357,y);p.parts.append(f'<path d="M{xx},{top} V{top+14*scale}" stroke="#CCD6D5" stroke-width="1"/>');p.text(xx,top+14*scale+37,str(y),23,C['muted'],anchor='middle')
for x in [345,350,355]:
 _,yy=point(x,0);p.parts.append(f'<path d="M{left},{yy} H{left+1140}" stroke="#CCD6D5" stroke-width="1"/>');p.text(left-20,yy+8,str(x),23,C['muted'],anchor='end')
p.text(82,541,'X mm',23,C['muted']);p.text(700,top+14*scale+77,'Cube Y (sideways), mm',24,C['muted'],anchor='middle')
episodes=s['latest_native']['episodes'];failure_index=0
for i,task in enumerate(s['planned_tasks']):
 x,y=task['cube_xy_mm'];cx,cy=point(x,y)
 if i>=len(episodes):p.parts.append(f'<circle cx="{cx}" cy="{cy}" r="11" fill="{C["bg"]}" stroke="{C["grey"]}" stroke-width="3"/>')
 elif episodes[i]['success']:p.parts.append(f'<circle cx="{cx}" cy="{cy}" r="12" fill="{C["green"]}"/>')
 else:
  failure_index+=1;p.parts.append(f'<path d="M{cx-11},{cy-11} L{cx+11},{cy+11} M{cx+11},{cy-11} L{cx-11},{cy+11}" stroke="{C["red"]}" stroke-width="5"/>');p.text(cx+21,cy-14,str(failure_index),22,C['red'],True)
p.text(130,905,'● 12 complete',25,C['green'],True);p.text(490,905,'× 3 failed',25,C['red'],True);p.text(800,905,'○ 5 not attempted',25,C['grey'],True)
for x,title,lines in [(70,'1  No grasp',['Cube [345, −30] mm','Never registered a grasp.','Stopped at the decision limit.']),(500,'2  Hold not qualified',['Cube [350, −30] mm','Grasped; final clearance 117.4 mm.','Final tilt 7.48° exceeds 5°.']),(930,'3  Grasp lost',['Cube [355, +30] mm','Cube dropped and tipped.','Final tilt was about 90°.'])]:
 p.rect(x,960,400,200,C['softred']);p.text(x+20,1000,title,28,C['red'],True);p.lines(x+20,1046,lines,23)
p.rect(70,1205,1260,175,C['fixed'])
p.text(94,1246,'Implemented and demonstrated',29,bold=True)
p.lines(94,1288,['UI Run/Reset, cube placement, both camera views, neural motor commands and the activity overlay.','Release works after a qualifying hold. If the hold never qualifies, the completion drop is not triggered.'],25)
p.rect(70,1415,1260,170,C['plain'])
p.text(94,1456,'Still outside the evidence',29,bold=True)
p.lines(94,1498,['Reliable pickup across arbitrary floor positions and other cube sizes remains unproven.','UI deployment gate: 18/20 qualifying trials; this candidate has not met that requirement.','The 12,000-iteration RTX 4090 fit finished; simulator validation is pending.'],23)
p.save('03-tested-results-and-limits')

sources=['# Evidence for the implementation infographics','',f'Snapshot: {s["captured_at"]}. All three graphics are static, dated snapshots.','',
'The configured default and the saved macOS checkpoint preference both name retain-grasp-20260911. The UI socket was not responding at capture time; no claim is made that a UI session was running.',
'', 'The historical 19/20 batch verifies pickup and hold, not release. Release is supported by the later normal UI test. The latest completed touch-direct batch has 12 complete pickup/hold/drop successes in 15 attempts, with five planned cases unrun. These runs use different grids and should not be compared as a controlled model A/B test.',
'', 'Model counts were read from MotorPolicy and its loaded circuit: 167,124 prepared graph nodes, 3,316 assigned retinal inputs, 206 visual-side plus 59 ascending-body features, 29 MBONs, 4,977 relay nodes, 1,129 descending nodes, 1,995 premotor nodes, and 75 selected motor neurons. They are counts of this implementation, not universal fly anatomy totals.',
'', 'The touch selector switches parameter arrays in one shared learned route. Both banks use existing synapses and the same fixed decoder. It is engineered selection based on two finger-contact flags, not a recovered biological switching mechanism.',
'', 'Default inference uses MLX for sensory propagation and NumPy/SciPy for the learned motor response. Offline fitting uses PyTorch (MPS/CUDA). A constant dopamine multiplier scales the supervised fitting objective; this is not evidence of reconstructed biological reinforcement learning.',
'', 'The CUDA continuation completed 12,000 optimizer iterations. Its results were copied to the Mac and their artifact checksums checked, but it has not been imported/qualified as a replacement controller.', '', '## Source files and checksums','']
for e in s['evidence']:sources += [f'- [{Path(e["path"]).name}]({e["path"]}) — `{e["sha256"]}`']
(OUT/'sources.md').write_text('\n'.join(sources)+'\n')
(OUT/'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MaleCNS implementation — verified snapshot</title><style>body{margin:0;background:#e9e6dd;color:#14343c;font:18px system-ui}main{max-width:1400px;margin:auto;padding:24px}figure{margin:24px 0}img{width:100%;height:auto}a{color:#215a70}nav{display:flex;gap:22px;flex-wrap:wrap}</style><main><h1>MaleCNS implementation and tested limits</h1><p>Snapshot: '''+STAMP+''' · Configuration, code and completed simulator runs.</p><nav><a href="sources.md">Evidence notes</a><a href="snapshot.json">Hashed snapshot</a></nav>'''+''.join(f'<figure><img src="{name}.svg" alt="{alt}"><figcaption><a href="{name}.png">Download PNG</a> · <a href="{name}.svg">Editable SVG</a></figcaption></figure>' for name,alt in [('01-runtime','Configured controller: images, feedback, neural pathways, commands and explicit task completion'),('02-training-and-touch-selection','Experimental touch-selected weight banks and offline teacher-supervised training'),('03-tested-results-and-limits','Completed test outcomes, failures, and limits of the current implementation')])+'</main></html>')
print(OUT)
