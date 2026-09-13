"""Register the supplied cradle assembly meshes in the simulator end_link frame.

Input meshes are extracted read-only from the user's FCStd preview with FreeCAD.
Generated master assets and provenance remain inside this project.
"""
import hashlib,json,struct
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/camera-rigs/cradle-v3'
NATIVE=Path.home()/'github/rebot-motion-lab-codex'
SOURCE=Path.home()/'github/rebot-b601-trinkets/output/cradle_support'
DTYPE=np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attribute','<u2')])
R=np.array([[0,0,1],[1,0,0],[0,1,0.]])
# CAD gripper fingertip is Z=103.2093277 mm; the URDF end_link is at that fingertip.
SHIFT=np.array([-103.2093276977539,0,0.])

def read(path):
    raw=path.read_bytes();count=struct.unpack_from('<I',raw,80)[0]
    if len(raw)!=84+count*50:raise ValueError('Expected binary STL')
    return np.frombuffer(raw,DTYPE,count=count,offset=84).copy()

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

report={'source_assembly':str(SOURCE/'B601_Gemini305_CradleSupportPreview.FCStd'),
        'source_image':str(SOURCE/'assembly_side.png'), 'reference_tilt_deg':15.,
        'cad_axes_to_tool':{'cad_Z':'tool_X','cad_X':'tool_Y','cad_Y':'tool_Z'},
        'cad_to_tool_rotation':R.tolist(),'translation_mm':SHIFT.tolist(),'source_files':{},'exports':{}}
for file in ['B601_Gemini305_CradleSupportPreview.FCStd','assembly_side.png','README.md']:
    report['source_files'][file]=digest(SOURCE/file)
for name,target in [('PrintableMount','gemini305-cradle.stl'),('OfficialCamera','gemini305-housing.stl')]:
    path=DATA/(name+'-cad.stl');mesh=read(path)
    mesh['vertices']=(mesh['vertices']@R.T+SHIFT)/1000
    mesh['normal']=mesh['normal']@R.T
    output=DATA/target
    output.write_bytes(b'Gemini 305 cradle reference; end_link; meters'.ljust(80,b' ')+struct.pack('<I',len(mesh))+mesh.tobytes())
    points=mesh['vertices'].reshape(-1,3)
    report['exports'][target]={'sha256':digest(output),'facets':len(mesh),'minimum_m':points.min(0).tolist(),'maximum_m':points.max(0).tolist()}
# Independent overlap against the existing gripper-base meshes establishes registration.
model=NATIVE/'Sources/RobotCore/Resources/model'
definition=json.loads((model/'model.json').read_text())
target=np.unique(np.concatenate([read(model/v['mesh'])['vertices'].reshape(-1,3)*1000
                 for link in definition['links'] if link['name']=='end_link' for v in link['visuals']]),axis=0)
reference=np.unique(read(DATA/'Gripper-cad.stl')['vertices'].reshape(-1,3),axis=0)@R.T+SHIFT
distance,_=cKDTree(reference).query(target)
assert np.median(distance)<.1 and np.mean(distance<.1)>.7
report['registration']={'samples':len(target),'nearest_vertex_median_mm':float(np.median(distance)),
                       'fraction_within_0_1_mm':float(np.mean(distance<.1)),
                       'method':'axes and fingertip datum, checked against existing gripper-base vertices; tessellation/source parts differ'}
rear=np.array([0.,66,15])@R.T+SHIFT
ray=np.array([np.cos(np.pi/12),0.,-np.sin(np.pi/12)])
report['camera_rear_midpoint_tool_mm']=rear.tolist()
report['nominal_front_midpoint_tool_mm']=(rear+23*ray).tolist()
report['optical_assumption']='Ideal RGB pinhole at nominal 23 mm front plane; measured factory intrinsics/entrance pupil unavailable'
(DATA/'provenance.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:report[k] for k in ['registration','camera_rear_midpoint_tool_mm','nominal_front_midpoint_tool_mm','exports']},indent=2))
