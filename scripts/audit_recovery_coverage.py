"""Audit held-cube and gripper-relative tilt coverage in completed recordings."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from fly_brain.assets import write_json,sha256_file
from fly_brain.teacher import grasp_pose


def audit(plan_path):
    plan=json.loads(plan_path.read_text());dataset=Path(plan['dataset']);cases={c['id']:c for c in plan['collection_cases']}
    collection=json.loads((dataset/'collection-report.json').read_text())
    episodes=[]
    for item in collection['episodes']:
        folder=Path(item['episode_directory']);manifest=json.loads((folder/'manifest.json').read_text())
        rows=[json.loads(line) for line in (folder/'steps.jsonl').read_text().splitlines()]
        if sha256_file(folder/'live-features.npz')!=manifest['live_features_sha256']:raise ValueError('Feature checksum mismatch')
        with np.load(folder/'live-features.npz',allow_pickle=False) as saved:
            if len(saved['features'])!=len(rows) or not np.isfinite(saved['features']).all() or saved['frame_ids'].tolist()!=[r['observation']['frame_id'] for r in rows]:
                raise ValueError('Feature/observation alignment failed')
        task=manifest['task'];ideal=Rotation.from_quat(grasp_pose(task['cube_xy_mm'],20,task.get('cube_yaw_deg',0))['quaternion_xyzw']).inv().apply([0,0,1])
        held=[]
        for index,row in enumerate(rows):
            if not row['evaluation']['held']:continue
            observation=row['observation']
            actual=(Rotation.from_quat(observation['grasp_pose']['quaternion_xyzw']).inv()*Rotation.from_quat(observation['cube_pose']['quaternion_xyzw'])).apply([0,0,1])
            held.append({'step':index,'neural':row['intervention']['policy_selected'],
                         'world_tilt_deg':row['evaluation']['tilt_deg'],
                         'relative_tilt_deg':float(np.degrees(np.arccos(np.clip(np.dot(actual,ideal),-1,1))))})
        neural=[r for r in held if r['neural']]
        episodes.append({'case':item['case_id'],'split':cases[item['case_id']]['split'],'success':item['success'],'error':item.get('error'),
                         'directory':str(folder),'frames':len(rows),'neural_held_frames':len(neural),
                         'neural_held_over_5_deg':sum(r['world_tilt_deg']>5 for r in neural),'neural_held_over_10_deg':sum(r['world_tilt_deg']>10 for r in neural),
                         'maximum_neural_tilt_deg':max((r['world_tilt_deg'] for r in neural),default=0),
                         'maximum_relative_tilt_deg':max((r['relative_tilt_deg'] for r in neural),default=0)})
    summary={split:{key:sum(e[key] for e in episodes if e['split']==split and e['success']) for key in ['frames','neural_held_frames','neural_held_over_5_deg','neural_held_over_10_deg']} for split in ['train','validation','test']}
    result={'episodes':episodes,'summary_verified_only':summary,'feature_alignment_passed':True,
            'note':'Angles and contact state are evaluator-only audit measurements; they are not actor inputs. Frame counts are correlated trajectory observations.'}
    write_json(plan_path.parent/'coverage.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);args=p.parse_args()
    result=audit(args.plan);print(json.dumps({'completed':len(result['episodes']),'successes':sum(e['success'] for e in result['episodes']),'summary':result['summary_verified_only']},indent=2))
