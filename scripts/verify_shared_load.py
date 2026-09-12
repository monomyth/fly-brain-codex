"""Load a real checkpoint from a separate project without downloading data."""
import argparse
import json
from pathlib import Path
from fly_brain import assets
from fly_brain.policy import Policy

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True, type=Path)
parser.add_argument('--observation', required=True, type=Path)
parser.add_argument('--home', type=Path)
parser.add_argument('--output', type=Path)
args = parser.parse_args()

def unexpected_download(*args, **kwargs):
    raise AssertionError('Checkpoint inference must reuse prepared assets')

assets.ensure_assets = unexpected_download
text = args.observation.read_text()
record = json.loads(text.splitlines()[0] if args.observation.suffix == '.jsonl' else text)
observation = record.get('observation', record)
policy = Policy(args.checkpoint, args.home)
policy.reset(observation)
action = policy.act(observation)
report = {'cwd': str(Path.cwd()), 'checkpoint': str(args.checkpoint),
          'shared_root': str(assets.home(args.home)), 'graph_id': policy.metadata['graph_id'],
          'reused_prepared_graph': True, 'action': action, 'task_success_claimed': False}
if args.output:
    assets.write_json(args.output, report)
print(json.dumps(report, indent=2))
