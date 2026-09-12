"""Install an isolated, pinned Metal test environment without changing .venv."""
import importlib.metadata,json,os,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1];run=root/'reports/temporal-control-20260911'
versions={name:importlib.metadata.version(name) for name in ['numpy','scipy','pyarrow','pillow']}
requirements=['mlx==0.32.2']+[f'{name}=={version}' for name,version in versions.items()]
(run/'mlx-requirements.txt').write_text('\n'.join(requirements)+'\n')
venv=root/'.venv-mlx';environment={**os.environ,'UV_CACHE_DIR':str(root/'.cache/uv-mlx')}
if not venv.exists():subprocess.run(['uv','venv','--python',str(root/'.venv/bin/python'),str(venv)],env=environment,check=True)
subprocess.run(['uv','pip','install','--python',str(venv/'bin/python'),'-r',str(run/'mlx-requirements.txt')],env=environment,check=True)
subprocess.run(['uv','pip','install','--python',str(venv/'bin/python'),'--no-deps','-e',str(root)],env=environment,check=True)
(run/'mlx-environment.json').write_text(json.dumps({'python':str(venv/'bin/python'),'requirements':requirements,'existing_environment_changed':False,'sources':['https://ml-explore.github.io/mlx/build/html/install.html','https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.metal_kernel.html']},indent=2)+'\n')
