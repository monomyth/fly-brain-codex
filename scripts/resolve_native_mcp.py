"""Call Blackmagic's bundled native Resolve MCP over its stdio protocol."""
import argparse,json,os,selectors,subprocess,time
from pathlib import Path
BINARY='/Applications/DaVinci Resolve/DaVinci Resolve.app/Contents/Applications/ResolveMCP'
p=argparse.ArgumentParser();p.add_argument('tool');p.add_argument('--arguments',type=Path);p.add_argument('--script',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
arguments=json.loads(a.arguments.read_text()) if a.arguments else {}
if a.script:arguments={'script':a.script.read_text(),'timeout':60}
log=Path(__file__).resolve().parents[1]/'reports/publication-20260912/resolve-stderr.log';log.parent.mkdir(parents=True,exist_ok=True)
with log.open('a') as err:
 proc=subprocess.Popen([BINARY],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True,bufsize=1)
 def send(value):proc.stdin.write(json.dumps(value)+'\n');proc.stdin.flush()
 def read(identifier,timeout=65):
  deadline=time.monotonic()+timeout;sel=selectors.DefaultSelector();sel.register(proc.stdout,selectors.EVENT_READ)
  try:
   while time.monotonic()<deadline:
    if not sel.select(max(.01,deadline-time.monotonic())):break
    line=proc.stdout.readline()
    if not line:raise RuntimeError('Native MCP exited')
    value=json.loads(line)
    if value.get('id')==identifier:return value
   raise TimeoutError('Native MCP response timed out')
  finally:sel.close()
 try:
  send({'jsonrpc':'2.0','id':1,'method':'initialize','params':{'protocolVersion':'2024-11-05','capabilities':{},'clientInfo':{'name':'fly-brain-codex-publication','version':'1'}}});init=read(1)
  if 'error' in init:raise RuntimeError(init['error'])
  send({'jsonrpc':'2.0','method':'notifications/initialized'})
  send({'jsonrpc':'2.0','id':2,'method':'tools/call','params':{'name':a.tool,'arguments':arguments}});result=read(2)
  if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2))
  print(json.dumps(result))
 finally:
  proc.stdin.close();proc.terminate()
  try:proc.wait(timeout=3)
  except subprocess.TimeoutExpired:proc.kill();proc.wait()
