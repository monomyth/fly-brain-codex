"""Build a local replay of recorded policy observations and native outcomes."""
import argparse
import json
import os
from pathlib import Path
from urllib.parse import quote


def build(batch, runs=None):
    recordings = []
    runs = runs or [("CPU temporal", batch / "full-temporal-20"), ("MLX temporal", batch / "full-temporal-mlx-20")]
    for label, directory in runs:
        directory = Path(directory).resolve()
        path = directory / "evaluation.json"
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        for index, episode in enumerate(report["episodes"]):
            frames = []
            for folder in sorted((directory / f"episode-{index:03d}").glob("step-*")):
                if not (folder / "decision.json").exists():
                    continue
                observation = json.loads((folder / "observation.json").read_text())
                decision = json.loads((folder / "decision.json").read_text())
                evaluation = decision["evaluation"]
                frames.append({
                    "views": {im["name"]: quote(os.path.relpath(folder / im["file"], batch), safe="/") for im in observation["images"]},
                    "step": decision["step"], "time": observation["simulation_time"],
                    "aperture": observation["gripper_mm"], "contacts": observation["finger_contacts"],
                    "response_bank": decision.get("response_bank","single"), "action": decision["action"], "inference": decision["timings_ms"]["inference"],
                    "held_after": evaluation["held"], "clearance_after": evaluation["clearance_mm"],
                    "tilt_after": evaluation["tilt_deg"], "hold_after": evaluation["hold_seconds"],
                    "floor_limited": decision["floor_limited"],
                })
            recordings.append({"label": label, "index": index, "success": episode["success"],
                               "end": episode.get("error") or episode.get("stop_reason") or ("Pickup, hold and drop completed" if episode.get("release_success") else "Native hold completed"),
                               "release": episode.get("release"), "xy": episode.get("task_for_evaluator_only", {}).get("cube_xy_mm"), "frames": frames})
    payload = json.dumps(recordings).replace("<", "\\u003c")
    page = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fly-brain control replay</title><style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#151a19;color:#ecf1ee;font:16px system-ui;line-height:1.5}main{max-width:1150px;margin:auto;padding:28px}h1{font-size:30px;margin:0 0 8px}p{color:#c0cdc4}select,button{font:inherit;padding:9px 12px;background:#2c3832;color:#fff;border:1px solid #627369;border-radius:6px}select{max-width:100%;flex:1}.controls{display:flex;gap:12px;margin:22px 0 14px}.views{display:grid;grid-template-columns:1fr 1fr;gap:16px}figure{margin:0}figcaption{margin:6px 0;color:#c0cdc4}img{display:block;width:100%;aspect-ratio:1.6;object-fit:contain;background:#090c0a;border-radius:8px}input{width:100%;margin:16px 0;accent-color:#bdd6c6}.readout{display:grid;grid-template-columns:1fr 1fr;gap:16px}.card{background:#242d28;padding:16px;border-radius:8px}h2{font-size:17px;margin:0 0 10px}pre{white-space:pre-wrap;font:14px ui-monospace,monospace;margin:0}.pass{color:#a4e4b5}.fail{color:#ffc89f}small{color:#b5c3ba}@media(max-width:700px){.views,.readout{grid-template-columns:1fr}main{padding:16px}}
</style></head><body><main><h1>Fly-brain control replay</h1>
<p>Recorded trials: 20 mm cube, folded start, Front and Gripper RGB. Success requires a native lift of at least 100 mm, tilt at most 5°, and a stable five-second hold. Pickup and hold use the neural controller. Runs marked “drop completed” then use an explicit gripper-open command and verify a floor landing.</p>
<div class="controls"><select aria-label="Recorded trial" id="trial"></select><button id="play">Play</button></div>
<div id="outcome"></div><p id="release"></p><input id="frame" aria-label="Observation frame" type="range" min="0" value="0">
<div id="timestamp"></div><div class="views"><figure><figcaption>Front · Gemini 336L</figcaption><img id="front" alt="Recorded front camera observation"></figure><figure><figcaption>Gripper · Gemini 305</figcaption><img id="gripper" alt="Recorded gripper camera observation"></figure></div>
<div class="readout"><div class="card"><h2>Observed state and neural command</h2><pre id="command"></pre></div><div class="card"><h2>Native result after this command</h2><pre id="result"></pre></div></div>
<p><small>Images show the observation before the command. The result panel is measured after the command and its settling interval. Playback advances recorded decisions every 500 ms; it is not a real-time video. Cube placement and success metrics are displayed for review and are not actor inputs.</small></p>
</main><script>const recordings=__DATA__;const $=id=>document.getElementById(id);let timer=null;
for(let i=0;i<recordings.length;i++){const r=recordings[i],o=document.createElement('option');o.value=i;o.textContent=`${r.label} · trial ${r.index+1} · ${r.success?'PASS':'FAIL'} · cube [${r.xy}]`; $('trial').appendChild(o)}
function stop(){clearInterval(timer);timer=null;$('play').textContent='Play'}
function draw(){const r=recordings[Number($('trial').value)],f=r.frames[Number($('frame').value)];$('outcome').textContent=(r.success?'PASS: ':'FAIL: ')+r.end;$('outcome').className=r.success?'pass':'fail';$('release').textContent=r.release ? `Release: ${r.release.success?'verified floor landing':'incomplete'} · commanded aperture 90 mm · ${r.release.elapsed_seconds.toFixed(1)} s. Opening is a task-completion command, not a neural decision. The frames below show neural pickup and hold.` : ''; if(!f){$('timestamp').textContent='No completed command was recorded.';for(const id of ['front','gripper'])$(id).removeAttribute('src');$('command').textContent='';$('result').textContent='';return}$('front').src=f.views.Front;$('gripper').src=f.views.Gripper;$('timestamp').textContent=`Step ${f.step+1} / ${r.frames.length} · observed simulation time ${f.time.toFixed(2)} s`;$('command').textContent=`Neural response: ${f.response_bank}\nAperture: ${f.aperture.toFixed(1)} mm\nFinger contacts: ${JSON.stringify(f.contacts)}\n\nJoint targets (degrees):\n${f.action.joints_deg.map(x=>x.toFixed(2)).join(', ')}\nGripper target: ${f.action.gripper_mm.toFixed(1)} mm\nInference: ${f.inference.toFixed(1)} ms`;$('result').textContent=`Held: ${f.held_after}\nCube clearance: ${f.clearance_after.toFixed(1)} mm\nTilt: ${f.tilt_after.toFixed(2)}°\nQualifying hold: ${f.hold_after.toFixed(2)} s\nFloor limited command: ${f.floor_limited}`}
function select(){stop();$('frame').value=0;$('frame').max=Math.max(0,recordings[Number($('trial').value)].frames.length-1);draw()}
$('trial').addEventListener('change',select);$('frame').addEventListener('input',draw);$('play').addEventListener('click',()=>{if(timer){stop();return}if(Number($('frame').value)>=Number($('frame').max))$('frame').value=0;$('play').textContent='Pause';timer=setInterval(()=>{if(Number($('frame').value)>=Number($('frame').max)){stop();return}$('frame').value=Number($('frame').value)+1;draw()},500)});if(recordings.length)select();
</script></body></html>'''.replace("__DATA__", payload)
    destination = batch / "control-replay.html"
    destination.write_text(page)
    print(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--recording", nargs=2, action="append", metavar=("LABEL", "DIRECTORY"))
    args = parser.parse_args()
    build(args.directory.resolve(), args.recording)
